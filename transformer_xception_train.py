import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models
import timm
import os
from PIL import Image
from utils import FDataset, AddGaussianNoise
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from celeb_utils import FDataset_extended, celebDataset
from sklearn.metrics import roc_auc_score

# Classic train parameters
num_epochs = 10
batch_size = 64
learning_rate = 0.0002
output_folder = "transformer_xception_output"
unfreeze_epoch = 0

# Split json file dirs
train_json_list = ["splits/train.json", "splits/test.json", "splits/val.json"]
val_json = None

# Define transform
train_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),  # Convert image to Tensor before applying tensor-specific transforms
    AddGaussianNoise(mean=0.0, std=0.1),
    transforms.RandomApply([transforms.RandomErasing()], p=0.2), # Added RandomCutOut equivalent using RandomErasing
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),  # Convert image to Tensor before applying tensor-specific transforms
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create datasets
data_dir = 'dataset/f++'
train_dataset = FDataset_extended(root_dir = data_dir, json_file_path_list = train_json_list, transform=test_transform)
val_dataset = celebDataset(root_dir='dataset/celeb_test', get_path = False, transform=test_transform)
test_dataset = celebDataset(root_dir='dataset/celeb_test', get_path = False, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size =  batch_size, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)
test_loader = DataLoader(test_dataset, batch_size =  batch_size, shuffle = True)

# Define Model - Using XceptionNet
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class XceptionTransformer(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.xception = timm.create_model('xception', pretrained=True)  # Import XceptionNet from timm
        self.xception.fc = nn.Identity()  # Remove original classifier

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=2048, nhead=8, dim_feedforward=4096, dropout=0.1, batch_first=True),  # Adjust d_model to match Xception output
            num_layers=2
        )
        self.fc = nn.Linear(2048, num_classes)
        
    def forward(self, x):
        x = self.xception(x)  # Extract features (B, 2048)
        x = x.unsqueeze(1)  # (B, 1, 2048) for Transformer
        x = self.transformer(x)  # Transformer processing
        x = x.squeeze(1)  # Back to (B, 2048)
        return self.fc(x)

# Load EfficientNet model from timm library
model = XceptionTransformer(num_classes=2).to(device)

for param in model.xception.parameters():
    param.requires_grad = False

# Loss and Optimizer
class_weights = torch.tensor([1, 0.2], device=device)  # Adjust weights as per class distribution (e.g., 1:4 ratio)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

# Training loop
best_loss = 9999
for epoch in range(num_epochs):
    if epoch == unfreeze_epoch:
        print("Unfreezing all layers for training.")
        for param in model.xception.parameters():
            param.requires_grad = True
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(train_loader, desc=f'Training [{epoch + 1}/{num_epochs}]:'):
        images, labels = images.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward + backward + optimize
        outputs = model(images)
        loss = criterion(outputs, labels.to(torch.long))
        loss.backward()
        optimizer.step()
        
        # Calculate statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    #scheduler.step()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

    # Validation phase
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    val_labels = []
    val_preds = []
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f'Evaluating [{epoch + 1}/{num_epochs}]:'):
            images, labels = images.to(device), labels.to(device, dtype=torch.long)
            outputs = model(images)
            loss = criterion(outputs, labels.to(torch.long))
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            val_labels.extend(labels.cpu().numpy())
            val_preds.extend(outputs.softmax(dim=1)[:, 1].cpu().numpy())
    
    val_epoch_loss = val_loss / len(val_loader)
    val_epoch_acc = 100 * val_correct / val_total
    val_roc_auc = roc_auc_score(val_labels, val_preds)
    print(f"Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_acc:.2f}%, Validation ROC AUC: {val_roc_auc:.4f}")

    # Save model
    if val_epoch_loss < best_loss:
        best_loss = val_epoch_loss
        os.makedirs(output_folder, exist_ok=True)
        save_path = os.path.join(output_folder, f'model_epoch{epoch+1}.pth')
        torch.save(model.state_dict(), save_path)
        print('Model Saved!')

print('Training complete.')

# Final test with test data
model.eval()
test_loss = 0.0
test_correct = 0
test_total = 0
test_labels = []
test_preds = []
with torch.no_grad():
    for images, labels in tqdm(test_loader, desc='Testing:'):
        images, labels = images.to(device), labels.to(device, dtype=torch.long)
        outputs = model(images)
        loss = criterion(outputs.float(), labels)  # Reshape outputs to match labels
        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()
        test_labels.extend(labels.cpu().numpy())
        test_preds.extend(outputs.softmax(dim=1)[:, 1].cpu().numpy())

test_epoch_loss = test_loss / len(test_loader)
test_epoch_acc = 100 * test_correct / test_total
test_roc_auc = roc_auc_score(test_labels, test_preds)
print(f"Test Loss: {test_epoch_loss:.4f}, Test Accuracy: {test_epoch_acc:.2f}%, Test ROC AUC: {test_roc_auc:.4f}")

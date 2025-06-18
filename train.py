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

# Classic train parameters
num_epochs = 18
batch_size = 32
learning_rate = 0.0002
output_folder = "output_unweight"
unfreeze_epoch = 3

# Split json file dirs
train_json = "splits/train.json"
val_json = "splits/val.json"
test_json = "splits/test.json"

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
data_dir = '30_frames'
train_dataset = FDataset(root_dir = data_dir, json_file_path = train_json, transform=test_transform)
val_dataset = FDataset(root_dir = data_dir, json_file_path = val_json, transform=test_transform)
test_dataset = FDataset(root_dir = data_dir, json_file_path = test_json, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size =  batch_size, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)
test_loader = DataLoader(test_dataset, batch_size =  batch_size, shuffle = True)

# Define Model - Using XceptionNet
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load XceptionNet model from timm library
model = timm.create_model('xception', pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # Assuming binary classification
model = model.to(device)

# Freeze all layers except the classifier
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the final fully connected layer
for param in model.fc.parameters():
    param.requires_grad = True

# Loss and Optimizer
class_weights = torch.tensor([1.0, 4.0], device=device)  # Adjust weights as per class distribution (e.g., 1:4 ratio)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

# Training loop
best_loss = 9999
for epoch in range(num_epochs):
    if epoch == unfreeze_epoch:
        print("Unfreezing all layers for training.")
        for param in model.parameters():
            param.requires_grad = True
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(train_loader, desc=f'Training [{epoch + 1}/{num_epochs}]:'):
        images, labels = images.to(device), labels.to(device, dtype=torch.long)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward + backward + optimize
        outputs = model(images)
        loss = criterion(outputs, labels)
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
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f'Evaluating [{epoch + 1}/{num_epochs}]:'):
            images, labels = images.to(device), labels.to(device, dtype=torch.long)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    val_epoch_loss = val_loss / len(val_loader)
    val_epoch_acc = 100 * val_correct / val_total
    print(f"Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_acc:.2f}%")

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
with torch.no_grad():
    for images, labels in tqdm(test_loader, desc='Testing:'):
        images, labels = images.to(device), labels.to(device, dtype=torch.long)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

test_epoch_loss = test_loss / len(test_loader)
test_epoch_acc = 100 * test_correct / test_total
print(f"Test Loss: {test_epoch_loss:.4f}, Test Accuracy: {test_epoch_acc:.2f}%")

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models
import timm
import os
import random
from PIL import Image
from utils import FDataset, AddGaussianNoise
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from celeb_utils import FDataset_extended, celebDataset
from sklearn.metrics import roc_auc_score

class MixStyle(nn.Module):
    """MixStyle (applied on the channels of normalized feature maps).
    Reference:
      Zhou et al. Domain Generalization with MixStyle. ICLR 2021.
    """
    def __init__(self, p=0.5, alpha=0.1, eps=1e-6):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha

    def forward(self, x):
        if not self.training:
            return x

        if random.random() > self.p:
            return x

        B = x.size(0)

        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True, unbiased=False)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu) / sig

        # Generate random indices for mixing
        perm = torch.randperm(B).to(x.device)
        
        # Sample from beta distribution
        lmda = self.beta.sample((B, 1, 1, 1)).to(x.device)
        
        # Get mu, sig of other instances
        mu2, sig2 = mu[perm], sig[perm]
        
        # Mix the normalized features with the style of other instances
        mu_mix = mu * lmda + mu2 * (1 - lmda)
        sig_mix = sig * lmda + sig2 * (1 - lmda)
        
        return x_normed * sig_mix + mu_mix

class SwinWithMixStyle(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.model = timm.create_model('swin_small_patch4_window7_224', pretrained=True, num_classes=num_classes)
        
        # Create two MixStyle modules for two different layers
        self.mixstyle1 = MixStyle(p=0.5, alpha=0.1)
        self.mixstyle2 = MixStyle(p=0.5, alpha=0.1)
        
        # Hook to apply MixStyle
        self.hooks = []
        self._register_hooks()
        
    def _register_hooks(self):
        def hook_fn1(module, input, output):
            return self.mixstyle1(output)
            
        def hook_fn2(module, input, output):
            return self.mixstyle2(output)
        
        # Register hooks after the first and second Swin blocks
        self.hooks.append(self.model.layers[1].register_forward_hook(hook_fn1))
        self.hooks.append(self.model.layers[2].register_forward_hook(hook_fn2))
        
    def forward(self, x):
        return self.model(x)

    def __del__(self):
        # Clean up hooks when the model is deleted
        for hook in self.hooks:
            hook.remove()

# Classic train parameters
num_epochs = 10
batch_size = 64
learning_rate = 1e-5
output_folder = "swin_mix_style_output"
unfreeze_epoch = 0

# Split json file dirs
train_json_list = ["splits/train.json", "splits/test.json", "splits/val.json"]

# Define transform
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),  # Convert image to Tensor before applying tensor-specific transforms
    AddGaussianNoise(mean=0.0, std=0.1),
    transforms.RandomApply([transforms.RandomErasing()], p=0.2), # Added RandomCutOut equivalent using RandomErasing
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # Convert image to Tensor before applying tensor-specific transforms
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create datasets
data_dir = 'dataset/f++'
test_data_dir = 'dataset/celeb_test'
train_dataset = FDataset_extended(root_dir = data_dir, json_file_path_list = train_json_list, transform=test_transform)
val_dataset = celebDataset(root_dir= test_data_dir, get_path = False, transform=test_transform)
test_dataset = celebDataset(root_dir= test_data_dir, get_path = False, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size =  batch_size, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)
test_loader = DataLoader(test_dataset, batch_size =  batch_size, shuffle = True)

# Define Model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize model with MixStyle
model = SwinWithMixStyle(num_classes=2)
model = model.to(device)

# Freeze all layers except the classifier
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the final head and last Swin block
for param in model.model.head.parameters():
    param.requires_grad = True
for param in model.model.layers[-1].parameters():
    param.requires_grad = True

# Loss and Optimizer
class_weights = torch.tensor([1, 0.2], device=device)  # Adjust weights as per class distribution (e.g., 1:4 ratio)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

#scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
best_roc = 0
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
    all_train_labels = []
    all_train_preds = []
    
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
        
        # Collect labels and predictions for ROC AUC calculation
        all_train_labels.extend(labels.cpu().numpy())
        all_train_preds.extend(outputs.detach().cpu().numpy()[:, 1])  # Use probabilities for class 1
        
    #scheduler.step()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    train_roc_auc = roc_auc_score(all_train_labels, all_train_preds, average='macro')
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%, ROC AUC: {train_roc_auc:.4f}")

    # Validation phase
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    all_val_labels = []
    all_val_preds = []
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f'Evaluating [{epoch + 1}/{num_epochs}]:'):
            images, labels = images.to(device), labels.to(device, dtype=torch.long)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            
            # Collect labels and predictions for ROC AUC calculation
            all_val_labels.extend(labels.cpu().numpy())
            all_val_preds.extend(outputs.detach().cpu().numpy()[:, 1])  # Use probabilities for class 1
    
    val_epoch_loss = val_loss / len(val_loader)
    val_epoch_acc = 100 * val_correct / val_total
    val_roc_auc = roc_auc_score(all_val_labels, all_val_preds, average='macro')
    print(f"Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_acc:.2f}%, ROC AUC: {val_roc_auc:.4f}")

    # Save model
    if val_roc_auc > best_roc:
        best_roc = val_roc_auc
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
all_labels = []
all_preds = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc='Testing:'):
        images, labels = images.to(device), labels.to(device, dtype=torch.long)
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Track test loss
        test_loss += loss.item()
        
        # Get predictions and track correct predictions
        _, predicted = torch.max(outputs, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()
        
        # Collect labels and predictions for ROC AUC calculation
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(outputs.cpu().numpy()[:, 1])  # Use probabilities for class 1

# Calculate test loss and accuracy
test_epoch_loss = test_loss / len(test_loader)
test_epoch_acc = 100 * test_correct / test_total

# Calculate ROC AUC score (for multi-class, you will need to handle this differently)
# If binary classification, use `roc_auc_score(all_labels, all_preds[:, 1])`
roc_auc = roc_auc_score(all_labels, all_preds, average='macro')

# Print results
print(f"Test Loss: {test_epoch_loss:.4f}, Test Accuracy: {test_epoch_acc:.2f}%")
print(f"ROC AUC Score: {roc_auc:.4f}")
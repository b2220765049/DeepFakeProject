import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision import models
import timm
import os
from PIL import Image
from utils import AddGaussianNoise
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from celeb_utils import celebDataset
from sklearn.metrics import roc_auc_score
import pywt  # Import PyWavelets for Haar wavelet decomposition
import json
import glob

# Classic train parameters
num_epochs = 10
batch_size = 64
learning_rate = 1e-5
output_folder = "swin_normal_wavelet"  
unfreeze_epoch = 0

# Split json file dirs
train_json_list = ["splits/train.json", "splits/test.json", "splits/val.json"]

# Define transforms
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
])

test_image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
])

def normalize_4ch(tensor):
    # Normalize RGB channels
    tensor[:3] = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                    std=[0.229, 0.224, 0.225])(tensor[:3])
    # Normalize residual channel
    tensor[3:] = transforms.Normalize(mean=[0.5], std=[0.5])(tensor[3:])
    return tensor

# Haar‐wavelet residual channel added.
# This script augments the Swin-S model to process 4-channel input (R, G, B + Haar-wavelet residual).
# The residual is computed using Haar wavelet decomposition and added as a 4th channel.

class FDataset_extended(Dataset):
    def __init__(self, root_dir, json_file_path_list, type_name=None, get_path=False, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.get_path = get_path
        class_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir)]
        class_dirs = [d for d in class_dirs if os.path.isdir(d)]

        json_data = []
        for json_file_path in json_file_path_list:
            with open(json_file_path, 'r') as f:
                current_data = json.load(f)
                json_data += sum(current_data, [])

        for label, class_dir in enumerate(class_dirs):
            print(label, class_dir)
            type_folders = os.listdir(class_dir)
            for type_folder in type_folders:
                if type_name:
                    if label == 1 and type_folder != type_name:
                        continue
                type_folder_path = os.path.join(class_dir, type_folder)
                image_folders = os.listdir(type_folder_path)
                for image_folder in image_folders:
                    folder_name = image_folder.split('_')[0] if '_' in image_folder else image_folder
                    if folder_name not in json_data:
                        continue
                    image_folder_path = os.path.join(type_folder_path, image_folder)
                    image_paths = [os.path.join(image_folder_path, image) for image in os.listdir(image_folder_path) if image.endswith('.png')]
                    labels = [1 - label for _ in range(len(image_paths))]
                    if len(image_paths) < 30:
                        print(f"Bad Folder: {image_folder_path}")
                        continue
                    self.image_paths += image_paths
                    self.labels += labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        img = Image.open(image_path).convert('RGB')

        # Apply PIL transforms before converting to tensor
        if self.transform:
            img = self.transform(img)

        # Convert RGB to tensor
        rgb = transforms.ToTensor()(img)

        # Convert to grayscale
        gray = transforms.Grayscale()(img)
        gray = transforms.ToTensor()(gray)

        # Apply Haar wavelet decomposition
        coeffs2 = pywt.dwt2(gray.squeeze(0).numpy(), 'haar')
        LL, (LH, HL, HH) = coeffs2

        # Reconstruct low-frequency component
        LL_up = pywt.idwt2((LL, (None, None, None)), 'haar')
        LL_up = torch.tensor(LL_up, dtype=torch.float32)

        # Ensure dimensions match between gray and LL_up
        if LL_up.shape != gray.shape[1:]:
            min_height = min(LL_up.shape[0], gray.shape[1])
            min_width = min(LL_up.shape[1], gray.shape[2])
            LL_up = LL_up[:min_height, :min_width]
            gray = gray[:, :min_height, :min_width]

        # Compute residual
        residual = gray - LL_up.unsqueeze(0)
        residual = torch.clamp(residual, 0, 1)  # Normalize to [0,1]

        # Stack residual as 4th channel
        image_4ch = torch.cat([rgb, residual], dim=0)

        # Apply normalization to the 4-channel tensor
        image_4ch = normalize_4ch(image_4ch)

        if not self.get_path:
            return image_4ch, torch.tensor(label, dtype=torch.float32)
        else:
            return image_4ch, torch.tensor(label, dtype=torch.float32), image_path

class celebDataset(Dataset):
    def __init__(self, root_dir, get_path=False, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.get_path = get_path
        class_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir)]
        class_dirs = [d for d in class_dirs if os.path.isdir(d)]

        for class_dir in class_dirs:
            label = 0 if 'original' in class_dir else 1
            print(f"{class_dir}/**/*.png")
            images = glob.glob(f"{class_dir}/**/*.png") 
            self.image_paths += images
            self.labels += [label for _ in range(len(images))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        img = Image.open(image_path).convert('RGB')

        # Apply transforms to the PIL image before converting to tensor
        if self.transform:
            img = test_image_transforms(img)

        # Convert RGB to tensor
        rgb = transforms.ToTensor()(img)

        # Convert to grayscale
        gray = transforms.Grayscale()(img)
        gray = transforms.ToTensor()(gray)

        # Apply Haar wavelet decomposition
        coeffs2 = pywt.dwt2(gray.squeeze(0).numpy(), 'haar')
        LL, (LH, HL, HH) = coeffs2

        # Reconstruct low-frequency component
        LL_up = pywt.idwt2((LL, (None, None, None)), 'haar')
        LL_up = torch.tensor(LL_up, dtype=torch.float32)

        # Ensure dimensions match between gray and LL_up
        if LL_up.shape != gray.shape[1:]:
            min_height = min(LL_up.shape[0], gray.shape[1])
            min_width = min(LL_up.shape[1], gray.shape[2])
            LL_up = LL_up[:min_height, :min_width]
            gray = gray[:, :min_height, :min_width]

        # Compute residual
        residual = gray - LL_up.unsqueeze(0)
        residual = torch.clamp(residual, 0, 1)  # Normalize to [0,1]

        # Stack residual as 4th channel
        image_4ch = torch.cat([rgb, residual], dim=0)

        # Apply normalization to the 4-channel tensor
        image_4ch = normalize_4ch(image_4ch)

        if not self.get_path:
            return image_4ch, torch.tensor(label, dtype=torch.float32)
        else:
            return image_4ch, torch.tensor(label, dtype=torch.float32), image_path

# Create datasets
data_dir = 'dataset/f++'
test_data_dir = 'dataset/celeb_test'
train_dataset = FDataset_extended(root_dir = data_dir, json_file_path_list = train_json_list, transform=image_transforms)
val_dataset = celebDataset(root_dir= test_data_dir, get_path = False, transform=test_image_transforms)
test_dataset = celebDataset(root_dir= test_data_dir, get_path = False, transform=test_image_transforms)

train_loader = DataLoader(train_dataset, batch_size =  batch_size, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)
test_loader = DataLoader(test_dataset, batch_size =  batch_size, shuffle = True)

# Define Model - Using Swin-S with 4-channel input
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = timm.create_model('swin_small_patch4_window7_224', pretrained=True, num_classes=2, in_chans=4)
model = model.to(device)

# Freeze all layers except the classifier
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the final fully connected layer
for param in model.head.parameters():
    param.requires_grad = True

# Loss and Optimizer
class_weights = torch.tensor([1, 0.2], device=device)  # Adjust weights as per class distribution (e.g., 1:4 ratio)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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
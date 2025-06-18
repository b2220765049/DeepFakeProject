import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models
import timm
import os
from PIL import Image
from utils import FDataset_ensemble, AddGaussianNoise
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

# Classic train parameters
num_epochs = 0
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

test_transform2 = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),  # Convert image to Tensor before applying tensor-specific transforms
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transform3 = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create datasets
data_dir = 'cropped_faces'
train_dataset = FDataset_ensemble(root_dir = data_dir, json_file_path = train_json, transform=test_transform, transform2=test_transform2, transform3=test_transform3)
val_dataset = FDataset_ensemble(root_dir = data_dir, json_file_path = val_json, transform=test_transform, transform2=test_transform2, transform3=test_transform3)
test_dataset = FDataset_ensemble(root_dir = data_dir, json_file_path = test_json, transform=test_transform, transform2=test_transform2, transform3=test_transform3)

train_loader = DataLoader(train_dataset, batch_size =  batch_size, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)
test_loader = DataLoader(test_dataset, batch_size =  batch_size, shuffle = True)

# Define Model - Using XceptionNet
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load XceptionNet model from timm library
model1 = timm.create_model('xception', pretrained=True)
num_ftrs = model1.fc.in_features
model1.fc = nn.Linear(num_ftrs, 2)  # Assuming binary classification
model1 = model1.to(device)

# Freeze all layers except the classifier
for param in model1.parameters():
    param.requires_grad = False

# Load the state dict for model1 (make sure to replace 'path_to_model1.pth' with your actual path)
model1.load_state_dict(torch.load('output/model_epoch9.pth'))

# Load EfficientNet model
model2 = models.efficientnet_b0(pretrained=True)
num_ftrs = model2.classifier[1].in_features
model2.fc = nn.Linear(num_ftrs, 2)  # Assuming binary classification
model2 = model2.to(device)

# Freeze all layers except the classifier
for param in model2.parameters():
    param.requires_grad = False

# Load the state dict for model2 (make sure to replace 'path_to_model2.pth' with your actual path)
model2.load_state_dict(torch.load('zeynep.pth'))

# Load ResNet50 model
model3 = models.resnet50(pretrained=True)
num_ftrs = model3.fc.in_features
model3.fc = nn.Linear(num_ftrs, 2)  
model3 = model3.to(device)

# Freeze all layers
for param in model3.parameters():
    param.requires_grad = False

# Load the state dict for model3 (make sure to replace 'path_to_model3.pth' with your actual path)
model3.load_state_dict(torch.load('sena.pth'))





ensemble_layer = nn.Linear(6, 2)
ensemble_layer = ensemble_layer.to(device)
ensemble_layer.load_state_dict(torch.load('output_unweight/ensemble_layer_epoch4.pth'))

# Loss and Optimizer
class_weights = torch.tensor([1.0, 4.0], device=device)  # Adjust weights as per class distribution (e.g., 1:4 ratio)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(ensemble_layer.parameters(), lr=learning_rate)

#scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
model1.eval()
model2.eval()
model3.eval()
# Training loop
# Training loop
best_loss = 9999
for epoch in range(num_epochs):            
    ensemble_layer.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images1, images2, images3, labels in tqdm(train_loader, desc=f'Training [{epoch + 1}/{num_epochs}]:'):
        images1, images2, images3, labels = images1.to(device), images2.to(device), images3.to(device), labels.to(device, dtype=torch.long)
        
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass through individual models
        with torch.no_grad():  # Models 1, 2, and 3 are frozen, so no need to calculate gradients
            outputs1 = model1(images1)
            outputs2 = model2(images2)[:,:2]
            outputs3 = model3(images3)
        # Concatenate the outputs from the models
        ensemble_input = torch.cat((outputs1, outputs2, outputs3), dim=1)  # Concatenate along feature dimension

        # Forward pass through the ensemble layer
        outputs = ensemble_layer(ensemble_input)

        # Compute loss and backpropagate
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Calculate statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

    # Validation phase
    ensemble_layer.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images1, images2, images3, labels in tqdm(val_loader, desc=f'Training [{epoch + 1}/{num_epochs}]:'):
            images1, images2, images3, labels = images1.to(device), images2.to(device), images3.to(device), labels.to(device, dtype=torch.long)

            # Forward pass through individual models
            outputs1 = model1(images1)
            outputs2 = model2(images2)[:,:2]
            outputs3 = model3(images3)


            # Concatenate the outputs from the models
            ensemble_input = torch.cat((outputs1, outputs2, outputs3), dim=1)

            # Forward pass through the ensemble layer
            outputs = ensemble_layer(ensemble_input)

            # Compute loss
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
        save_path = os.path.join(output_folder, f'ensemble_layer_epoch{epoch+1}.pth')
        torch.save(ensemble_layer.state_dict(), save_path)
        print('Model Saved!')

print('Training complete.')
y_true = []
y_pred = []
y_prob = []
# Final test with test data
ensemble_layer.eval()
test_loss = 0.0
test_correct = 0
test_total = 0
with torch.no_grad():
    for images1, images2, images3, labels in tqdm(test_loader, desc=f'Testing:'):
        images1, images2, images3, labels = images1.to(device), images2.to(device), images3.to(device), labels.to(device, dtype=torch.long)
        # Forward pass through individual models
        outputs1 = model1(images1)
        outputs2 = model2(images2)[:,:2]
        outputs3 = model3(images3)

        # Concatenate the outputs from the models
        ensemble_input = torch.cat((outputs1, outputs2, outputs3), dim=1)

        # Forward pass through the ensemble layer
        outputs = ensemble_layer(ensemble_input)
        probabilities = torch.softmax(outputs, dim=1)[:, 1]  # Get probabilities for class 1
        # Compute loss
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()
        
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())
        y_prob.extend(probabilities.cpu().numpy())  # Store class 1 probabilities

# Calculate final test accuracy and loss
test_epoch_loss = test_loss / len(test_loader)
test_epoch_acc = 100 * test_correct / test_total
print(f"Test Loss: {test_epoch_loss:.4f}, Test Accuracy: {test_epoch_acc:.2f}%")

y_true = 1 - np.array(y_true)
y_pred = 1 - np.array(y_pred)
y_prob = 1 - np.array(y_prob)
# Generate and display the confusion matrix
cm = confusion_matrix(y_true, y_pred)
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Class 0", "Class 1"])

plt.figure(figsize=(8, 6))
cm_display.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("Confusion Matrix")
plt.show()

# Calculate ROC Curve and AUC
fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC AUC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()


import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import timm
import os
from utils import FDataset, celebDataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

# Load test parameters
data_dir = '30_frames'
test_json = "splits/test.json"
output_folder = "celeb_output"
model_checkpoint = "model_epoch4.pth"
batch_size = 32

# Define test transforms
test_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),  # Convert image to Tensor before applying tensor-specific transforms
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load test dataset
# test_dataset = FDataset(root_dir=data_dir, json_file_path=test_json, type_name=None, get_path = True, transform=test_transform)
test_dataset = celebDataset(root_dir='dataset/celeb_test', get_path = True, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define Model - Using XceptionNet
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load XceptionNet model from timm library
model = timm.create_model('xception', pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # Assuming binary classification
model = model.to(device)

# Load model checkpoint
checkpoint_path = os.path.join(output_folder, model_checkpoint)
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Loaded model from {checkpoint_path}")
else:
    raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

# Set model to evaluation mode
model.eval()

# After collecting predictions and true labels
y_true = []
y_pred = []
y_prob = []  # To store the probabilities for ROC curve
image_paths = []
criterion = nn.CrossEntropyLoss()
test_loss = 0.0
test_correct = 0
test_total = 0

import random

with torch.no_grad():
    for images, labels, paths in tqdm(test_loader, desc='Testing:'):
        images, labels = images.to(device), labels.to(device, dtype=torch.long)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.softmax(outputs, dim=1)[:, 1]  # Get probabilities for class 1

        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

        # Collect predictions and labels for confusion matrix and ROC AUC
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())
        y_prob.extend(probabilities.cpu().numpy())  # Store class 1 probabilities
        image_paths.extend(paths)  # Store class 1 probabilities
count = 0
for path, pred, true in zip(image_paths, y_pred, y_true):
    if (pred != true):
        print(path)
        count+=1
print(count)

# Calculate final test accuracy and loss
test_epoch_loss = test_loss / len(test_loader)
test_epoch_acc = 100 * test_correct / test_total
print(f"Test Loss: {test_epoch_loss:.4f}, Test Accuracy: {test_epoch_acc:.2f}%")

# Generate and display the confusion matrix
cm = confusion_matrix(y_true, y_pred)
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Class 0", "Class 1"])

cm_display.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("Confusion Matrix")
plt.show()

y_true = 1 - np.array(y_true)
y_pred = 1 - np.array(y_pred)
y_prob = 1 - np.array(y_prob)

# Calculate ROC Curve and AUC
fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)
print(f"ROC AUC Score: {roc_auc:.4f}")
# # Plot ROC AUC Curve
# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, color='blue', lw=2, label=f"ROC curve (AUC = {roc_auc})")
# plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("Receiver Operating Characteristic (ROC) Curve")
# plt.legend(loc="lower right")
# plt.grid(alpha=0.3)
# plt.show()
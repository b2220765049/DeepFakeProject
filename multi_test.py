import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import timm
import os
from utils import FDataset, FDatasetTest
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

# Load test parameters
data_dir = '30_frames'
test_json = "splits/test.json"
output_folder = "output_unweight"
model_checkpoint = "model_epoch6.pth"
batch_size = 32

# Define test transform
test_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),  # Convert image to Tensor before applying tensor-specific transforms
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load test dataset
test_dataset = FDatasetTest(root_dir=data_dir, json_file_path=test_json, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define Model - Using XceptionNet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load XceptionNet model from timm library
model = timm.create_model('xception', pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # Assuming binary classification
model = model.to(device)

# Load model checkpoint
checkpoint_path = os.path.join(output_folder, model_checkpoint)
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path))
    print(f"Loaded model from {checkpoint_path}")
else:
    raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

# Set model to evaluation mode
model.eval()

# After collecting predictions and true labels
y_true = []
y_pred = []
y_prob = []  # To store the probabilities for ROC curve

type_predictions = {0:[],
                    1:[],
                    2:[],
                    3:[],
                    4:[]}

type_probs = {0:[],
            1:[],
            2:[],
            3:[],
            4:[]}

criterion = nn.CrossEntropyLoss()
test_loss = 0.0
test_correct = 0
test_total = 0

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc='Testing:'):
        images, labels = images.to(device), labels.to(device, dtype=torch.long)
        real_labels = torch.where(labels > 1, torch.tensor(1, device=device), labels)
        real_labels = torch.tensor(1, device=device) - real_labels
        outputs = model(images)
        loss = criterion(outputs, real_labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.softmax(outputs, dim=1)[:, 1]  # Get probabilities for class 1

        test_total += labels.size(0)
        test_correct += (predicted == real_labels).sum().item()

        # Collect predictions and labels for confusion matrix and ROC AUC
        for label, predict, prob in zip(labels, predicted, probabilities):
            type_predictions[int(label)].append(int(predict))
            type_probs[int(label)].append(float(prob))

# Calculate final test accuracy and loss
test_epoch_loss = test_loss / len(test_loader)
test_epoch_acc = 100 * test_correct / test_total
print(f"Test Loss: {test_epoch_loss:.4f}, Test Accuracy: {test_epoch_acc:.2f}%")
roc_auc_scores = []

def evaluate(y_true, y_pred, y_prob):
    # Generate and display the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Class 0", "Class 1"])

    cm_display.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title("Confusion Matrix")
    plt.show()

    # Calculate ROC Curve and AUC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    roc_auc_scores.append(roc_auc)

    # Plot ROC AUC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()

org_preds = type_predictions[0]
org_probs = type_probs[0]

for type_index in range(1,5):
    type_preds= type_predictions[type_index]
    type_probs2 = type_probs[type_index]

    y_true = [1 for _ in org_preds] + [0 for _ in type_preds]
    y_pred = org_preds + type_preds
    y_prob = org_probs + type_probs2

    y_true = 1 - np.array(y_true)
    y_pred = 1 - np.array(y_pred)
    y_prob = 1 - np.array(y_prob)
    
    evaluate(y_true, y_pred, y_prob)

print(roc_auc_scores)
print("Mean ROC AUC: ", sum(roc_auc_scores)/ len(roc_auc_scores))
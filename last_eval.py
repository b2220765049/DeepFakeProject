import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import timm
import os
from celeb_utils import celebDataset
from utils import AddGaussianNoise
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Parameters
model_path = "vit16/model_epoch5.pth"  # Replace X with actual epoch number
batch_size = 64
test_data_dir = 'dataset/celeb_test'

# Define test transform
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Dataset and DataLoader
test_dataset = celebDataset(root_dir=test_data_dir, get_path=True, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load model
model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=2)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# Loss function (must match training)
class_weights = torch.tensor([1, 0.2], device=device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Test loop
test_loss = 0.0
test_correct = 0
test_total = 0
all_labels = []
all_preds = []
video_predictions = defaultdict(list)  # Dictionary to store predictions grouped by video

with torch.no_grad():
    for images, labels, paths in tqdm(test_loader, desc='Testing:'):
        images, labels = images.to(device), labels.to(device, dtype=torch.long)
        outputs = model(images)
        loss = criterion(outputs, labels)

        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

        # Group predictions by video
        for i, path in enumerate(paths):
            video_name = str(Path(path).parent)  # Remove the last part of the path to get the video name
            video_predictions[video_name].append(predicted[i].item())

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(outputs.cpu().numpy()[:, 1])  # Probabilities for class 1

# Perform majority voting for each video
video_classifications = {}
for video, predictions in video_predictions.items():
    majority_class = max(set(predictions), key=predictions.count)  # Majority vote
    video_classifications[video] = majority_class

# Print video classifications
print("Video Classifications:")
for video, classification in video_classifications.items():
    print(f"{video}: Classified as {classification}")

# Final metrics
test_epoch_loss = test_loss / len(test_loader)
test_epoch_acc = 100 * test_correct / test_total
roc_auc = roc_auc_score(all_labels, all_preds, average='macro')

print(f"Test Loss: {test_epoch_loss:.4f}")
print(f"Test Accuracy: {test_epoch_acc:.2f}%")
print(f"ROC AUC Score: {roc_auc:.4f}")

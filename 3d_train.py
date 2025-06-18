import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.video import r3d_18, mc3_18
from utils import FDataset3D
from tqdm import tqdm
import os
import copy
from celeb_utils import celeb_Dataset3D
from sklearn.metrics import roc_auc_score

data_dir = "dataset/f++"
num_epochs = 10
output_folder = '3d_model_again'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
])

# Move dataset initialization outside of any loops or functions to ensure it is loaded only once
train_dataset = FDataset3D(root_dir=data_dir,
                           json_file_path_list=["splits/train.json", "splits/val.json", "splits/test.json"],
                           num_frames=30,
                           transform=transform)

val_dataset = celeb_Dataset3D(root_dir='dataset/celeb_test',
                               json_file_path="splits/test.json",
                               num_frames=30,
                               transform=transform)

test_dataset = celeb_Dataset3D(root_dir='dataset/celeb_test',
                                json_file_path="splits/test.json",
                                num_frames=30,
                                transform=transform)

# Create DataLoaders once and reuse them
train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True, num_workers=12)
val_loader = DataLoader(val_dataset, batch_size=6, shuffle=False, num_workers=12)
test_loader = DataLoader(test_dataset, batch_size=6, shuffle=False, num_workers=12)

if __name__ == '__main__':
    # Model
    model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
    model.blocks[-1].proj = nn.Linear(model.blocks[-1].proj.in_features, 2)
    model = model.to(device)

    for name, param in model.named_parameters():
        if "proj" not in name:
            param.requires_grad = False

    class_weights = torch.tensor([1, 0.2], device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)

    best_accuracy = 0
    # Training
    for epoch in range(num_epochs):
        if epoch == 0:
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(model.parameters(), lr=1e-5)
        model.train()
        total_loss = 0
        all_labels = []
        all_preds = []
        for imgs, labels in tqdm(train_loader, desc=f"Training Epoch[{epoch}/{num_epochs}]"):
            imgs, labels = imgs.to(device), labels.to(device).long()  # Convert labels to Long type

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)  # Remove .float() as labels should be integers for CrossEntropyLoss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(outputs.detach().cpu().numpy()[:, 1])  # Use probabilities for class 1

        avg_train_loss = total_loss / len(train_loader)
        train_roc_auc = roc_auc_score(all_labels, all_preds)
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Train ROC AUC={train_roc_auc:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        all_val_labels = []
        all_val_preds = []
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Validation Epoch[{epoch}/{num_epochs}]"):
                imgs, labels = imgs.to(device), labels.to(device).long()
                outputs = model(imgs)
                outputs = outputs.view(-1, 2)  # Ensure outputs have the correct shape for CrossEntropyLoss
                loss = criterion(outputs, labels)  # Remove .float() as labels should be integers for CrossEntropyLoss

                val_loss += loss.item()
                preds = torch.sigmoid(outputs)[:, 1]  # Use probabilities for class 1
                all_val_labels.extend(labels.cpu().numpy())
                all_val_preds.extend(preds.cpu().numpy())
                correct += (preds.round() == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_roc_auc = roc_auc_score(all_val_labels, all_val_preds)
        accuracy = correct / len(val_dataset)
        if accuracy > best_accuracy:
            best_model = copy.deepcopy(model)
            best_accuracy = accuracy
            os.makedirs(output_folder, exist_ok=True)
            save_path = os.path.join(output_folder, f'3d_model_r50_epoch{epoch+1}.pth')
            torch.save(model.state_dict(), save_path)
            print('Model Saved!')
        print(f"Validation Loss={avg_val_loss:.4f}, Accuracy={accuracy:.2f}, Validation ROC AUC={val_roc_auc:.4f}")

    # Validation
    best_model.eval()
    val_loss = 0
    correct = 0
    all_test_labels = []
    all_test_preds = []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device).long()
            outputs = model(imgs)
            outputs = outputs.view(-1, 2)  # Ensure outputs have the correct shape for CrossEntropyLoss
            loss = criterion(outputs, labels)  # Remove .float() as labels should be integers for CrossEntropyLoss

            val_loss += loss.item()
            preds = torch.sigmoid(outputs)[:, 1]  # Use probabilities for class 1
            all_val_labels.extend(labels.cpu().numpy())
            all_val_preds.extend(preds.cpu().numpy())
            correct += (preds.round() == labels).sum().item()


    avg_val_loss = val_loss / len(val_loader)
    test_roc_auc = roc_auc_score(all_test_labels, all_test_preds)
    accuracy = correct / len(val_dataset)
    print(f"Validation Loss={avg_val_loss:.4f}, Accuracy={accuracy:.2f}, Test ROC AUC={test_roc_auc:.4f}")

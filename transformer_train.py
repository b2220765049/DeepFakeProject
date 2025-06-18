import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import os
from utils import FDataset3D
from tqdm import tqdm
import copy
from celeb_utils import celeb_Dataset3D
from sklearn.metrics import roc_auc_score
import numpy as np

def main():
    data_dir = "dataset/f++"

    # ------------- Model: 3D ResNet + Transformer -------------
    class ResNet3DTransformer(nn.Module):
        def __init__(self, num_classes=2):
            super().__init__()
            self.resnet3d = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)  # Load 3D ResNet50
            self.resnet3d.blocks[-1].proj = nn.Identity()  # Remove original classifier

            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=2048, nhead=8, dim_feedforward=4096, dropout=0.1, batch_first=True),  # Set batch_first=True
                num_layers=2
            )
            self.fc = nn.Linear(2048, num_classes)
            
        def forward(self, x):
            x = self.resnet3d(x)  # Extract features (B, 2048)
            x = x.unsqueeze(1)  # (B, 1, 2048) for Transformer
            x = self.transformer(x)  # Transformer processing
            x = x.squeeze(1)  # Back to (B, 2048)
            return self.fc(x)
        
    # ------------- Training Function -------------
    def train(model, dataloader, criterion, optimizer, device):
        model.train()
        total_loss = 0
        all_labels = []
        all_probs = []
        for videos, labels in tqdm(dataloader, desc="Training"):
            videos, labels = videos.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy())
        
        roc_auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else float('nan')
        accuracy = (np.array(all_labels) == (np.array(all_probs) > 0.5)).mean()
        return total_loss / len(dataloader), accuracy, roc_auc

    # ------------- Validation & Test Function -------------
    def evaluate(model, dataloader, criterion, device):
        model.eval()
        total_loss, correct, total = 0, 0, 0
        all_labels = []
        all_probs = []
        with torch.no_grad():
            for videos, labels in tqdm(dataloader, desc="Evaluating"):
                videos, labels = videos.to(device), labels.to(device)
                outputs = model(videos)
                loss = criterion(outputs, labels.long())

                total_loss += loss.item()
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)

                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())

        roc_auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else float('nan')
        accuracy = (np.array(all_labels) == (np.array(all_probs) > 0.5)).mean()
        return total_loss / len(dataloader), accuracy, roc_auc

    # ------------- Main Script -------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = ResNet3DTransformer(num_classes=2).to(device)

    # Transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
    ])

    # Dataset & Dataloader
    train_dataset = FDataset3D(root_dir=data_dir, json_file_path_list=["splits/train.json", "splits/val.json", "splits/test.json"], num_frames=30, transform=transform)
    val_dataset = celeb_Dataset3D(root_dir='dataset/celeb_test', json_file_path="splits/test.json", num_frames=30, transform=transform)
    test_dataset = celeb_Dataset3D(root_dir='dataset/celeb_test', json_file_path="splits/test.json", num_frames=30, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=6, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=6, shuffle=False, num_workers=4)

    for param in model.resnet3d.parameters():
        param.requires_grad = False

    # Loss & Optimizer
    class_weights = torch.tensor([0.1, 0.2]).cuda()  # Adjust values based on the distribution
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)

    # Training Loop
    best_val_roc_auc = 0.0
    epochs = 10
    for epoch in range(epochs):
        if epoch == 0:
            for param in model.resnet3d.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(model.parameters(), lr=1e-5)
        train_loss, train_acc, train_roc_auc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_roc_auc = evaluate(model, test_loader, criterion, device)
        print(f"Epoch {epoch+1}/{epochs}\nTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.2%}, Train ROC AUC: {train_roc_auc:.4f}\nVal Loss: {val_loss:.4f}, Val Acc: {val_acc:.2%}, Val ROC AUC: {val_roc_auc:.4f}")

        # Save best model
        if val_roc_auc > best_val_roc_auc:
            best_val_roc_auc = val_roc_auc
            torch.save(model.state_dict(), "best_resnet3d_transformer.pth")
            print(f"Best model saved with Val ROC AUC: {best_val_roc_auc:.4f}")

    # Final Test Evaluation
    test_loss, test_acc, test_roc_auc = evaluate(model, test_loader, criterion, device)
    print("Celeb Dataset Evaluation:")
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2%}, Test ROC AUC: {test_roc_auc:.4f}")

if __name__ == '__main__':
    main()
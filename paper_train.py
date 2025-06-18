import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import timm
from celeb_utils import celebDataset, FDataset_extended
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import random

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, x1, x2, y):
        # Euclidean distance
        diff = x1 - x2
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)
        
        # All genuine pairs should have distance close to 0
        # All fake pairs should have distance greater than margin
        loss = (1-y) * torch.pow(dist, 2) + y * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2)
        loss = torch.mean(loss)
        return loss

class PairwiseInteractionModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # Xception backbone for both images
        self.backbone = timm.create_model('xception', pretrained=True)
        self.backbone.fc = nn.Identity()
        
        # MLP1: dimension reduction (2048->128)
        self.mlp1 = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )
        
        # MLP for combining features
        self.combine_mlp = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )
        
        # MLP2 for interaction features
        self.mlp2 = nn.Sequential(
            nn.Linear(128 * 128, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256)
        )
        
        self.classifier = nn.Linear(256, num_classes)
    
    def extract_features(self, x):
        features = self.backbone(x)
        features = F.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)
        return self.mlp1(features)
    
    def forward(self, x1, x2):
        # Extract features from both images
        feat1 = self.extract_features(x1)  # [B, 128]
        feat2 = self.extract_features(x2)  # [B, 128]
        
        # Normalize features
        feat1 = F.normalize(feat1, p=2, dim=1)
        feat2 = F.normalize(feat2, p=2, dim=1)
        
        # Combine features
        combined = torch.cat([feat1, feat2], dim=1)  # [B, 256]
        mutual_vector = self.combine_mlp(combined)  # [B, 128]
        
        # Compute interactions
        B = mutual_vector.size(0)
        interactions = torch.bmm(
            mutual_vector.unsqueeze(2),
            mutual_vector.unsqueeze(1)
        )  # [B, 128, 128]
        
        # Process interactions
        interactions = interactions.view(B, -1)
        interactions = self.mlp2(interactions)
        
        # Classification
        logits = self.classifier(interactions)
        
        return logits, mutual_vector, feat1, feat2

class PairDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, transform=None):
        self.base_dataset = base_dataset
        self.transform = transform
        self.indices = list(range(len(base_dataset)))
        
    def __getitem__(self, idx):
        # Get first image
        img1, label1 = self.base_dataset[idx]
        
        # For the second image:
        # If real (label=0), get another real image
        # If fake (label=1), get a real image to compare against
        if label1 == 1:  # Fake
            # Find a real image
            real_indices = [i for i, (_, l) in enumerate(self.base_dataset) if l == 0]
            idx2 = random.choice(real_indices)
        else:  # Real
            # Find another real image
            real_indices = [i for i, (_, l) in enumerate(self.base_dataset) if l == 0 and i != idx]
            idx2 = random.choice(real_indices)
        
        img2, _ = self.base_dataset[idx2]
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return (img1, img2), label1
    
    def __len__(self):
        return len(self.base_dataset)

# Training parameters
num_epochs = 10
batch_size = 16
learning_rate = 1e-4
output_folder = "pairwise_model_output"

# Data transforms
train_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def train_epoch(model, train_loader, contrastive_criterion, ce_criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    
    for (images1, images2), labels in tqdm(train_loader, desc='Training'):
        images1, images2, labels = images1.to(device), images2.to(device), labels.to(device, dtype=torch.long)
        
        optimizer.zero_grad()
        logits, mutual_vec, feat1, feat2 = model(images1, images2)
        
        # Contrastive loss between feature pairs
        contrastive_loss = contrastive_criterion(feat1, feat2, labels)
        
        # Classification loss
        ce_loss = ce_criterion(logits, labels)
        
        # Combined loss (λ=0.5 as mentioned in paper)
        loss = ce_loss + 0.5 * contrastive_loss
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(logits.softmax(dim=1)[:, 1].detach().cpu().numpy())
    
    epoch_loss = running_loss / len(train_loader)
    epoch_auc = roc_auc_score(all_labels, all_preds)
    return epoch_loss, epoch_auc

def evaluate(model, val_loader, contrastive_criterion, ce_criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for (images1, images2), labels in tqdm(val_loader, desc='Evaluating'):
            images1, images2, labels = images1.to(device), images2.to(device), labels.to(device, dtype=torch.long)
            logits, mutual_vec, feat1, feat2 = model(images1, images2)
            
            contrastive_loss = contrastive_criterion(feat1, feat2, labels)
            ce_loss = ce_criterion(logits, labels)
            loss = ce_loss + 0.5 * contrastive_loss
            
            running_loss += loss.item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(logits.softmax(dim=1)[:, 1].detach().cpu().numpy())
    
    val_loss = running_loss / len(val_loader)
    val_auc = roc_auc_score(all_labels, all_preds)
    return val_loss, val_auc

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Wrap datasets with PairDataset
    train_dataset = PairDataset(
        FDataset_extended(
            root_dir='dataset/f++',
            json_file_path_list=["splits/train.json", "splits/val.json", "splits/test.json"],
            transform=None
        ),
        transform=train_transform
    )
    
    val_dataset = PairDataset(
        celebDataset(
            root_dir='dataset/celeb_test',
            get_path=False,
            transform=None
        ),
        transform=test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model, criteria, and optimizer
    model = PairwiseInteractionModel().to(device)
    contrastive_criterion = ContrastiveLoss(margin=1.0)
    ce_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    best_val_auc = 0.0
    
    # Training loop
    for epoch in range(num_epochs):
        train_loss, train_auc = train_epoch(model, train_loader, contrastive_criterion, ce_criterion, optimizer, device)
        val_loss, val_auc = evaluate(model, val_loader, contrastive_criterion, ce_criterion, device)
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
        print(f"Current LR: {scheduler.get_last_lr()[0]}")
        
        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), f"{output_folder}/best_model.pth")
            print(f"Saved new best model with validation AUC: {best_val_auc:.4f}")

if __name__ == '__main__':
    main()
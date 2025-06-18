import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from timm import create_model
from celeb_utils import celebDataset
from sklearn.metrics import roc_auc_score
import numpy as np
from tqdm import tqdm

# Define the XceptionTransformer model
class XceptionTransformer(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.xception = create_model('xception', pretrained=False)  # Set pretrained=False for loading weights later
        self.xception.fc = nn.Identity()  # Remove original classifier

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=2048, nhead=8, dim_feedforward=4096, dropout=0.1, batch_first=True),
            num_layers=2
        )
        self.fc = nn.Linear(2048, num_classes)
        
    def forward(self, x):
        x = self.xception(x)  # Extract features (B, 2048)
        x = x.unsqueeze(1)  # (B, 1, 2048) for Transformer
        x = self.transformer(x)  # Transformer processing
        x = x.squeeze(1)  # Back to (B, 2048)
        return self.fc(x)

# Evaluation function
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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the model
    model = XceptionTransformer(num_classes=2).to(device)
    model.load_state_dict(torch.load("best_xception_transformer.pth", map_location=device))
    print("Model loaded successfully.")

    # Define test dataset and dataloader
    test_transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_dataset = celebDataset(root_dir='dataset/celeb_test', get_path=False, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=80, shuffle=False)

    # Define loss function
    class_weights = torch.tensor([0.8, 0.2]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Evaluate the model
    test_loss, test_acc, test_roc_auc = evaluate(model, test_loader, criterion, device)
    print("Celeb Dataset Evaluation:")
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2%}, Test ROC AUC: {test_roc_auc:.4f}")

if __name__ == '__main__':
    main()

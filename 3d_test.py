import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import timm
import os
from utils import FDataset, celebDataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from celeb_utils import FDataset_extended, celebDataset, celeb_Dataset3D

if __name__ == '__main__':
    # Load test parameters
    data_dir = 'dataset/celeb_test'
    test_json = "splits/test.json"
    output_folder = "30_3d_output"
    model_checkpoint = "3d_model_r50_epoch5.pth"
    batch_size = 32

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
    ])

    test_dataset = celeb_Dataset3D(root_dir= data_dir,
                           num_frames=30,
                           transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)

    # Define Model - Using XceptionNet
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load XceptionNet model from timm library
    model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
    model.blocks[-1].proj = nn.Linear(model.blocks[-1].proj.in_features, 1)
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
    # Evaluation Setup
    y_true = []
    y_pred = []
    y_scores = []

    # Disable gradient calculation for inference
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device).float()
            
            # Forward pass
            outputs = model(inputs).squeeze(1)  # Assuming the model outputs logits
            
            # Convert logits to probabilities using sigmoid
            probs = torch.sigmoid(outputs)
            
            # Store true labels and predictions
            y_true.extend(labels.cpu().numpy())
            y_scores.extend(probs.cpu().numpy())
            y_pred.extend((probs > 0.5).cpu().numpy())  # Convert probs to binary predictions

    # Convert lists to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)

    # Calculate Accuracy
    accuracy = np.mean(y_pred == y_true)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Compute ROC-AUC Score
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    print(f"ROC-AUC Score: {roc_auc:.4f}")

    # # Plot ROC Curve
    # plt.figure(figsize=(8, 6))
    # plt.plot(fpr, tpr, color="blue", label=f"ROC curve (AUC = {roc_auc:.4f})")
    # plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.title("ROC Curve")
    # plt.legend()
    # plt.grid()
    # plt.show()

    # # Display Confusion Matrix
    # cm = confusion_matrix(y_true, y_pred)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    # disp.plot(cmap=plt.cm.Blues)
    # plt.title("Confusion Matrix")
    # plt.show()
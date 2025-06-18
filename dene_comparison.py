import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from utils import FDataset
from celeb_utils import celebDataset
from dene import DeepFakeDetector
import timm
import json

class XceptionModel(nn.Module):
    """Xception model for comparison"""
    def __init__(self, num_classes=2):
        super().__init__()
        self.model = timm.create_model('xception', pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
    
    def forward(self, x):
        return self.model(x)

class ResNetModel(nn.Module):
    """ResNet model for comparison"""
    def __init__(self, num_classes=2):
        super().__init__()
        self.model = timm.create_model('resnet50', pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
    
    def forward(self, x):
        return self.model(x)

def load_pretrained_model(model_path, model_type='xception'):
    """Load a pretrained model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_type == 'xception':
        model = XceptionModel()
    elif model_type == 'resnet':
        model = ResNetModel()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded {model_type} model from {model_path}")
    else:
        print(f"Model not found at {model_path}, using pretrained weights")
    
    model = model.to(device)
    model.eval()
    return model

def evaluate_model(model, data_loader, model_name, device):
    """Evaluate a model and return metrics"""
    print(f"Evaluating {model_name}...")
    
    predictions = []
    true_labels = []
    probabilities = []
    
    with torch.no_grad():
        for images, batch_labels in tqdm(data_loader, desc=f'Evaluating {model_name}'):
            images, batch_labels = images.to(device), batch_labels.to(device)
            
            if isinstance(model, DeepFakeDetector):
                # Dene model uses different prediction method
                for i in range(len(images)):
                    pred, prob = model.predict(images[i])
                    predictions.append(pred)
                    true_labels.append(batch_labels[i].item())
                    probabilities.append(prob[1])
            else:
                # Standard PyTorch model
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(batch_labels.cpu().numpy())
                probabilities.extend(probs[:, 1].cpu().numpy())
    
    accuracy = accuracy_score(true_labels, predictions)
    auc = roc_auc_score(true_labels, probabilities)
    
    return {
        'accuracy': accuracy,
        'auc': auc,
        'predictions': predictions,
        'true_labels': true_labels,
        'probabilities': probabilities
    }

def plot_comparison_results(results, dataset_name, save_path):
    """Plot comparison results"""
    models = list(results.keys())
    accuracies = [results[model]['accuracy'] for model in models]
    aucs = [results[model]['auc'] for model in models]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy comparison
    bars1 = ax1.bar(models, accuracies, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
    ax1.set_title(f'{dataset_name} - Accuracy Comparison')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    for bar, acc in zip(bars1, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.3f}', ha='center', va='bottom')
    
    # AUC comparison
    bars2 = ax2.bar(models, aucs, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
    ax2.set_title(f'{dataset_name} - ROC AUC Comparison')
    ax2.set_ylabel('ROC AUC')
    ax2.set_ylim(0, 1)
    for bar, auc in zip(bars2, aucs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{auc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("DeepFake Detection Model Comparison")
    print("="*50)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Dataset paths
    fpp_data_dir = "dataset/f++"
    celeb_data_dir = "dataset/celeb_test"
    output_folder = 'dene_comparison_results'
    os.makedirs(output_folder, exist_ok=True)
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    print("Loading datasets...")
    
    # F++ test dataset
    fpp_test_dataset = FDataset(
        root_dir=fpp_data_dir,
        json_file_path="splits/test.json",
        transform=transform
    )
    
    # CelebDF dataset
    celeb_dataset = celebDataset(
        root_dir=celeb_data_dir,
        get_path=False,
        transform=transform
    )
    
    # Create data loaders with full datasets
    fpp_loader = DataLoader(fpp_test_dataset, batch_size=16, shuffle=False, num_workers=4)
    celeb_loader = DataLoader(celeb_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    print(f"F++ Test samples: {len(fpp_test_dataset)}")
    print(f"CelebDF samples: {len(celeb_dataset)}")
    
    # Load models
    models = {}
    
    # 1. Dene model (if available)
    dene_model_path = "dene_outputs/deepfake_detector_fpp_celebdf.pkl"
    if os.path.exists(dene_model_path):
        dene_model = DeepFakeDetector()
        dene_model.load_model(dene_model_path)
        models['Dene (F++ + CelebDF)'] = dene_model
        print("Loaded Dene model")
    else:
        print("Dene model not found, skipping...")
    
    # 2. Xception model (if available)
    xception_model_path = "xception_again/model_epoch10.pth"
    if os.path.exists(xception_model_path):
        xception_model = load_pretrained_model(xception_model_path, 'xception')
        models['Xception'] = xception_model
        print("Loaded Xception model")
    else:
        print("Xception model not found, skipping...")
    
    # 3. ResNet model (if available)
    resnet_model_path = "best_resnet3d_transformer.pth"
    if os.path.exists(resnet_model_path):
        resnet_model = load_pretrained_model(resnet_model_path, 'resnet')
        models['ResNet'] = resnet_model
        print("Loaded ResNet model")
    else:
        print("ResNet model not found, skipping...")
    
    # 4. Transformer model (if available)
    transformer_model_path = "best_xception_transformer.pth"
    if os.path.exists(transformer_model_path):
        transformer_model = load_pretrained_model(transformer_model_path, 'xception')
        models['Transformer'] = transformer_model
        print("Loaded Transformer model")
    else:
        print("Transformer model not found, skipping...")
    
    if not models:
        print("No models found for comparison!")
        return
    
    # Evaluate on F++ dataset
    print("\n" + "="*50)
    print("EVALUATING ON F++ DATASET")
    print("="*50)
    
    fpp_results = {}
    for model_name, model in models.items():
        fpp_results[model_name] = evaluate_model(model, fpp_loader, model_name, device)
    
    # Evaluate on CelebDF dataset
    print("\n" + "="*50)
    print("EVALUATING ON CELEBDF DATASET")
    print("="*50)
    
    celeb_results = {}
    for model_name, model in models.items():
        celeb_results[model_name] = evaluate_model(model, celeb_loader, model_name, device)
    
    # Generate comparison plots
    print("\nGenerating comparison plots...")
    plot_comparison_results(fpp_results, "F++ Dataset", 
                           os.path.join(output_folder, 'fpp_comparison.png'))
    plot_comparison_results(celeb_results, "CelebDF Dataset", 
                           os.path.join(output_folder, 'celebdf_comparison.png'))
    
    # Print summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    print(f"{'Model':<25} {'F++ Accuracy':<15} {'F++ AUC':<12} {'CelebDF Accuracy':<18} {'CelebDF AUC':<12}")
    print("-" * 100)
    
    for model_name in models.keys():
        fpp_acc = fpp_results[model_name]['accuracy']
        fpp_auc = fpp_results[model_name]['auc']
        celeb_acc = celeb_results[model_name]['accuracy']
        celeb_auc = celeb_results[model_name]['auc']
        
        print(f"{model_name:<25} {fpp_acc:<15.4f} {fpp_auc:<12.4f} {celeb_acc:<18.4f} {celeb_auc:<12.4f}")
    
    # Save detailed results
    comparison_results = {
        'fpp_results': {name: {'accuracy': float(result['accuracy']), 'auc': float(result['auc'])} 
                       for name, result in fpp_results.items()},
        'celebdf_results': {name: {'accuracy': float(result['accuracy']), 'auc': float(result['auc'])} 
                           for name, result in celeb_results.items()}
    }
    
    with open(os.path.join(output_folder, 'comparison_results.json'), 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    print(f"\nDetailed results saved to {output_folder}/")
    print("Comparison completed!")

if __name__ == "__main__":
    main() 
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from utils import FDataset
from celeb_utils import FDataset_extended, celebDataset
from dene import DeepFakeDetector

def evaluate_model_on_dataset(model, data_loader, dataset_name):
    """Evaluate model on a specific dataset and return detailed metrics"""
    print(f"\nEvaluating on {dataset_name}...")
    
    predictions = []
    true_labels = []
    probabilities = []
    
    for images, batch_labels in tqdm(data_loader, desc=f'Evaluating {dataset_name}'):
        for i in range(len(images)):
            pred, prob = model.predict(images[i])
            predictions.append(pred)
            true_labels.append(batch_labels[i].item())
            probabilities.append(prob[1])  # Probability of fake class
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    auc = roc_auc_score(true_labels, probabilities)
    conf_matrix = confusion_matrix(true_labels, predictions)
    
    # Classification report
    report = classification_report(true_labels, predictions, 
                                 target_names=['Real', 'Fake'], 
                                 output_dict=True)
    
    print(f"{dataset_name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {auc:.4f}")
    print(f"Precision (Fake): {report['Fake']['precision']:.4f}")
    print(f"Recall (Fake): {report['Fake']['recall']:.4f}")
    print(f"F1-Score (Fake): {report['Fake']['f1-score']:.4f}")
    
    return {
        'accuracy': accuracy,
        'auc': auc,
        'confusion_matrix': conf_matrix,
        'classification_report': report,
        'predictions': predictions,
        'true_labels': true_labels,
        'probabilities': probabilities
    }

def plot_confusion_matrix(conf_matrix, title, save_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'])
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curve(true_labels, probabilities, title, save_path):
    """Plot and save ROC curve"""
    from sklearn.metrics import roc_curve
    
    fpr, tpr, _ = roc_curve(true_labels, probabilities)
    auc = roc_auc_score(true_labels, probabilities)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("Comprehensive DeepFake Detection Model Evaluation")
    
    # Load the trained model
    model_path = "dene_outputs/deepfake_detector_fpp_celebdf.pkl"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please train the model first.")
        return
    
    detector = DeepFakeDetector()
    detector.load_model(model_path)
    print("Model loaded successfully!")
    
    # Dataset paths
    fpp_data_dir = "dataset/f++"
    celeb_data_dir = "dataset/celeb_test"
    output_folder = 'dene_evaluation_results'
    os.makedirs(output_folder, exist_ok=True)
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    print("Loading datasets...")
    
    # F++ datasets (separate splits)
    fpp_train_dataset = FDataset(
        root_dir=fpp_data_dir,
        json_file_path="splits/train.json",
        transform=transform
    )
    
    fpp_val_dataset = FDataset(
        root_dir=fpp_data_dir,
        json_file_path="splits/val.json",
        transform=transform
    )
    
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
    fpp_train_loader = DataLoader(fpp_train_dataset, batch_size=16, shuffle=False, num_workers=4)
    fpp_val_loader = DataLoader(fpp_val_dataset, batch_size=16, shuffle=False, num_workers=4)
    fpp_test_loader = DataLoader(fpp_test_dataset, batch_size=16, shuffle=False, num_workers=4)
    celeb_loader = DataLoader(celeb_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    print(f"F++ Train samples: {len(fpp_train_dataset)}")
    print(f"F++ Val samples: {len(fpp_val_dataset)}")
    print(f"F++ Test samples: {len(fpp_test_dataset)}")
    print(f"CelebDF samples: {len(celeb_dataset)}")
    
    # Evaluate on all datasets
    results = {}
    
    # F++ evaluation
    results['fpp_train'] = evaluate_model_on_dataset(detector, fpp_train_loader, "F++ Train")
    results['fpp_val'] = evaluate_model_on_dataset(detector, fpp_val_loader, "F++ Validation")
    results['fpp_test'] = evaluate_model_on_dataset(detector, fpp_test_loader, "F++ Test")
    
    # CelebDF evaluation
    results['celebdf'] = evaluate_model_on_dataset(detector, celeb_loader, "CelebDF")
    
    # Generate plots
    print("\nGenerating evaluation plots...")
    
    # Confusion matrices
    plot_confusion_matrix(results['fpp_test']['confusion_matrix'], 
                         'F++ Test Confusion Matrix',
                         os.path.join(output_folder, 'fpp_test_confusion_matrix.png'))
    
    plot_confusion_matrix(results['celebdf']['confusion_matrix'], 
                         'CelebDF Confusion Matrix',
                         os.path.join(output_folder, 'celebdf_confusion_matrix.png'))
    
    # ROC curves
    plot_roc_curve(results['fpp_test']['true_labels'], 
                  results['fpp_test']['probabilities'],
                  'F++ Test ROC Curve',
                  os.path.join(output_folder, 'fpp_test_roc_curve.png'))
    
    plot_roc_curve(results['celebdf']['true_labels'], 
                  results['celebdf']['probabilities'],
                  'CelebDF ROC Curve',
                  os.path.join(output_folder, 'celebdf_roc_curve.png'))
    
    # Summary report
    print("\n" + "="*60)
    print("SUMMARY REPORT")
    print("="*60)
    
    summary_data = []
    for dataset_name, result in results.items():
        summary_data.append({
            'Dataset': dataset_name.replace('_', ' ').title(),
            'Accuracy': f"{result['accuracy']:.4f}",
            'ROC AUC': f"{result['auc']:.4f}",
            'Precision (Fake)': f"{result['classification_report']['Fake']['precision']:.4f}",
            'Recall (Fake)': f"{result['classification_report']['Fake']['recall']:.4f}",
            'F1-Score (Fake)': f"{result['classification_report']['Fake']['f1-score']:.4f}"
        })
    
    # Print summary table
    print(f"{'Dataset':<20} {'Accuracy':<10} {'ROC AUC':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 80)
    for row in summary_data:
        print(f"{row['Dataset']:<20} {row['Accuracy']:<10} {row['ROC AUC']:<10} {row['Precision (Fake)']:<10} {row['Recall (Fake)']:<10} {row['F1-Score (Fake)']:<10}")
    
    # Save detailed results
    import json
    detailed_results = {}
    for dataset_name, result in results.items():
        detailed_results[dataset_name] = {
            'accuracy': float(result['accuracy']),
            'auc': float(result['auc']),
            'classification_report': result['classification_report']
        }
    
    with open(os.path.join(output_folder, 'detailed_results.json'), 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"\nDetailed results saved to {output_folder}/")
    print("Evaluation completed!")

if __name__ == "__main__":
    main() 
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
import timm
from utils import FDataset
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import numpy as np
import os
from celeb_utils import celebDataset, celeb_Dataset3D
from tqdm import tqdm
from PIL import Image, ImageEnhance, ImageFilter

def load_model(model_name, checkpoint_path, device):
    if model_name == 'vit':
        model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=2)
    elif model_name == 'swin':
        model = timm.create_model('swin_small_patch4_window7_224', pretrained=False, num_classes=2)
    elif model_name == 'convnext':
        model = timm.create_model('convnext_small', pretrained=False, num_classes=2)
    elif model_name == 'slow_r50':
        model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
        model.blocks[-1].proj = nn.Linear(model.blocks[-1].proj.in_features, 2)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    return model

def apply_downsample_upsample(img, downsample_size=(200, 200)):
    """Downsample and then upsample back to original size"""
    if isinstance(img, torch.Tensor):
        img = TF.to_pil_image(img)
    
    # Downsample
    img_down = img.resize(downsample_size, Image.Resampling.BILINEAR)
    # Upsample back to original size
    img_up = img_down.resize((224, 224), Image.Resampling.BILINEAR)
    
    return img_up

def apply_blur(img, radius=0.2):
    """Apply Gaussian blur"""
    if isinstance(img, torch.Tensor):
        img = TF.to_pil_image(img)
    
    return img.filter(ImageFilter.GaussianBlur(radius=radius))

def apply_sharpen(img, factor=0.2):
    """Apply sharpening"""
    if isinstance(img, torch.Tensor):
        img = TF.to_pil_image(img)
    
    enhancer = ImageEnhance.Sharpness(img)
    return enhancer.enhance(factor)

def apply_salt_and_pepper(img, prob=0.001):
    """Apply salt and pepper noise"""
    if isinstance(img, torch.Tensor):
        img = TF.to_pil_image(img)
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Create salt and pepper noise mask
    noise = np.random.random(img_array.shape[:2])
    
    # Salt (white) noise
    salt = noise > 1 - prob/2
    # Pepper (black) noise
    pepper = noise < prob/2
    
    # Apply salt noise
    img_array[salt] = [255, 255, 255]
    # Apply pepper noise
    img_array[pepper] = [0, 0, 0]
    
    return Image.fromarray(img_array)

def evaluate_model_with_perturbation(model, test_loader, device, perturbation_fn=None, transform=None, is_3d=False):
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Evaluating"):
            if is_3d:
                videos, labels = data
                if perturbation_fn is not None:
                    # Apply perturbation to each frame in each video
                    perturbed_videos = []
                    for video in videos:  # video shape: [C, T, H, W]
                        perturbed_frames = []
                        for t in range(video.shape[1]):  # Iterate over frames
                            frame = video[:, t, :, :]  # Get single frame
                            # Convert to PIL, apply perturbation, convert back
                            frame_pil = TF.to_pil_image(frame)
                            frame_perturbed = perturbation_fn(frame_pil)
                            # Just convert back to tensor without normalization
                            frame_tensor = TF.to_tensor(frame_perturbed)
                            perturbed_frames.append(frame_tensor)
                        # Stack frames back to video
                        video_tensor = torch.stack(perturbed_frames, dim=1)  # [C, T, H, W]
                        perturbed_videos.append(video_tensor)
                    videos = torch.stack(perturbed_videos, dim=0)
                
                videos = videos.to(device)
                outputs = model(videos)
            else:
                images, labels = data
                if perturbation_fn is not None:
                    # Apply perturbation to each image in the batch
                    perturbed_images = []
                    for img in images:
                        # Convert to PIL, apply perturbation, and convert back to tensor
                        img_pil = TF.to_pil_image(img)
                        img_perturbed = perturbation_fn(img_pil)
                        # Just convert back to tensor without normalization
                        img_tensor = TF.to_tensor(img_perturbed)
                        perturbed_images.append(img_tensor)
                    images = torch.stack(perturbed_images)
                
                images = images.to(device)
                outputs = model(images)
            
            # Handle both CrossEntropyLoss and BCELoss cases
            if outputs.shape[1] == 2:  # CrossEntropyLoss case
                probs = torch.softmax(outputs, dim=1)[:, 1]
            else:  # BCELoss case
                probs = torch.sigmoid(outputs).squeeze()
                
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    auc_score = roc_auc_score(all_labels, all_preds)
    predictions = [1 if p >= 0.5 else 0 for p in all_preds]
    cm = confusion_matrix(all_labels, predictions)
    
    return auc_score, cm, all_preds, all_labels

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data transforms
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform_3d = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
    ])

    # Load test datasets
    test_dataset = celebDataset(root_dir='dataset/celeb_test', get_path=False, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    test_dataset_3d = celeb_Dataset3D(
        root_dir='dataset/celeb_test',
        num_frames=30,
        transform=test_transform_3d
    )
    test_loader_3d = DataLoader(test_dataset_3d, batch_size=4, shuffle=False, num_workers=4)

    # Model configurations
    model_configs = [
        # {
        #     'name': 'slow_r50',
        #     'path': '3d_model_again/3d_model_r50_epoch4.pth',
        #     'title': 'Slow-R50',
        #     'is_3d': True
        # },
        # {
        #     'name': 'vit',
        #     'path': 'vit16/model_epoch5.pth',
        #     'title': 'ViT-16',
        #     'is_3d': False
        # },
        # {
        #     'name': 'swin',
        #     'path': 'swin_normal_again/model_epoch8.pth',
        #     'title': 'Swin-S',
        #     'is_3d': False
        # },
        {
            'name': 'convnext',
            'path': 'convnexts_output/model_epoch10.pth',
            'title': 'ConvNeXt-S',
            'is_3d': False
        }
    ]

    # Perturbation configurations
    perturbation_configs = [
        # {
        #     'name': 'Original',
        #     'fn': None
        # },
        # {
        #     'name': 'Downsample-Upsample',
        #     'fn': lambda x: apply_downsample_upsample(x, (200, 200))
        # },
        # {
        #     'name': 'Blur',
        #     'fn': lambda x: apply_blur(x, radius=0.2)
        # },
        # {
        #     'name': 'Sharpen',
        #     'fn': lambda x: apply_sharpen(x, factor=0.2)
        # },
        # {
        #     'name': 'Salt & Pepper',
        #     'fn': lambda x: apply_salt_and_pepper(x, prob=0.001)
        # },

        {
            'name': 'Downsample-Upsample',
            'fn': lambda x: apply_downsample_upsample(x, (168, 168))
        },
        {
            'name': 'Blur',
            'fn': lambda x: apply_blur(x, radius=0.5)
        },
        {
            'name': 'Sharpen',
            'fn': lambda x: apply_sharpen(x, factor=0.5)
        },
        {
            'name': 'Salt & Pepper',
            'fn': lambda x: apply_salt_and_pepper(x, prob=0.01)
        },
        
        {
            'name': 'Downsample-Upsample',
            'fn': lambda x: apply_downsample_upsample(x, (112, 112))
        },
        {
            'name': 'Blur',
            'fn': lambda x: apply_blur(x, radius=2)
        },
        {
            'name': 'Sharpen',
            'fn': lambda x: apply_sharpen(x, factor=2)
        },
        {
            'name': 'Salt & Pepper',
            'fn': lambda x: apply_salt_and_pepper(x, prob=0.03)
        },
    ]

    # Results dictionary to store all evaluations
    all_results = {}

    # Evaluate each model with each perturbation
    for model_config in model_configs:
        print(f"\nLoading {model_config['title']}...")
        model = load_model(model_config['name'], model_config['path'], device)
        model_results = {}

        for pert_config in perturbation_configs:
            print(f"\nEvaluating {model_config['title']} with {pert_config['name']}...")
            auc_score, cm, preds, labels = evaluate_model_with_perturbation(
                model, 
                test_loader_3d if model_config['is_3d'] else test_loader, 
                device, 
                pert_config['fn'],
                test_transform_3d if model_config['is_3d'] else test_transform,
                model_config['is_3d']
            )
            
            model_results[pert_config['name']] = {
                'auc': auc_score,
                'cm': cm,
                'preds': preds,
                'labels': labels
            }
            
            print(f"{model_config['title']} - {pert_config['name']} - AUC: {auc_score:.4f}")
        
        all_results[model_config['title']] = model_results

    # Print summary table
    print("\nRobustness Summary (AUC Scores):")
    print("-" * 100)
    headers = ['Model', 'Original', 'Downsample', 'Blur', 'Sharpen', 'Salt & Pepper']
    print(f"{headers[0]:<15} | " + " | ".join(f"{h:<12}" for h in headers[1:]))
    print("-" * 100)
    
    for model_name in all_results:
        scores = [
            all_results[model_name]['Original']['auc'],
            all_results[model_name]['Downsample-Upsample']['auc'],
            all_results[model_name]['Blur']['auc'],
            all_results[model_name]['Sharpen']['auc'],
            all_results[model_name]['Salt & Pepper']['auc']
        ]
        print(f"{model_name:<15} | {scores[0]:<12.4f} | {scores[1]:<12.4f} | {scores[2]:<12.4f} | {scores[3]:<12.4f} | {scores[4]:<12.4f}")

if __name__ == '__main__':
    main()

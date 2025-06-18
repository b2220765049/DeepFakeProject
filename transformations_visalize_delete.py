import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from celeb_utils import celebDataset
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import random
import os

def apply_downsample_upsample(img, downsample_size=(112, 112)):
    """Downsample and then upsample back to original size"""
    if isinstance(img, torch.Tensor):
        img = TF.to_pil_image(img)
    
    # Downsample
    img_down = img.resize(downsample_size, Image.Resampling.BILINEAR)
    # Upsample back to original size
    img_up = img_down.resize((224, 224), Image.Resampling.BILINEAR)
    
    return img_up

def apply_blur(img, radius=2.0):
    """Apply Gaussian blur"""
    if isinstance(img, torch.Tensor):
        img = TF.to_pil_image(img)
    
    return img.filter(ImageFilter.GaussianBlur(radius=radius))

def apply_sharpen(img, factor=2.0):
    """Apply sharpening"""
    if isinstance(img, torch.Tensor):
        img = TF.to_pil_image(img)
    
    enhancer = ImageEnhance.Sharpness(img)
    return enhancer.enhance(factor)

def apply_salt_and_pepper(img, prob=0.03):
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

def visualize_transformations():
    # Data transforms (only resize and convert to tensor, no normalization for visualization)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Load dataset
    dataset = celebDataset(root_dir='dataset/celeb_test', get_path=False, transform=transform)
    
    # Get random indices without repetition
    num_samples = 3
    total_samples = len(dataset)
    random_indices = random.sample(range(total_samples), num_samples)
    
    # Create figure with 5 columns (original + 4 perturbations)
    fig, axes = plt.subplots(num_samples, 5, figsize=(25, 5*num_samples))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    # Process each random sample
    for i, idx in enumerate(random_indices):
        img, label = dataset[idx]
        
        # Convert tensor to PIL for visualization
        img_pil = TF.to_pil_image(img)
        
        # Original
        axes[i, 0].imshow(np.array(img_pil))
        axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')
        
        # Downsample-Upsample
        img_down_up = apply_downsample_upsample(img_pil)
        axes[i, 1].imshow(np.array(img_down_up))
        axes[i, 1].set_title('Downsample-Upsample')
        axes[i, 1].axis('off')
        
        # Blur
        img_blur = apply_blur(img_pil)
        axes[i, 2].imshow(np.array(img_blur))
        axes[i, 2].set_title('Blur')
        axes[i, 2].axis('off')
        
        # Sharpen
        img_sharp = apply_sharpen(img_pil)
        axes[i, 3].imshow(np.array(img_sharp))
        axes[i, 3].set_title('Sharpen')
        axes[i, 3].axis('off')
        
        # Salt and Pepper Noise
        img_noisy = apply_salt_and_pepper(img_pil)
        axes[i, 4].imshow(np.array(img_noisy))
        axes[i, 4].set_title('Salt & Pepper Noise')
        axes[i, 4].axis('off')

        # Save individual images
        os.makedirs('perturbation_samples', exist_ok=True)
        img_pil.save(f'perturbation_samples/sample_{i}_original.png')
        img_down_up.save(f'perturbation_samples/sample_{i}_downup.png')
        img_blur.save(f'perturbation_samples/sample_{i}_blur.png')
        img_sharp.save(f'perturbation_samples/sample_{i}_sharp.png')
        img_noisy.save(f'perturbation_samples/sample_{i}_noisy.png')

    plt.suptitle('Effects of Different Image Perturbations', fontsize=16)
    plt.savefig('perturbation_effects1.png', bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == '__main__':
    visualize_transformations()
    print("Visualization complete! Check 'perturbation_effects2.png' and 'perturbation_samples' directory for results.")

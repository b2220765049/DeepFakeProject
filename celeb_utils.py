import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import json
import glob
from tqdm import tqdm
import pywt  # Import PyWavelets for Haar wavelet decomposition

class FDataset_extended(Dataset):
    def __init__(self, root_dir, json_file_path_list, type_name = None, get_path=False, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths= []
        self.labels = []
        self.get_path = get_path
        class_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir)]
        class_dirs = [d for d in class_dirs if os.path.isdir(d)]

        json_data = []
        for json_file_path in json_file_path_list:
            with open(json_file_path, 'r') as f:
                current_data = json.load(f)
                json_data += sum(current_data, [])

        for label, class_dir in enumerate(class_dirs):
            print(label, class_dir)
            type_folders = os.listdir(class_dir)
            for type_folder in type_folders:
                if type_name:
                    if label == 1 and type_folder != type_name:
                        continue
                type_folder_path = os.path.join(class_dir, type_folder)
                image_folders = os.listdir(type_folder_path)
                for image_folder in image_folders:
                     # Filter the videos that are not in json data
                    folder_name = image_folder.split('_')[0] if '_' in image_folder else image_folder
                    if folder_name not in json_data: continue
                    image_folder_path = os.path.join(type_folder_path, image_folder)
                    image_paths = [os.path.join(image_folder_path, image) for image in os.listdir(image_folder_path) if image.endswith('.png')]
                    labels = [1 - label for _ in range(len(image_paths))]
                    if len(image_paths) < 30:
                        print(f"Bad Folder: {image_folder_path}")
                        continue
                    self.image_paths += image_paths
                    self.labels += labels

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        img = Image.open(image_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        
        if not self.get_path:
            return img, torch.tensor(label, dtype=torch.float32)
        else:
            return img, torch.tensor(label, dtype=torch.float32), image_path

# Custom transforms
class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
class celebDataset(Dataset):
    def __init__(self, root_dir, get_path=False, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths= []
        self.labels = []
        self.get_path = get_path
        class_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir)]
        class_dirs = [d for d in class_dirs if os.path.isdir(d)]

        for class_dir in class_dirs:
            label = 0 if 'original' in class_dir else 1
            print(f"{class_dir}/**/*.png")
            images = glob.glob(f"{class_dir}/**/*.png") 
            self.image_paths += images
            self.labels += [label for _ in range(len(images))]

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        img = Image.open(image_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        
        if not self.get_path:
            return img, torch.tensor(label, dtype=torch.float32)
        else:
            return img, torch.tensor(label, dtype=torch.float32), image_path
        

class celeb_Dataset3D(Dataset):
    def __init__(self, root_dir, json_file_path=None, num_frames=10, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.num_frames = num_frames
        self.video_folders = []
        self.labels = []

        class_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir)]

        for class_dir in class_dirs:
            label = 0 if 'original' in class_dir else 1
            for vid_folder in tqdm(os.listdir(class_dir), desc=f'Class: {class_dir}'):
                vid_path = os.path.join(class_dir, vid_folder)
                if vid_path.endswith('.txt'):
                    continue
                frames = sorted([os.path.join(vid_path, img) for img in os.listdir(vid_path) if img.endswith('.png')])
                if len(frames) >= self.num_frames:
                    self.video_folders.append(frames)
                    self.labels.append(label)

    def _is_valid_image(self, image_path):
        """Check if an image is truncated or corrupted"""
        try:
            with Image.open(image_path) as img:
                img.verify()  # Verify integrity
            with Image.open(image_path) as img:
                img.load()
            return True
        except (IOError, SyntaxError):
            print(f"Corrupt image found and skipped: {image_path}")
            return False

    def __len__(self):
        return len(self.video_folders)

    def __getitem__(self, idx):
        frames = self.video_folders[idx]
        
        # If we have more frames than needed, randomly select a continuous segment
        if len(frames) > self.num_frames:
            start_idx = torch.randint(0, len(frames) - self.num_frames + 1, (1,)).item()
            selected_frames = frames[start_idx:start_idx + self.num_frames]
        else:
            # If we have exactly num_frames frames, use all of them
            selected_frames = frames[:self.num_frames]

        # Ensure frames are valid and in order
        valid_frames = []
        for f in selected_frames:
            if self._is_valid_image(f):
                img = Image.open(f).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                valid_frames.append(img)
            
        if len(valid_frames) < self.num_frames:
            # If we don't have enough valid frames, duplicate the last valid frame
            while len(valid_frames) < self.num_frames:
                valid_frames.append(valid_frames[-1])

        # Stack frames and ensure correct shape: (C, T, H, W)
        video_tensor = torch.stack(valid_frames, dim=0).permute(1, 0, 2, 3)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return video_tensor, label

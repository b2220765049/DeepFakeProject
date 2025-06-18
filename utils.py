import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import json
import glob

class FDataset(Dataset):
    def __init__(self, root_dir, json_file_path, type_name = None, get_path=False, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths= []
        self.labels = []
        self.get_path = get_path
        class_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir)]
        class_dirs = [d for d in class_dirs if os.path.isdir(d)]

        with open(json_file_path, 'r') as f:
            json_data = json.load(f)
            json_data = sum(json_data, [])

        for label, class_dir in enumerate(class_dirs):
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
    
class FDatasetTest(Dataset):
    def __init__(self, root_dir, json_file_path, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths= []
        self.labels = []

        class_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir)]
        class_dirs = [d for d in class_dirs if os.path.isdir(d)]

        with open(json_file_path, 'r') as f:
            json_data = json.load(f)
            json_data = sum(json_data, [])

        for label, class_dir in enumerate(class_dirs):
            type_folders = os.listdir(class_dir)
            for type_index, type_folder in enumerate(type_folders, 1):
                print(type_index, type_folder)
                type_folder_path = os.path.join(class_dir, type_folder, 'c23', 'images')
                image_folders = os.listdir(type_folder_path)
                for image_folder in image_folders:
                     # Filter the videos that are not in json data
                    folder_name = image_folder.split('_')[0] if '_' in image_folder else image_folder
                    if folder_name not in json_data: continue
                    image_folder_path = os.path.join(type_folder_path, image_folder)
                    for image in os.listdir(image_folder_path):
                        path = os.path.join(image_folder_path, image)
                        self.image_paths.append(path)
                        if label == 0:
                            self.labels.append(0)
                        else:
                            self.labels.append(type_index)

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        img = Image.open(image_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        
        return img, torch.tensor(label, dtype=torch.float32)

class FDatasetSingleType(Dataset):
    def __init__(self, root_dir, json_file_path, type_name, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths= []
        self.labels = []

        class_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir)]
        class_dirs = [d for d in class_dirs if os.path.isdir(d)]

        with open(json_file_path, 'r') as f:
            json_data = json.load(f)
            json_data = sum(json_data, [])

        for label, class_dir in enumerate(class_dirs):
            type_folders = os.listdir(class_dir)
            for type_folder in type_folders:
                if label == 1 and type_folder != type_name:
                    continue
                type_folder_path = os.path.join(class_dir, type_folder, 'c23', 'images')
                image_folders = os.listdir(type_folder_path)
                for image_folder in image_folders:
                     # Filter the videos that are not in json data
                    folder_name = image_folder.split('_')[0] if '_' in image_folder else image_folder
                    if folder_name not in json_data: continue
                    image_folder_path = os.path.join(type_folder_path, image_folder)
                    for image in os.listdir(image_folder_path):
                        path = os.path.join(image_folder_path, image)
                        self.image_paths.append(path)
                        self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        img = Image.open(image_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        
        return img, torch.tensor(label, dtype=torch.float32)

class FDataset3D(Dataset):
    def __init__(self, root_dir, json_file_path_list, num_frames=10, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.num_frames = num_frames
        self.video_folders = []
        self.labels = []

        json_data = []
        for json_file_path in json_file_path_list:
            with open(json_file_path, 'r') as f:
                current_data = json.load(f)
                json_data += sum(current_data, [])

        class_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

        for label, class_dir in enumerate(class_dirs):
            for type_folder in os.listdir(class_dir):
                type_folder_path = os.path.join(class_dir, type_folder)
                if not os.path.isdir(type_folder_path):
                    continue
                for vid_folder in os.listdir(type_folder_path):
                    folder_name = vid_folder.split('_')[0] if '_' in vid_folder else vid_folder
                    if folder_name not in json_data:
                        continue
                    vid_path = os.path.join(type_folder_path, vid_folder)
                    frames = sorted([os.path.join(vid_path, img) for img in os.listdir(vid_path) if img.endswith('.png')])
                    if len(frames) >= self.num_frames:
                        self.video_folders.append(frames)
                        self.labels.append(1 - label)

    def __len__(self):
        return len(self.video_folders)

    def __getitem__(self, idx):
        frames = self.video_folders[idx]
        # Take first N frames or a random segment if you want
        selected_frames = frames[:self.num_frames]

        imgs = []
        for f in selected_frames:
            img = Image.open(f).convert('RGB')
            if self.transform:
                img = self.transform(img)
            imgs.append(img)

        # Stack: (T, C, H, W) => transpose to (C, T, H, W) if needed
        imgs = torch.stack(imgs, dim=0).permute(1, 0, 2, 3)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return imgs, label


class FDataset_ensemble(Dataset):
    def __init__(self, root_dir, json_file_path, transform=None, transform2=None, transform3=None):
        self.root_dir = root_dir
        self.transform1 = transform
        self.transform2 = transform2
        self.transform3 = transform3
        self.image_paths= []
        self.labels = []

        class_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir)]
        class_dirs = [d for d in class_dirs if os.path.isdir(d)]

        with open(json_file_path, 'r') as f:
            json_data = json.load(f)
            json_data = sum(json_data, [])

        for label, class_dir in enumerate(class_dirs):
            type_folders = os.listdir(class_dir)
            for type_folder in type_folders:
                type_folder_path = os.path.join(class_dir, type_folder)
                image_folders = os.listdir(type_folder_path)
                for image_folder in image_folders:
                     # Filter the videos that are not in json data
                    folder_name = image_folder.split('_')[0] if '_' in image_folder else image_folder
                    if folder_name not in json_data: continue
                    image_folder_path = os.path.join(type_folder_path, image_folder)
                    for image in os.listdir(image_folder_path):
                        path = os.path.join(image_folder_path, image)
                        self.image_paths.append(path)
                        self.labels.append(1 - label)

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        img = Image.open(image_path).convert('RGB')
        img1 = self.transform1(img)
        img2 = self.transform2(img)
        img3 = self.transform3(img)
        
        return img1,img2,img3, torch.tensor(label, dtype=torch.float32)


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
            label = 1 if 'original' in class_dir else 0
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
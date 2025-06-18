import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import cv2
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader
from scipy.fft import fft, fftfreq
import pywt
import pickle
import os
from tqdm import tqdm
from utils import FDataset
from celeb_utils import FDataset_extended, celebDataset

class DeepFakeDetector:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cnn = self._load_densenet()
        self.pca = PCA(n_components=100)
        self.scaler = StandardScaler()
        self.svm = SVC(kernel='linear', C=1.0, probability=True)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.is_trained = False
        
    def _load_densenet(self):
        model = models.densenet121(pretrained=True)
        model.classifier = nn.Identity()
        model.eval()
        model = model.to(self.device)
        return model
    
    def _extract_cnn_features(self, image_tensor):
        with torch.no_grad():
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)
            image_tensor = image_tensor.to(self.device)
            features = self.cnn(image_tensor)
        return features.squeeze().cpu().numpy()
    
    def _stockwell_transform(self, signal):
        N = len(signal)
        f = fftfreq(N, 1.0)
        F = fft(signal)
        S = np.zeros((N, N//2 + 1), dtype=complex)
        
        for k in range(N//2 + 1):
            if k == 0:
                S[:, k] = np.mean(signal)
            else:
                gaussian = np.exp(-2 * np.pi**2 * np.arange(N)**2 / k**2)
                shifted_F = np.roll(F, k)
                S[:, k] = np.fft.ifft(shifted_F * gaussian)
        
        return S
    
    def _extract_wavelet_features(self, magnitude_spectrum):
        features_list = []
        
        for i in range(magnitude_spectrum.shape[0]):
            row = magnitude_spectrum[i, :]
            coeffs = pywt.wavedec(row, 'db4', level=3)
            
            energy_features = []
            for coeff in coeffs[-3:]:
                energy = np.sum(coeff**2)
                energy_features.append(energy)
            
            features_list.append(energy_features)
        
        return np.array(features_list)
    
    def preprocess_image(self, image_tensor):
        cnn_features = self._extract_cnn_features(image_tensor)
        s_transform = self._stockwell_transform(cnn_features)
        magnitude_spectrum = np.abs(s_transform)
        wavelet_features = self._extract_wavelet_features(magnitude_spectrum)
        feature_vector = wavelet_features.flatten()
        return feature_vector
    
    def train(self, train_loader, val_loader=None):
        all_features = []
        labels = []
        
        print("Extracting features from training data...")
        for images, batch_labels in tqdm(train_loader):
            for i in range(len(images)):
                features = self.preprocess_image(images[i])
                all_features.append(features)
                labels.append(batch_labels[i].item())
        
        X = np.array(all_features)
        y = np.array(labels)
        
        print("Applying PCA and training SVM...")
        X_scaled = self.scaler.fit_transform(X)
        X_pca = self.pca.fit_transform(X_scaled)
        
        self.svm.fit(X_pca, y)
        self.is_trained = True
        
        if val_loader:
            print("Validating...")
            val_accuracy = self.evaluate(val_loader)
            print(f"Validation Accuracy: {val_accuracy:.4f}")
    
    def predict(self, image_tensor):
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        features = self.preprocess_image(image_tensor)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        features_pca = self.pca.transform(features_scaled)
        
        prediction = self.svm.predict(features_pca)[0]
        probabilities = self.svm.predict_proba(features_pca)[0]
        
        return prediction, probabilities
    
    def evaluate(self, test_loader):
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        predictions = []
        true_labels = []
        
        for images, batch_labels in tqdm(test_loader):
            for i in range(len(images)):
                pred, _ = self.predict(images[i])
                predictions.append(pred)
                true_labels.append(batch_labels[i].item())
        
        accuracy = accuracy_score(true_labels, predictions)
        return accuracy
    
    def save_model(self, filepath):
        model_data = {
            'pca': self.pca,
            'scaler': self.scaler,
            'svm': self.svm,
            'is_trained': self.is_trained
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.pca = model_data['pca']
        self.scaler = model_data['scaler']
        self.svm = model_data['svm']
        self.is_trained = model_data['is_trained']

def main():
    print("Starting DeepFake Detection Training with F++ and CelebDF")
    
    # Dataset paths
    fpp_data_dir = "dataset/f++"
    celeb_data_dir = "dataset/celeb_test"
    output_folder = 'dene_outputs'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # Transform for preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create F++ training dataset (using all splits for training)
    print("Loading F++ training dataset...")
    train_json_list = ["splits/train.json", "splits/val.json", "splits/test.json"]
    train_dataset = FDataset_extended(
        root_dir=fpp_data_dir,
        json_file_path_list=train_json_list,
        transform=transform
    )
    
    # Create CelebDF validation dataset
    print("Loading CelebDF validation dataset...")
    val_dataset = celebDataset(
        root_dir=celeb_data_dir,
        get_path=False,
        transform=transform
    )
    
    # Create CelebDF test dataset
    print("Loading CelebDF test dataset...")
    test_dataset = celebDataset(
        root_dir=celeb_data_dir,
        get_path=False,
        transform=transform
    )
    
    print(f"F++ Training samples: {len(train_dataset)}")
    print(f"CelebDF Validation samples: {len(val_dataset)}")
    print(f"CelebDF Test samples: {len(test_dataset)}")
    
    # Create data loaders with full datasets
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    # Initialize and train the detector
    detector = DeepFakeDetector()
    
    print("Training DeepFake Detector with F++ dataset...")
    detector.train(train_loader, val_loader)
    
    print("Testing on CelebDF dataset...")
    test_accuracy = detector.evaluate(test_loader)
    print(f"Test Accuracy on CelebDF: {test_accuracy:.4f}")
    
    # Save the trained model
    os.makedirs(output_folder, exist_ok=True)
    model_path = os.path.join(output_folder, 'deepfake_detector_fpp_celebdf.pkl')
    detector.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    print("Training and evaluation completed!")

if __name__ == "__main__":
    main()
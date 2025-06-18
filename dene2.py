import streamlit as st
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
import tempfile
from PIL import Image
import time

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
    
    def predict(self, image_tensor):
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        features = self.preprocess_image(image_tensor)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        features_pca = self.pca.transform(features_scaled)
        
        prediction = self.svm.predict(features_pca)[0]
        probabilities = self.svm.predict_proba(features_pca)[0]
        
        return prediction, probabilities
    
    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.pca = model_data['pca']
        self.scaler = model_data['scaler']
        self.svm = model_data['svm']
        self.is_trained = model_data['is_trained']

def extract_faces_from_video(video_path, max_frames=30):
    """Extract faces from video frames"""
    cap = cv2.VideoCapture(video_path)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    faces = []
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Sample frames evenly throughout the video
    frame_indices = np.linspace(0, total_frames - 1, min(max_frames, total_frames), dtype=int)
    
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            break
            
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        detected_faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in detected_faces:
            # Extract face with some padding
            padding = 20
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(frame_rgb.shape[1], x + w + padding)
            y2 = min(frame_rgb.shape[0], y + h + padding)
            
            face = frame_rgb[y1:y2, x1:x2]
            
            if face.size > 0:
                faces.append(face)
                break  # Take only the first face per frame
    
    cap.release()
    return faces

def preprocess_face_for_model(face_array, transform):
    """Convert face array to tensor for model input"""
    face_pil = Image.fromarray(face_array)
    face_tensor = transform(face_pil)
    return face_tensor

def main():
    st.set_page_config(
        page_title="DeepFake Detection",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 DeepFake Video Detection")
    st.markdown("Upload a video to detect if it contains deepfake content")
    
    # Initialize session state
    if 'detector' not in st.session_state:
        st.session_state.detector = None
    
    # Model loading section
    st.sidebar.header("Model Configuration")
    
    # Check for local model file first
    local_model_path = "deepfake_detector.pkl"
    model_loaded = False
    
    if os.path.exists(local_model_path):
        st.sidebar.info(f"📁 Found local model: {local_model_path}")
        
        if st.sidebar.button("🔄 Load Local Model"):
            try:
                with st.spinner("Loading local model..."):
                    st.session_state.detector = DeepFakeDetector()
                    st.session_state.detector.load_model(local_model_path)
                st.sidebar.success("✅ Local model loaded successfully!")
                model_loaded = True
            except Exception as e:
                st.sidebar.error(f"❌ Error loading local model: {str(e)}")
                st.session_state.detector = None
    
    # Auto-load local model on first run if available
    if os.path.exists(local_model_path) and st.session_state.detector is None and not model_loaded:
        try:
            with st.spinner("Auto-loading local model..."):
                st.session_state.detector = DeepFakeDetector()
                st.session_state.detector.load_model(local_model_path)
            st.sidebar.success("✅ Local model auto-loaded!")
            model_loaded = True
        except Exception as e:
            st.sidebar.error(f"❌ Error auto-loading local model: {str(e)}")
            st.session_state.detector = None
    
    # Fallback: Manual upload option
    st.sidebar.markdown("---")
    st.sidebar.subheader("📤 Manual Upload")
    model_file = st.sidebar.file_uploader(
        "Upload trained model (.pkl)",
        type=['pkl'],
        help="Upload your pre-trained deepfake detection model"
    )
    
    if model_file is not None:
        try:
            # Save uploaded model to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
                tmp_file.write(model_file.read())
                tmp_path = tmp_file.name
            
            # Load model
            if st.session_state.detector is None or not model_loaded:
                with st.spinner("Loading uploaded model..."):
                    st.session_state.detector = DeepFakeDetector()
                    st.session_state.detector.load_model(tmp_path)
                st.sidebar.success("✅ Uploaded model loaded successfully!")
            
            # Clean up temporary file
            os.unlink(tmp_path)
            
        except Exception as e:
            st.sidebar.error(f"❌ Error loading uploaded model: {str(e)}")
            st.session_state.detector = None
    
    # Main content
    if st.session_state.detector is None or not st.session_state.detector.is_trained:
        st.warning("⚠️ Please load a trained model first!")
        
        # Show available options
        if os.path.exists("deepfake_detector.pkl"):
            st.info("💡 Found `deepfake_detector.pkl` in the current directory. Click 'Load Local Model' in the sidebar to use it.")
        else:
            st.info("📝 Please upload a trained model file (.pkl) using the sidebar, or place `deepfake_detector.pkl` in the same folder as this script.")
        return
    
    # Video upload section
    st.header("📹 Upload Video")
    uploaded_video = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv', 'wmv'],
        help="Supported formats: MP4, AVI, MOV, MKV, WMV"
    )
    
    if uploaded_video is not None:
        # Display video info
        st.subheader("📊 Video Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"**Filename:** {uploaded_video.name}")
            st.info(f"**Size:** {uploaded_video.size / (1024*1024):.2f} MB")
        
        with col2:
            # Save video to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_video.read())
                video_path = tmp_file.name
            
            # Get video properties
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            cap.release()
            
            st.info(f"**Duration:** {duration:.2f} seconds")
            st.info(f"**FPS:** {fps:.2f}")
        
        # Display video
        st.subheader("🎬 Video Preview")
        st.video(uploaded_video)
        
        # Analysis section
        st.header("🔬 Analysis")
        
        # Configuration options
        col1, col2 = st.columns(2)
        with col1:
            max_frames = st.slider("Max frames to analyze", 5, 50, 20)
        with col2:
            confidence_threshold = st.slider("Confidence threshold", 0.1, 0.9, 0.5)
        
        if st.button("🚀 Analyze Video", type="primary"):
            try:
                with st.spinner("Extracting faces from video..."):
                    faces = extract_faces_from_video(video_path, max_frames)
                
                if not faces:
                    st.error("❌ No faces detected in the video!")
                    return
                
                st.success(f"✅ Found {len(faces)} faces to analyze")
                
                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Analyze each face
                predictions = []
                probabilities = []
                
                for i, face in enumerate(faces):
                    status_text.text(f"Analyzing face {i+1}/{len(faces)}...")
                    progress_bar.progress((i + 1) / len(faces))
                    
                    # Preprocess face
                    face_tensor = preprocess_face_for_model(face, st.session_state.detector.transform)
                    
                    # Make prediction
                    pred, prob = st.session_state.detector.predict(face_tensor)
                    predictions.append(pred)
                    probabilities.append(prob)
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                # Calculate results
                fake_predictions = sum(predictions)
                total_faces = len(predictions)
                fake_percentage = (fake_predictions / total_faces) * 100
                
                # Calculate average probabilities
                avg_probabilities = np.mean(probabilities, axis=0)
                real_confidence = avg_probabilities[0] * 100
                fake_confidence = avg_probabilities[1] * 100
                
                # Display results
                st.header("📊 Results")
                
                # Main result
                if fake_percentage > (confidence_threshold * 100):
                    st.error(f"🚨 **DEEPFAKE DETECTED** 🚨")
                    st.error(f"**{fake_percentage:.1f}%** of faces classified as fake")
                else:
                    st.success(f"✅ **AUTHENTIC VIDEO** ✅")
                    st.success(f"**{100-fake_percentage:.1f}%** of faces classified as real")
                
                # Detailed metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Faces Analyzed", total_faces)
                
                with col2:
                    st.metric("Real Confidence", f"{real_confidence:.1f}%")
                
                with col3:
                    st.metric("Fake Confidence", f"{fake_confidence:.1f}%")
                
                # Show some sample faces
                st.subheader("🖼️ Sample Analyzed Faces")
                cols = st.columns(min(5, len(faces)))
                
                for i, (face, pred, prob) in enumerate(zip(faces[:5], predictions[:5], probabilities[:5])):
                    with cols[i]:
                        st.image(face, caption=f"Face {i+1}")
                        if pred == 1:
                            st.error(f"Fake ({prob[1]*100:.1f}%)")
                        else:
                            st.success(f"Real ({prob[0]*100:.1f}%)")
                
                # Detailed breakdown
                with st.expander("📈 Detailed Analysis"):
                    st.write("**Per-face predictions:**")
                    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                        status = "FAKE" if pred == 1 else "REAL"
                        confidence = prob[pred] * 100
                        st.write(f"Face {i+1}: {status} (confidence: {confidence:.2f}%)")
                
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
            
            finally:
                # Clean up temporary file
                if os.path.exists(video_path):
                    os.unlink(video_path)
    
    # Footer
    st.markdown("---")
    st.markdown("💡 **Tips:**")
    st.markdown("- Higher quality videos give better results")
    st.markdown("- Videos with clear, frontal faces work best")
    st.markdown("- The model analyzes multiple frames for more accurate detection")

if __name__ == "__main__":
    main()
# DeepFake Detection with Dene Model

This repository contains a comprehensive deepfake detection system that trains on the F++ dataset and validates on the CelebDF dataset. The model combines CNN features with Stockwell transform and wavelet analysis for robust deepfake detection.

## Model Architecture

The Dene model uses a hybrid approach combining:

1. **DenseNet-121 Backbone**: Extracts deep features from images
2. **Stockwell Transform**: Time-frequency analysis for detecting artifacts
3. **Wavelet Decomposition**: Multi-scale feature extraction using Daubechies-4 wavelets
4. **PCA Dimensionality Reduction**: Reduces feature dimensionality
5. **SVM Classifier**: Final classification with linear kernel

## Files Overview

### Core Files
- `dene.py`: Main training script for the Dene model
- `dene_evaluation.py`: Comprehensive evaluation script
- `dene_comparison.py`: Model comparison with other architectures
- `utils.py`: Dataset utilities for F++ dataset
- `celeb_utils.py`: Dataset utilities for CelebDF dataset

### Training Scripts
- `train_with_celeb.py`: Xception model training with F++ and CelebDF
- `paper_train.py`: Pairwise interaction model training
- `swin_plus_train.py`: Swin Transformer training
- `transformer_xception_train.py`: Transformer-Xception hybrid training

## Usage

### 1. Training the Dene Model

To train the Dene model on F++ dataset and validate on CelebDF:

```bash
python dene.py
```

This will:
- Load F++ dataset for training (using train, val, and test splits)
- Load CelebDF dataset for validation and testing
- Use the complete datasets for comprehensive training and evaluation
- Train the model with CNN + Stockwell + Wavelet features
- Save the trained model to `dene_outputs/deepfake_detector_fpp_celebdf.pkl`

### 2. Comprehensive Evaluation

To evaluate the trained model on both datasets:

```bash
python dene_evaluation.py
```

This will:
- Load the trained model
- Evaluate on F++ train/val/test splits using full datasets
- Evaluate on CelebDF dataset using full dataset
- Generate confusion matrices and ROC curves
- Save detailed results to `dene_evaluation_results/`

### 3. Model Comparison

To compare the Dene model with other architectures:

```bash
python dene_comparison.py
```

This will:
- Load available pretrained models (Xception, ResNet, Transformer)
- Compare performance on both F++ and CelebDF datasets using full datasets
- Generate comparison plots
- Save comparison results to `dene_comparison_results/`

## Dataset Structure

### F++ Dataset
```
dataset/f++/
├── original_sequences/
│   └── [video_folders]/
│       └── [frame_images].png
└── manipulated_sequences/
    └── [manipulation_types]/
        └── [video_folders]/
            └── [frame_images].png
```

### CelebDF Dataset
```
dataset/celeb_test/
├── celeb_original/
│   └── [video_folders]/
│       └── [frame_images].png
├── celeb_fake/
│   └── [video_folders]/
│       └── [frame_images].png
└── youtube_original/
    └── [video_folders]/
        └── [frame_images].png
```

## Model Features

### 1. CNN Feature Extraction
- Uses DenseNet-121 pretrained on ImageNet
- Extracts 1024-dimensional feature vectors
- Removes the final classification layer

### 2. Stockwell Transform
- Time-frequency analysis technique
- Detects frequency-domain artifacts in deepfake images
- Provides localized frequency information

### 3. Wavelet Analysis
- Uses Daubechies-4 wavelet family
- 3-level decomposition
- Extracts energy features from wavelet coefficients

### 4. Feature Processing
- PCA reduces dimensions to 100 components
- StandardScaler normalizes features
- SVM with linear kernel for final classification

## Performance Metrics

The evaluation provides:
- **Accuracy**: Overall classification accuracy
- **ROC AUC**: Area under the ROC curve
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

## Output Files

### Training Output
- `dene_outputs/deepfake_detector_fpp_celebdf.pkl`: Trained model

### Evaluation Output
- `dene_evaluation_results/`
  - `fpp_test_confusion_matrix.png`
  - `celebdf_confusion_matrix.png`
  - `fpp_test_roc_curve.png`
  - `celebdf_roc_curve.png`
  - `detailed_results.json`

### Comparison Output
- `dene_comparison_results/`
  - `fpp_comparison.png`
  - `celebdf_comparison.png`
  - `comparison_results.json`

## Requirements

```bash
pip install torch torchvision
pip install scikit-learn
pip install scipy
pip install PyWavelets
pip install matplotlib
pip install seaborn
pip install tqdm
pip install timm
```

## Key Advantages

1. **Cross-Dataset Generalization**: Trains on F++ and validates on CelebDF
2. **Hybrid Feature Extraction**: Combines CNN, frequency, and wavelet features
3. **Robust Artifact Detection**: Stockwell transform captures subtle artifacts
4. **Multi-Scale Analysis**: Wavelet decomposition provides multi-resolution features
5. **Complete Dataset Usage**: Uses full datasets for comprehensive training and evaluation

## Comparison with Other Models

The Dene model is compared against:
- **Xception**: Deep CNN architecture
- **ResNet-50**: Residual network architecture
- **Transformer**: Attention-based architecture

This comparison helps understand the effectiveness of the hybrid approach versus pure CNN or transformer-based methods.

## Notes

- The model uses the complete datasets for training and evaluation
- Training uses all available F++ data (train, val, test splits combined)
- Evaluation uses full CelebDF dataset for comprehensive testing
- The Stockwell transform is computationally intensive but provides valuable frequency-domain information
- The wavelet analysis helps capture artifacts at different scales

## Performance Considerations

- **Memory Usage**: Using full datasets requires more memory
- **Training Time**: Complete dataset training takes longer but provides better results
- **Evaluation Time**: Full dataset evaluation provides more reliable metrics
- **Batch Size**: May need to reduce batch size if memory is limited

## Troubleshooting

1. **CUDA Memory Issues**: Reduce batch size or use CPU
2. **Dataset Loading Errors**: Check dataset paths and structure
3. **Model Loading Errors**: Ensure trained model exists before evaluation
4. **Memory Issues**: Reduce batch size or use gradient accumulation
5. **Long Training Time**: Consider using a subset for initial testing

## Citation

If you use this code, please cite the relevant papers for the datasets and techniques used. 
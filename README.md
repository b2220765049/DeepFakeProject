# Advanced DeepFake Detection Research Toolkit

This project is a collection of scripts and models for research and experimentation in DeepFake detection. It implements various deep learning architectures, training pipelines, evaluation methodologies, and custom approaches for identifying manipulated images and videos.

## Overview

The toolkit focuses on exploring diverse techniques for DeepFake detection, including:
*   **State-of-the-Art Models:** Leveraging architectures like Xception, ResNet3D (Slow R50), Swin Transformer, Vision Transformer (ViT), and ConvNeXt.
*   **Custom Model Enhancements:** Implementing modifications such as frequency domain analysis with Swin Transformers, MixStyle for domain generalization, and Deep Residual Feature Mining (DRFM).
*   **Temporal Analysis:** Utilizing 3D convolutional networks and transformers for video-based detection.
*   **Ensemble Methods:** Combining predictions from multiple models to improve robustness.
*   **Novel Detection Approaches:** Includes a custom detector (`dene.py`) based on DenseNet features, Stockwell transform, wavelet analysis, and SVM classification.
*   **Dataset Support:** Scripts are designed to work with common DeepFake datasets like FaceForensics++ (FF++) and CelebDF.
*   **Comprehensive Evaluation:** Includes scripts for detailed performance analysis, ROC AUC calculation, confusion matrices, and model comparison.
*   **Interactive Demos:** Streamlit applications for easy testing and visualization of detection results.

## Features

*   **Multiple Model Architectures:**
    *   XceptionNet
    *   ResNet3D (Slow R50)
    *   Swin Transformer (Base, with Frequency analysis, MixStyle, DRFM)
    *   Vision Transformer (ViT)
    *   ConvNeXt
    *   Custom Xception + Transformer
    *   Custom ResNet3D + Transformer
    *   Ensemble models
    *   Custom Pairwise Interaction Model (from `paper_train.py`)
    *   DenseNet + Stockwell Transform + Wavelet + SVM (`dene.py`)
*   **Training Scripts:** Dedicated scripts for training various models (e.g., `train.py`, `3d_train.py`, `swin_temporal.py`, `vmamba_train.py` (likely ViT), `transformer_train.py`).
*   **Evaluation Scripts:** Scripts for testing model performance on different datasets (e.g., `test.py`, `3d_test.py`, `last_eval.py`, `multi_test.py`).
*   **Dataset Utilities:** Helper scripts (`utils.py`, `celeb_utils.py`) for loading and preprocessing FF++ and CelebDF datasets.
*   **Data Augmentation & Perturbation:**
    *   Standard augmentations (RandomHorizontalFlip, GaussianNoise, RandomErasing).
    *   Analysis of model robustness against perturbations like downsampling/upsampling, blur, sharpen, salt & pepper noise (`test_all_models.py`, `transformations_visalize_delete.py`).
*   **Streamlit Applications:**
    *   `dene2.py`: Interactive demo for the custom DenseNet-based detector.
    *   `final_test2.py`: Interactive demo for a Swin Transformer based detector with face extraction.
*   **Model Comparison:** Scripts like `dene_comparison.py` and `test_all_models.py` for comparing the performance of different models.

## Datasets

The scripts are primarily designed to work with:
*   **FaceForensics++ (FF++)**: A large-scale dataset for DeepFake detection.
*   **CelebDF**: A challenging dataset with high-quality DeepFake videos.

Ensure your datasets are structured appropriately for the `FDataset`, `FDataset_extended`, `celebDataset`, and `celeb_Dataset3D` classes in `utils.py` and `celeb_utils.py`. This typically involves organizing real and fake videos/images into respective directories and using JSON files (from `splits/`) to define train/validation/test splits.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone github.com/b2220765049/DeepFakeProject
    cd DeepFakeProject
    ```

2.  **Create a Python environment:**
    It's recommended to use a virtual environment (e.g., conda or venv).
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    You will need to install PyTorch, torchvision, timm, scikit-learn, OpenCV, dlib, Streamlit, Matplotlib, Seaborn, PyWavelets, and NumPy.
    ```bash
    pip install torch torchvision torchaudio
    pip install timm scikit-learn opencv-python dlib streamlit matplotlib seaborn pywavelets numpy tqdm
    ```
    *Note: For dlib, you might need to install CMake and a C++ compiler first. Refer to dlib installation guides for your OS.*
    *Note: Ensure PyTorch is installed with CUDA support if you have a compatible NVIDIA GPU.*

4.  **Download Pretrained Models/Landmark Predictors:**
    *   Some scripts might require pretrained model weights (e.g., `shape_predictor_81_face_landmarks.dat` for dlib, or specific model checkpoints). Place these in the expected directories (e.g., a `models/` folder or as specified in the scripts).
    *   The `timm` library and `torch.hub` will automatically download pretrained weights for backbone models if `pretrained=True` is used and weights are not found locally.

5.  **Prepare Datasets:**
    *   Download FF++ and CelebDF datasets.
    *   Preprocess them (e.g., extract frames from videos, crop faces) as required by the dataset loader classes.
    *   Place them in a `dataset/` directory (e.g., `dataset/f++`, `dataset/celeb_test`).
    *   Ensure the `splits/` directory contains the JSON files defining train/validation/test splits.

## Usage

### Training Models

Most training scripts can be run directly. For example:
```bash
python train.py  # Example for XceptionNet on FF++
python 3d_train.py # Example for Slow R50 on FF++
python swin_temporal.py # Example for Swin Transformer
# ... and so on for other training scripts.
```
*   Modify parameters like `num_epochs`, `batch_size`, `learning_rate`, `output_folder`, and dataset paths directly within the scripts.
*   Trained model checkpoints will typically be saved in the specified `output_folder`.

### Evaluating Models

Evaluation scripts load trained checkpoints and test them on specified datasets.
```bash
python test.py  # Example for testing a trained XceptionNet
python 3d_test.py # Example for testing a trained Slow R50
python last_eval.py # Example for ViT evaluation
# ... and so on for other testing scripts.
```
*   Update `model_checkpoint` and `data_dir` paths in the test scripts.
*   Results often include accuracy, ROC AUC, and sometimes confusion matrices.

### Running Streamlit Demos

To run the interactive demos:
```bash
streamlit run dene2.py  # For the custom DenseNet-based detector
streamlit run final_test2.py # For the Swin Transformer based detector
```
These applications usually require you to upload a video file for analysis. Ensure the necessary model checkpoints (`.pkl` or `.pth`) are available in the paths specified within these Streamlit scripts.

### Model Comparison

```bash
python dene_comparison.py # Compares 'dene' model with Xception, ResNet, Transformer
python test_all_models.py # Evaluates multiple models against various image perturbations
```

## Key Scripts

*   **`utils.py`, `celeb_utils.py`**: Core dataset loading classes (FDataset, celebDataset, FDataset3D, etc.) and utility functions like `AddGaussianNoise`.
*   **`train.py` / `train_with_celeb.py`**: General training script, often used for 2D CNNs like Xception, training on FF++ and evaluating/testing on CelebDF.
*   **`test.py`**: General testing script for 2D CNNs.
*   **`3d_train.py`, `3d_test.py`**: Training and testing scripts for 3D CNNs (e.g., Slow R50).
*   **`swin_*.py` files (`swin_temporal.py`, `swin_frequency.py`, `swin_mix_style.py`, `swin_plus_train.py`):** Scripts for training Swin Transformer models with various modifications (temporal, frequency domain input, MixStyle augmentation, DRFM module).
*   **`transformer_train.py`, `transformer_test.py`**: Training and testing for ResNet3D + Transformer architecture.
*   **`transformer_xception_train.py`, `transformer_xception_test.py`**: Training and testing for Xception + Transformer architecture.
*   **`vmamba_train.py`, `last_eval.py`**: Training and evaluation for Vision Transformer models.
*   **`dene.py`**: Implements a custom DeepFake detection pipeline using DenseNet, Stockwell Transform, Wavelet features, and SVM.
*   **`dene2.py`**: Streamlit application for the `dene.py` model.
*   **`final_test2.py`**: Streamlit application for a Swin Transformer based detector.
*   **`ensemble_model.py`**: Trains an ensemble layer on top of outputs from multiple pretrained models.
*   **`paper_train.py`**: Implements a pairwise interaction model, based on a research paper, using contrastive and classification losses.
*   **`test_all_models.py`**: Script for evaluating the robustness of different models against various image perturbations.
*   **`dene_comparison.py`**: Compares the performance of the `dene` model against other standard models.
*   **`transformations_visalize_delete.py`**: Utility to visualize the effect of image perturbations.

## Notes

*   Many scripts have hardcoded paths and parameters. You will need to adjust these according to your setup and specific experiments.
*   The project is research testbed, so some scripts might be experimental or specific to certain explorations.
*   Ensure your GPU has enough VRAM for training larger models, especially 3D CNNs and Transformers.

# Image-Based Multi-Target Regression with EfficientNet
## Project Overview

This project implements an image-based multi-target regression system using a deep convolutional neural network to predict five continuous target variables from each input image. The primary goal is to leverage a pretrained vision backbone to extract informative visual features and learn an efficient mapping from those features to multiple numeric outputs.

The core problem addressed is how to translate high-dimensional image representations into structured numerical predictions across multiple labels simultaneously. This setup commonly arises in real-world computer vision applications such as medical imaging, quality inspection, and remote sensing, where a single image must be mapped to several associated measurements or scores.

Rather than training an image model from scratch, this pipeline focuses on:

* Feature reuse via a pretrained EfficientNet backbone

* Lightweight regression modeling on top of extracted features

## Methodology
### 1. Feature Extraction with EfficientNet

A pretrained EfficientNet-B2 model is used as a fixed feature extractor. The classification head is removed, and only the convolutional backbone and global pooling layers are retained. This transforms images into dense numerical feature vectors suitable for regression.

Images are standardized using bilinear resizing and cropping, ImageNet normalization, and GPU-accelerated preprocessing via torchvision.transforms.

### 2. Multi-Target Regression Head

A custom regression head (DenseHead) maps extracted image features to five continuous outputs. The head is designed generically, allowing for arbitrarily deep fully-connected networks, though in this implementation it is instantiated as a single linear layer.

The combined architecture is constructed via the SingleHeadNetwork wrapper, linking:

EfficientNet Backbone → Dense Prediction Head

### 3. Closed-Form Linear Initialization

Instead of random initialization, the regression weights are computed directly using the normal equation for linear least squares:

𝜃
\=
(
𝑋
𝑇
𝑋
)
−
1
𝑋
𝑇
𝑌
θ=(X
T
X)
−1
X
T
Y

Where:

X = extracted image features (with a bias term),

Y = target matrix.

This provides a deterministic, globally optimal solution for linear regression under Gaussian noise assumptions and avoids the instability of warm-start optimization.

The resulting solution is injected directly into the model’s parameters.

### 4. Loss Function

This Kaggle competition is scored on a custom metric based on a weighted coefficient of determination (R²-like metric) to emphasize certain targets more than others:

weights = [0.1, 0.1, 0.1, 0.5, 0.2]

This weighting scheme prioritizes higher-value targets and the loss is computed as a normalized residual-to-variance ratio.

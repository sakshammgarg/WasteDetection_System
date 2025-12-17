# Waste Classification using Deep Learning

This project focuses on automated waste classification using deep learning and ensemble learning techniques on image data. The system accurately classifies waste images into organic and recyclable categories, addressing robustness and interpretability challenges in real-world waste management systems.

## Project Description

This project aims to build an automated waste classification system using a stacked ensemble deep learning model with transfer learning. Multiple pretrained CNN architectures are combined to improve classification accuracy and generalization. To address the black-box nature of deep learning models, several Explainable AI (XAI) techniques are applied to interpret model predictions and visualize class-discriminative regions in waste images.

## Key Steps

### 1) Data Loading and Exploration
- Load the waste classification dataset from Kaggle
- Inspect dataset structure and check class distribution
- Analyze sample images across different waste categories
- Identify class imbalance between organic and recyclable waste

### 2) Data Preprocessing
- Resize all images to `224 × 224 × 3` for model compatibility
- Normalize pixel values for improved training
- Apply data augmentation techniques (rotation, flipping, scaling)
- Compute class weights to handle class imbalance

### 3) Model Architecture
Train multiple pretrained CNNs as base learners:
- **ResNet18**: Efficient residual network
- **ResNet34**: Deeper residual network for complex patterns
- **MobileNetV2**: Lightweight architecture for deployment

**Ensemble Strategy:**
- Extract features independently from each pretrained model
- Concatenate feature representations from all models
- Apply a **Logistic Regression meta-learner** for stacked ensembling

### 4) Training Strategy
- Transfer learning using ImageNet-pretrained weights
- Fine-tuning on waste classification dataset
- **Optimizer**: AdamW
- **Loss Function**: Cross-Entropy Loss
- **Epochs**: 6

### 5) Model Evaluation
Evaluate models using comprehensive metrics:
- **Accuracy**: Overall correctness of predictions
- **Precision**: Proportion of predicted waste type that is correct
- **Recall**: Proportion of actual waste type that is detected
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrices**: Detailed breakdown of predictions
- Comparison of individual CNNs vs ensemble strategies

### 6) Explainable AI (XAI)
Interpret model predictions using multiple XAI techniques:
- **Grad-CAM++**: Gradient-weighted class activation mapping
- **LIME**: Local interpretable model-agnostic explanations
- **Saliency Maps**: Gradient-based feature importance
- **Integrated Gradients**: Path-based attribution method

These techniques highlight image regions influencing predictions and validate that the model has learned relevant waste features.

## Results

The project demonstrates highly effective waste classification with strong ensemble performance:

**Key Findings:**
- Individual CNNs achieve >94% accuracy on the test set
- Stacked ensemble achieves ~96.5% accuracy
- Ensemble significantly reduces misclassification errors compared to single models
- MobileNetV2 offers strong performance with low computational cost
- XAI visualizations confirm models focus on relevant waste regions rather than background

## Dataset

**Source**: [Kaggle - Waste Classification Dataset](https://www.kaggle.com/datasets/techsash/waste-classification-data)

**Classes**: Organic, Recyclable  
**Size**: ~25,000 images  

## Dependencies

The project requires the following Python libraries:

```bash
numpy
pandas
torch
torchvision
scikit-learn
opencv-python
matplotlib
lime
```

Install the dependencies using:

```bash
pip install -r requirements.txt
```

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/sakshammgarg/WasteDetection_System.git
cd WasteDetection_System
```

2. **Download the dataset**
   - Visit [Kaggle Dataset](https://www.kaggle.com/datasets/waste-classification)
   - Download and extract the dataset
   - Organize images into class-wise directories (organic, recyclable)
   - Update dataset paths in the notebook/script

## Usage

Run the notebook or training script to explore the complete analysis.

The notebook will:
1. Load and explore the waste classification image dataset
2. Preprocess images and apply data augmentation
3. Train individual CNN models (ResNet18, ResNet34, MobileNetV2)
4. Train the stacked ensemble with a Logistic Regression meta-learner
5. Display comprehensive evaluation metrics for all models
6. Generate XAI visualizations (Grad-CAM++, LIME, etc.) for interpretability
7. Provide model comparison summary and deployment recommendations

## Project Structure

```
waste-classification/
│
├── data/                   # Dataset (download separately)
├── models/                 # Saved trained models
├── notebooks/              # Training & analysis notebooks
├── src/                    # Training and evaluation scripts
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Model Selection Guide

**For Production Deployment:**
- **Best Accuracy & Robustness**: Stacked Ensemble (ResNet18 + ResNet34 + MobileNetV2)
- **Lightweight Deployment**: MobileNetV2 (mobile/edge devices)
- **Balanced Trade-off**: ResNet18 (speed vs accuracy)
- **Maximum Interpretability**: Any model with XAI techniques applied

## Advanced Features

This comprehensive implementation includes:
- Multiple CNN architectures comparison
- Stacked ensemble learning approach
- Transfer learning with pretrained ImageNet weights
- Data augmentation for improved generalization
- Class imbalance handling with weighted loss
- Explainable AI techniques for model interpretability
- Visualization of learned features and attention regions
- Model comparison with detailed performance metrics

## Applications

Practical use cases for this waste classification system:
- Smart waste segregation systems in public spaces
- Automated recycling pipelines in waste management facilities
- Smart city waste management infrastructure
- Environmental monitoring solutions
- Educational tools for waste awareness

## Future Improvements

Potential enhancements for even better results:
1. Extend to multi-class waste classification (plastic, metal, glass, paper, etc.)
2. Real-time detection using object detection models (YOLO, Faster R-CNN)
3. Deployment as a web or mobile application with user interface
4. Integration with robotic waste-sorting systems
5. Dataset expansion to capture real-world variability and edge cases
6. Implementation of attention mechanisms for better feature learning
7. Model optimization and quantization for edge deployment
8. Active learning pipeline for continuous model improvement

---

This project demonstrates how ensemble learning and explainable AI can be combined to build accurate, reliable, and interpretable waste classification systems suitable for practical deployment in real-world waste management scenarios.

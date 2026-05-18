# Animal Image Classification: Cats vs Dogs vs Panda

This repository implements a **Traditional Machine Learning Pipeline** for image classification, leveraging deep feature extraction from pretrained CNNs.

## 🚀 Project Overview
Unlike end-to-end Deep Learning, this project treats pretrained Convolutional Neural Networks (CNNs) as feature extractors and feeds those features into classical ML algorithms like Logistic Regression, SVM, and Random Forests.

## 🛠 Pipeline Stages
1. **Data Acquisition**: Automated download of the Kaggle Animal Image Dataset (3,000 unique images).
2. **EDA**: Comprehensive Exploratory Data Analysis including resolution checks, aspect ratio distribution, and duplicate detection using MD5 hashing.
3. **Preprocessing**: Image cleaning, RGB conversion, and resizing to 224x224.
4. **Feature Extraction**: Comparison of three architectures:
   - VGG16
   - ResNet50
   - EfficientNetB0
5. **Model Training**: Comparison of:
   - Logistic Regression
   - Support Vector Machine (SVM)
   - Random Forest
6. **Evaluation**: Metrics include Accuracy, Macro F1-Score, and Confusion Matrices.
7. **Inference**: A Gradio-based web interface for real-time image prediction.

## 📊 Results Summary
- **Best Model**: ResNet50 features + Logistic Regression.
- **Performance**: Reached ~98-99% Accuracy/F1-score on the test set.

## 💻 How to Run
1. Open the notebook in Google Colab.
2. Run all cells to download the dataset and train models.
3. The Gradio interface will provide a public URL to test your own images.

## 📂 Folder Structure
- `features/`: Extracted .npy vectors.
- `results/`: CSV comparisons and EDA visualizations.
- `eda_figures/`: Plots generated during data inspection.

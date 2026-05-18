# Assignment 3: Machine Learning with Image Data
## Animal Image Classification (Cats vs Dogs vs Panda)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/14lB21T0qARvqldPLxZ1IsXwLlK82LwJU)

This repository implements a **traditional machine learning pipeline for image data**. Unlike an end-to-end Deep Learning approach, this project treats pre-trained Convolutional Neural Networks (CNNs) purely as **feature extractors**, and feeds those high-level features into **classical Machine Learning algorithms** (Logistic Regression, SVM, and Random Forests) for image classification.

## 🚀 Project Overview & Pipeline

The pipeline is fully automated in the provided Jupyter Notebook and covers the following stages:

1. **Dataset Acquisition**: Automatically downloads the [Kaggle Animal Image Dataset: Dog, Cat and Panda](https://www.kaggle.com/datasets/ashishjangra27/animal-image-datasetdog-cat-and-panda).
2. **Exploratory Data Analysis (EDA)**: Analyzes image dimensions, corrupt files, and detects exact duplicates using MD5 hashing.
3. **Data Preprocessing**: Cleans the dataset (removes duplicates), converts to RGB, resizes all images to `224x224`, and splits into Train (70%), Validation (15%), and Test (15%) sets.
4. **Deep Feature Extraction**: Removes the top classification layers of standard pre-trained architectures and extracts flattened feature vectors using:
   - **VGG16**
   - **ResNet50**
   - **EfficientNetB0**
5. **Model Training & Comparison**: Extracted features are evaluated across three classic ML classifiers:
   - Logistic Regression
   - Support Vector Machine (SVM)
   - Random Forest
6. **Evaluation**: Models are evaluated using Accuracy, Precision, Recall, and Macro F1-Score.
7. **Deployment**: Launches a live **Gradio** web interface to test custom image uploads in real-time.

## 📊 Results Summary

All combinations of feature extractors and classifiers were evaluated. 

* **Best Model Combination**: **ResNet50 Features + Logistic Regression**
* **Validation Accuracy**: 99.56%
* **Final Test Accuracy**: **98.44%**
* **Performance on Pandas**: 100% Precision and Recall (0 misclassifications).

## 💻 How to Run

Since the entire pipeline is configured for Google Colab, no local setup is required:

1. Click the **"Open In Colab"** badge at the top of this README, or open `animal_dog_cat_panda_assignment3_pipeline.ipynb` directly in Colab.
2. Go to `Runtime` > `Run all` to execute the pipeline. 
3. The dataset will be downloaded directly into the Colab runtime (no Google Drive mounting required).
4. At the end of the notebook, a **Gradio** public link will be generated for interactive testing.

## 📂 Repository Structure

```text
├── .gitignore
├── README.md
├── animal_dog_cat_panda_assignment3_pipeline.ipynb
├── features/
│   ├── X_test_EfficientNetB0.npy
│   ├── X_test_ResNet50.npy
│   ├── X_test_VGG16.npy
│   ├── X_train_EfficientNetB0.npy
│   ├── X_train_ResNet50.npy
│   ├── X_train_VGG16.npy
│   ├── X_val_EfficientNetB0.npy
│   ├── X_val_ResNet50.npy
│   ├── X_val_VGG16.npy
│   ├── y_test.npy
│   ├── y_train.npy
│   └── y_val.npy
└── results/
    ├── optional_test_model_comparison.csv
    └── validation_model_comparison.csv
```

*(Note: EDA figures and plots are generated dynamically at runtime in the Colab notebook).*

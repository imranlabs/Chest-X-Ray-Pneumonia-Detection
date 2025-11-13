# Chest X-Ray Pneumonia Detection

## Overview
This project applies **machine learning** and **deep learning** techniques to the [Kaggle Chest X-Ray Images (Pneumonia) Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).
The objective is to classify chest X-ray images as **Normal** or **Pneumonia**, demonstrating both classical and modern modeling workflows — from feature extraction with **PCA + SVM** to **CNNs** and **transfer learning** using **ResNet-18**.


![Python](https://img.shields.io/badge/Python-3.7%2B-green) <!-- Open in Colab for your main notebook -->
<p>
  <a href="https://colab.research.google.com/github/imranlabs/Chest-X-Ray-Pneumonia-Detection/blob/main/Chest_X_Ray.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
  </a>
</p>

---
## Tech Stack
- **Python**, **NumPy**, **Pandas**
- **Scikit-learn**, **PyTorch**
- **Matplotlib**, 
- **OpenCV**, **PIL** (for image preprocessing)

---

## Dataset
- Source: Kaggle — *Chest X-Ray Images (Pneumonia)*  
- Classes: **Normal (1,341)** and **Pneumonia (3,875)** images  
- Image Size: Variable grayscale images resized for model compatibility  
- Train/Validation/Test splits maintained from original dataset
- Patient Cohort: The images were selected from retrospective cohorts of pediatric patients (aged one to five years old) 
from Guangzhou Women and Children’s Medical Center.

- Quality Control: All chest radiographs were initially screened for quality control, and diagnoses were graded by two expert physicians, with a third expert checking the evaluation set.

To reproduce results:
1. Download the dataset from Kaggle.
2. Unzip into a local `data/` folder (not included in this repository).
3. Update the dataset path in the notebook if necessary.

---

## Workflow

### 1. Data Preprocessing
- Image resizing and normalization  
- Data augmentation (horizontal flips, rotations, zoom)  
- Balanced sampling to address class imbalance  

### 2. Classical ML Approach
- **Feature Extraction**: PCA (~97% variance retention).
  This allowed for a massive dimensionality reduction, 
  for example, from 6,144 features to 497 features (as observed in later notebook cells).

- Support Vector Machines (SVM):
  - Linear SVM was used as the initial baseline.
  - Radial Basis Function (RBF) SVM was then implemented, revealing and exploiting non-linear patterns in the data to achieve better performance.
  - **Performance**: ROC-AUC ≈ **0.91**, Accuracy ≈ **0.89**

### 3. Deep Learning Approach
This section focused on deep feature learning for final performance:

- **Simple CNNs:** Initial custom CNNs were implemented, but the results were "not as promising" as the PCA -> SVM approach.

- **Transfer Learning (ResNet-18):** The project transitioned to using a pre-trained ResNet-18 model.

-**Optimizations:**

  - Images were converted from grayscale to RGB (3-channel) format.

  - The model was fine-tuned using the recommended mean and standard deviation for the pre-trained ResNet-18 model, which was found to be more effective than mean/std calculated from the current data.

  - Data augmentations were applied to the training set to improve generalization.

**Grad-CAM:** The notebook includes a visualization of the model's focus using Grad-CAM, demonstrating which regions of the X-ray image the ResNet-18 model used to make its classification.
- **Metrics**:
  - **ROC-AUC:** ~0.95  
  - **F1 (Weighted):** 0.898  
  - **F1 (Pneumonia):** 0.917  
  - **Accuracy:** ~90%  

### 4. Evaluation
- Confusion matrix visualization  
- Precision-Recall and ROC curves  
- Grad-CAM interpretability visualization added at the end of the workflow

---

## Results Summary
| Model | Method | ROC-AUC | Accuracy | F1 (Pneumonia) |
|:------|:--------|:--------|:----------|:---------------|
| PCA + SVM | Classical ML | 0.91 | 0.89 | 0.88 |
| CNN | Deep Learning | 0.94 | 0.89 | 0.90 |
| ResNet-18 | Transfer Learning | **0.95** | **0.90** | **0.92** |

---

## Repository Structure
```
├── Chest_X_Ray.ipynb     # Complete workflow notebook
├── assets/               # plots, images             
├── README.md             # Project documentation
└── requirements.txt      # Python dependencies
```

---

## Future Work 
- Extend to multi-class classification (viral vs bacterial)  
- Deploy model as a Gradio web demo  

---

## References
- [Kaggle Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)  
- Andrew Ng’s *Deep Learning Specialization*  
- *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (A. Géron)

---

**Author:** Imran Khan  
**GitHub:** [imranlabs](https://github.com/imranlabs)

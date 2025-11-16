# Chest X-Ray Pneumonia Classification  
### From Classical ML Baseline to Deep Multiclass CNN 

This repository contains a **structured Jupyter notebooks** that walks through the full evolution of a pneumonia detection pipeline on chest X-rays:

1. **Part 1 – Classical ML + Deep Learning (Binary)**  
   - PCA -> SVM baseline for Normal vs Pneumonia  
   - Deep CNN / ResNet-18 for Normal vs Pneumonia  

2. **Part 2 – Advanced Deep Learning (Multiclass)**  
   - ResNet-34 and ResNet-50 for Normal vs Viral vs Bacterial pneumonia  
   - Regularization tuning, class weighting, and Test-Time Augmentation (TTA)  
   - Grad-CAM interpretability  

![Python](https://img.shields.io/badge/Python-3.7%2B-green) <!-- Open in Colab for your main notebook -->
  <a href="https://colab.research.google.com/github/imranlabs/Chest-X-Ray-Pneumonia-Detection/blob/main/Chest_X_Ray_Classical_ML_Deep_Learning_Binary.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
  </a>
</p>

---



## Project Overview

- **Task:** Classify chest X-rays into:
  - Normal  
  - Pneumonia (binary setting)  
  - Normal, Viral pneumonia, Bacterial pneumonia (multiclass setting)  

- **Dataset:**  
  - Kaggle: *Chest X-Ray Images (Pneumonia)*  
  - Split into train / validation / test  
  - Custom stratified split for balanced validation
  - Dataset not included in this repository

- **Goals:**  
  - Build a **classical ML baseline** (PCA + SVM) and analyze its limitations  
  - Show how **deep CNNs (ResNet-18)** improve Normal vs Pneumonia detection  
  - Extend to a **multiclass setting** with ResNet-34/50  
  - Use **Grad-CAM** to interpret model decisions  
  - Discuss **clinical realism**: Normal vs Pneumonia vs Viral/Bacterial

---

## Repository Layout


```bash
ChestXRay_Pneumonia_Classification/
│
├── notebooks/
│   └── Chest_X_Ray_Classical_ML_Deep_Learning_Binary.ipynb   # PCA->SVM, Deep learning Binary class
│   └── Chest_X_Ray_Advanced_Deep_Learning_Multiclass.ipynb # Deep learning multiclass  
│
├── assets/
│   ├── gradcam_normal.png
│   ├── gradcam_pneumonia.png
│   ├── pneumonia_virus_vs_bacteria.png      # side-by-side example
│   └── confusion_matrix_multiclass.png      # optional
│
├── requirements.txt
└── README.md
```

- The **entire exploration** are in two notebooks, split into **Part 1** and **Part 2**.  
- The **assets/** folder holds Grad-CAM visualizations and sample images (e.g., Normal vs Viral vs Bacterial).

---

## Part 1 – Classical ML & Binary Deep Learning

### part 1.1 – Classical Baseline: PCA → SVM

- Preprocessing:
  - Grayscale conversion (if needed)  
  - Resize and flatten image  
  - Standard scaling  
- Feature extraction:
  - PCA retaining ~95–97% variance  
- Classifier:
  - RBF SVM  
  - Hyperparameter tuning (C, gamma)  

**Results (Normal vs Pneumonia):**

- Accuracy: ~88%  
- ROC-AUC: ~0.91  

**Key Insight:**  
Classical methods provide a solid baseline but struggle to capture the full complexity of lung textures and opacities.

---

### Part 1.2 – Deep Learning (Binary Normal vs Pneumonia)

Two variants are explored:

1. A **simple custom CNN**  
2. A **fine-tuned ResNet-18** (ImageNet pretrained)

**Training setup:**

- Loss: Cross-entropy (with class weights)  
- Optimizer: Adam  
- Learning rate scheduling  
- Early stopping based on validation performance  
- Data augmentation:
  - Random horizontal flip  
  - Small rotations  
  - Random affine transforms  
  - Normalization (ImageNet mean/std)  

**Best Results (ResNet-18 Binary):**

- AUC-ROC: ≈ 0.95  
- F1-score: ≈ 0.90  

**Takeaway:**  
Deep CNNs substantially outperform PCA + SVM, especially in challenging cases where pneumonia presents with subtle or diffuse patterns.

---

## Part 2 – Multiclass Deep Learning (Normal / Viral / Bacterial)

Here, the notebook shifts to a more realistic—but harder—problem:

> **Normal vs Viral pneumonia vs Bacterial pneumonia**

### Model Architectures

- **ResNet-34**  
  - Pretrained on ImageNet  
  - Initially fine-tune last block (layer4)  
  - Optionally fine-tune layer3 + layer4  
  - Class-weighted cross-entropy  

- **ResNet-50 (Final Model)**  
  - Higher capacity backbone  
  - Stronger regularization: `weight_decay = 2e-3`  
  - Test-Time Augmentation (TTA) at inference  
  - Carefully tuned to avoid overfitting

### Regularization & Tuning

The notebook documents experiments with:

- Different **weight decay** values  
- Different layers unfrozen (layer4 only vs layer3+4)  
- Early stopping based on validation loss and F1  
- Evaluation with and without **TTA**

---

## Final Model: ResNet-50 + TTA

The best configuration found:

- **Backbone:** ResNet-50 (ImageNet pretrained)  
- **Loss:** Class-weighted cross-entropy  
- **Regularization:** `weight_decay = 2e-3`  
- **Inference:** Test-Time Augmentation (original + horizontal flip, averaged logits)

### Final Test Metrics (Multiclass)

| Metric         | Score    |
|----------------|----------|
| **Accuracy**   | **85.10%** |
| **Weighted F1**| **0.8525** |
| **Weighted AUC** | **0.955** |
| Virus F1       | 0.7601   |
| Normal F1      | 0.8598   |
| Bacteria F1    | 0.9018   |

### Confusion Matrix (Final Model)

```text
[[184, 41,  9],    # Normal
 [  3,122, 23],    # Virus
 [  7, 10,225]]    # Bacteria
```

- Normal and Bacterial pneumonia are well-recognized.  
- Viral pneumonia shows overlap with Bacterial, as expected from the modality.

---

## Clinical Interpretation

The multiclass results reveal a key reality:

- Chest X-rays are **excellent for detecting pneumonia**,  
- but **limited for determining viral vs bacterial etiology**.

In real clinical workflows, pathogen identification depends on:

- Lab tests   
- Biomarkers   
- Symptoms and patient history  

Therefore, the most clinically impactful result is:

> A strong, reliable model for separating **Normal vs Pneumonia**,  
> with the multiclass experiments providing additional insight rather than a definitive diagnostic tool.

---

## Grad-CAM & Visualization

The notebook uses **Grad-CAM** on the final ResNet models to visualize where the network is focusing:

- **Normal image:**  
  - Grad-CAM highlights mostly clear lung fields, minimal activation.
- **Pneumonia image (Viral/Bacterial):**  
  - Heatmap focuses on areas of opacity, consolidation, or infiltrates.

---

## How to Run

1. **Clone the repo:**
   ```bash
   git clone https://github.com/imranlabs/Chest-X-Ray-Pneumonia-Detection.git
   cd Chest-X-Ray-Pneumonia-Detection
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up the dataset:**
   - Download the Kaggle Chest X-Ray Pneumonia dataset.  
   - Point the notebook’s data path to your local copy (documented in the notebook).

4. **Open the notebooks:**
   ```bash
   jupyter notebook notebooks/Chest_X_Ray_Classical_ML_Deep_Learning_Binary.ipynb
   ```

---


## Future Improvements

- Add DenseNet-121 experiments  
- Implement a simple Gradio demo for interactive predictions  
- Introduce clinical metadata if available  
- Experiment with Vision Transformers or Swin-based backbones  

---

## Summary

This project demonstrates:

- A disciplined exploration from classical ML to advanced CNNs  
- Thoughtful model selection and regularization  
- Use of TTA and Grad-CAM for robustness and interpretability  
- Awareness of clinical constraints and realistic deployment scenarios  



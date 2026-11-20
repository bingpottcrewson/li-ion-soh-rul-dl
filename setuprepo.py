import os

PROJECT_DIR = r"C:\Users\krish\OneDrive\Documents\study\s5\deep learning\miniproject"

gitignore_content = """# Python & Environments
__pycache__/
*.py[cod]
*$py.class
venv/
.venv/
env/

# Raw NASA PCoE Dataset Binaries (download via Kaggle/NASA)
*.mat

# OS & Editor Artifacts
.DS_Store
Thumbs.db
.vscode/
.idea/
"""

requirements_content = """numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.3.0
tensorflow>=2.15.0
matplotlib>=3.7.0
joblib>=1.3.0
"""

readme_content = """# Cross-Battery State of Health (SoH) and Remaining Useful Life (RUL) Prognostics Using a Hybrid 1D-CNN-LSTM Architecture

**Authors:** Vishnu Narippatta, Pratyush M  
**Department of Electrical and Electronics Engineering, Amrita Vishwa Vidyapeetham (Amritapuri Campus)**

## Overview
This repository provides a leakage-free deep learning framework for cross-battery State of Health (SoH) and Remaining Useful Life (RUL) estimation on the **NASA Prognostics Center of Excellence (PCoE)** lithium-ion battery aging dataset (`B0005`, `B0006`, `B0007`, `B0018`, and pulsed-load cells `B0025`–`B0028`).

### Key Highlights
- **Uniform Intra-Cycle Resampling:** Variable-length discharge curves ($V, I, T$) are linearly interpolated onto a fixed **200-step** normalized time grid $\\mathbf{X}_k \\in \\mathbb{R}^{200 \\times 3}$.
- **Capacity Normalization:** Ground-truth targets are scaled by initial cycle capacity ($\\text{SoH}_k = C_k / C_0$) to remove cell-to-cell baseline capacity offsets, with `MinMaxScaler` fitted strictly on training folds.
- **4-Fold Leave-One-Battery-Out Cross-Validation (LOOCV):** Evaluates true zero-shot generalization on completely unseen cells without cycle-level data leakage.
- **Analytical & Domain-Shift Benchmarks:** Compares the Hybrid 1D-CNN-LSTM against standalone 1D-CNN, standalone LSTM, a fitted empirical power-law fade equation ($\\widehat{\\text{SoH}}(k) = 1 - a \\cdot k^b$), and a zero-shot $4.0\\,\\text{A}$ square-wave pulsed stress test.

---

## Model Architecture
```text
Input (200 x 3) 
  -> Conv1D(64, k=3, ReLU) -> MaxPool1D(2) -> (99 x 64)
  -> Conv1D(32, k=3, ReLU) -> MaxPool1D(2) -> (48 x 32)
  -> LSTM(50) -> Dropout(0.2) -> (50)
  -> Dense(32, ReLU) -> Dense(1) -> SoH_hat in R^1"""

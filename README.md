# Research-Grade Anomaly Detection — SECOM Semiconductor Dataset

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.x-green)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](anomaly_research_final.ipynb)

> **GNN-LSTM-Transformer vs Classical ML Baselines — Full Pipeline with SMOTE**

A complete, research-grade anomaly detection pipeline on the [UCI SECOM Semiconductor Manufacturing dataset](https://archive.ics.uci.edu/ml/datasets/SECOM), comparing classical ML and a custom deep learning architecture with full explainability.

---

## Overview

| Property | Detail |
|---|---|
| **Dataset** | UCI SECOM Semiconductor Manufacturing |
| **Task** | Binary anomaly detection (wafer pass/fail) |
| **Class Imbalance** | ~14:1 (Normal : Anomaly) |
| **Models** | Logistic Regression · Random Forest · XGBoost · Isolation Forest · GNN-LSTM-Transformer |
| **Metrics** | Accuracy · Precision · Recall · F1 · ROC-AUC · PR-AUC |

---

## Architecture

```
Sensor Data → GCN (spatial correlations)
                 ↓
             LSTM (temporal patterns)
                 ↓
         Multi-Head Attention
                 ↓
         Binary Classifier (Anomaly / Normal)
```

---

## Problem
Semiconductor factories produce thousands of wafers daily — missing one defect = scrapped wafer = huge financial loss.
- **Key challenges:**

1. 14:1 class imbalance — models ignore rare failures
2. 40%+ missing sensor data
3. 591 sensors with complex spatial & temporal patterns
4. Standard accuracy is misleading (93% by predicting all "Normal")


## Solution
Built an end-to-end pipeline mimicking real factory deployment:

1. **SMOTE** — synthetically balances defective vs normal samples
2. **GNN —** maps sensor-to-sensor correlations (like a factory wiring diagram)
3. **LSTM —** detects drift patterns over time
4. **Threshold Tuning —** prioritises catching defects over false alarms
5. **SHAP —** tells engineers which sensors caused the alert

## Results:
- `Accuracy - ~99%+`
- `F1-Score - 0.9966`
- `ROC-AUC - 1.0000`
- `PR-AUC - 1.0000`
- `Recall - 1.0000`

## Pipeline Steps

1. **Setup & Imports** — Libraries, reproducibility seeds, device setup
2. **Data Loading & Preprocessing** — UCI download, missing value pruning, imputation, scaling
3. **Exploratory Data Analysis (EDA)** — Class distribution, missing values, feature variance, correlation heatmap
4. **SMOTE Balancing & Sequence Construction** — SMOTE · Borderline-SMOTE · ADASYN · SMOTE-Tomek comparison
5. **Model Definitions** — GNN-LSTM-Transformer architecture
6. **Training GNN-LSTM-Transformer** — AdamW + Cosine Annealing, 30 epochs
7. **Classical ML Baselines** — Logistic Regression, Random Forest, XGBoost, Isolation Forest
8. **Model Evaluation** — Full metric collection across all models
9. **Optimal Threshold Tuning** — Sweep for best F1 / Recall@Precision≥0.5
10. **SMOTE Variant Comparison** — XGBoost performance across all balancing strategies
11. **Stratified K-Fold Cross-Validation** — 5-fold CV with error bars
12. **SHAP Explainability** — Feature importance + beeswarm for XGBoost
13. **Comprehensive Visualisation Suite** — Grouped bar, ROC/PR curves, confusion matrices, radar chart, heatmap
14. **Final Results Summary** — Research conclusions

---

## Generated Plots

| File | Description |
|---|---|
| `eda_plots.png` | EDA: class distribution, missing values, correlations |
| `smote_comparison.png` | Before vs after balancing for all SMOTE variants |
| `pca_smote.png` | 2D PCA projection before/after SMOTE |
| `training_curve.png` | GNN-LSTM train/val loss curve |
| `threshold_tuning.png` | Precision/Recall/F1 vs threshold (XGBoost) |
| `smote_variant_comparison.png` | F1/Recall/PR-AUC across SMOTE strategies |
| `cross_validation.png` | 5-fold CV results with error bars |
| `shap_analysis.png` | SHAP summary + beeswarm plots |
| `metrics_grouped_bar.png` | All metrics for all models |
| `roc_pr_curves.png` | ROC and Precision-Recall curves |
| `confusion_matrices.png` | Confusion matrices for all models |
| `radar_chart.png` | Radar chart — model performance profiles |
| `metrics_heatmap.png` | Heatmap of all metrics |
| `feature_importance.png` | Top 20 features (RF + XGBoost) |

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/anomaly-detection-secom.git
cd anomaly-detection-secom
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter and run the notebook

```bash
jupyter notebook anomaly_research_final.ipynb
```

> **Note:** The dataset is downloaded automatically from the UCI repository at runtime — no manual download needed.

---

## Requirements

See [`requirements.txt`](requirements.txt). Key packages:

- `torch >= 2.0`
- `scikit-learn >= 1.3`
- `imbalanced-learn >= 0.11`
- `xgboost >= 2.0`
- `shap >= 0.44`
- `pandas`, `numpy`, `matplotlib`, `seaborn`

---

## Key Research Findings

1. **SMOTE** raised minority-class Recall by ~35–50% vs raw imbalanced training.
2. **Borderline-SMOTE** and **SMOTE-Tomek** generally outperform vanilla SMOTE on PR-AUC.
3. **Threshold tuning** (shifting from 0.5 → ~0.3–0.4) delivers further gains without retraining.
4. **GNN-LSTM-Transformer** best captures spatial sensor correlations + temporal drift patterns.
5. **SHAP** identifies top sensors driving anomaly predictions — actionable for process engineers.

---

## Repository Structure

```
anomaly-detection-secom/
├── anomaly_research_final.ipynb   # Main notebook (full pipeline)
├── requirements.txt               # Python dependencies
├── .gitignore                     # Files to exclude from Git
└── README.md                      # This file
```

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- Dataset: [UCI Machine Learning Repository — SECOM](https://archive.ics.uci.edu/ml/datasets/SECOM)
- SHAP: [Lundberg & Lee, 2017](https://arxiv.org/abs/1705.07874)
- imbalanced-learn: [Lemaître et al., 2017](https://jmlr.org/papers/v18/16-365.html)

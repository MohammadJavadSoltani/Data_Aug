<div align="center">
<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/ddbafd1f-319e-451e-94a2-66ba03d91c23" />

# HybridAug-OBIA-Wetland

**Hybrid Data Augmentation and Object-Based Image Analysis for LULC Mapping**  
*with Imbalanced Training Data in Complex Wetland Environments*

![Status](https://img.shields.io/badge/Status-Under%20Revision-e9c46a?style=flat-square)
![Journal](https://img.shields.io/badge/Journal-Wetlands%20%C2%B7%20Springer-2a9d8f?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.10-264653?style=flat-square&logo=python&logoColor=white)
![GEE](https://img.shields.io/badge/GEE-Sentinel--1%2F2-2a9d8f?style=flat-square)
![Augmentation](https://img.shields.io/badge/Aug-SMOTE%20%C2%B7%20ADASYN-e76f51?style=flat-square)
![License](https://img.shields.io/badge/License-Apache%202.0-f4a261?style=flat-square)

*Soltani MJ · Latifi H · Naghavi H — K.N. Toosi University of Technology, Tehran*

</div>

---

## Overview

Wetland LULC classification is chronically challenged by **imbalanced training samples** across heterogeneous classes. This repository implements a complete pipeline:

> GEE-based data preparation → spectral/SAR/GLCM feature engineering → six augmentation strategies → OBIA-integrated Random Forest classification → multi-year trend analysis

The workflow is the codebase for the paper currently under second revision at **Wetlands (Springer)**.

---

## Pipeline

| Step | Module | Description |
|---|---|---|
| **01 · Preprocess** | `Preprocessings/` | Sentinel-1/2 export handling, band splitting, VI computation, class separability |
| **02 · Engineer** | `Feature_Engineering/` | Spectral + GLCM texture + SAR merge, temporal interpolation, NaN filling |
| **03 · Augment** | `Model_Training/` | Six augmentation strategies, stratified cross-validation |
| **04 · Classify** | `Predict/` | RF / Extra Trees training, multi-year LULC maps, trend analysis |

---

## Augmentation Strategies

Six class-balancing strategies are compared against a no-augmentation baseline:

| Strategy | Description |
|---|---|
| **SMOTE** | Synthetic minority oversampling via k-NN interpolation |
| **ADASYN** | Adaptive density-based synthesis for hard-to-learn regions |
| **Borderline-SMOTE** | Synthesis focused near decision boundaries |
| **SMOTE-Tomek** | Oversampling + Tomek link cleaning for cleaner class margins |
| **Bootstrap** | Resampling with replacement from minority classes |
| **Gaussian noise** | Perturbation-based augmentation for robustness |

---

## Core Workflow

```python
# Load, augment, train, predict, map
system = EnhancedClassificationSystem("final_dataset_enhanced_all.csv")

results = system.run_all_augmentations(
    strategies=['smote', 'adasyn', 'borderline', 'smote_tomek', 'bootstrap', 'noise'],
    cv=StratifiedKFold(n_splits=5),
    models=['RandomForest', 'ExtraTrees']
)

# Generate multi-year prediction maps
system.predict_years([2019, 2020, 2021, 2022, 2023])
```

Outputs include: model `.pkl`, label encoder, per-class metrics, feature importance, prediction CSVs, LULC shapefiles, and trend analysis figures.

---

## Repository Structure

```
HybridAug-OBIA-Wetland/
│
├── Preprocessings/              # GEE export prep, band org., separability tests
│   ├── Gee_Data_to_Individuals.ipynb
│   ├── Data_Separability.ipynb
│   └── rename.ipynb
│
├── Base_Models/                 # Baseline RF before feature engineering
│   ├── train_Models_PixelBased.ipynb
│   ├── train_Models_OBIA_Spec.ipynb
│   ├── Spec_VIs.ipynb
│   ├── Spec_VIs_GLCM.ipynb
│   └── Spec_VIs_GLCM_SAR.ipynb
│
├── Feature_Engineering/         # Full feature merge, temporal NaN handling
│   ├── 1.FE_CFD_AllClass.ipynb
│   ├── 1.TrainModel.ipynb
│   ├── 2.TrainModel_RawData.ipynb
│   └── Predict/                 # Feature creation for prediction years
│
├── Merged_Classes/              # Reduced class-scheme sensitivity experiments
│
├── Model_Training/              # Augmentation experiments & final training
│   ├── Augment_LULC.py          # Raw feature pipeline
│   ├── LULC_Augmented.py        # Engineered feature pipeline
│   ├── Data_Aug_Test.ipynb
│   ├── Data_Aug_LOO.ipynb
│   └── CrossValidation.ipynb
│
└── Predict/                     # Multi-year prediction, shapefiles, trend plots
    ├── Data_Aug_Predict.ipynb
    ├── Create_Shapefile_Mask_Predicted.ipynb
    ├── Comparisons.ipynb
    └── Comparisons-Enhanced.ipynb   # Mann-Kendall, Theil-Sen, Huber regression
```

---

## Installation

```bash
git clone https://github.com/MohammadJavadSoltani/HybridAug-OBIA-Wetland.git
cd HybridAug-OBIA-Wetland

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Key Dependencies

| Package | Role |
|---|---|
| `geopandas` · `rasterio` · `shapely` | Geospatial I/O and vector operations |
| `scikit-learn` | Random Forest, Extra Trees, cross-validation |
| `imbalanced-learn` | SMOTE, ADASYN, Borderline-SMOTE, Tomek links |
| `scipy` · `statsmodels` | Mann-Kendall, Theil-Sen trend analysis |
| `matplotlib` · `seaborn` · `plotly` | Visualization and comparison figures |
| `pandas` · `numpy` · `joblib` | Data management and parallelization |

> **Note:** Input paths in `Augment_LULC.py` and `LULC_Augmented.py` are currently hardcoded. Update to your local paths before running.

---

## Citation

If you use this code, please cite:

> Soltani, M.J., Latifi, H., Naghavi, H. (2025). *Hybrid Data Augmentation and Object-Based Analysis for LULC Mapping with Imbalanced Training Data in Complex Wetlands.* **Wetlands** (Springer) — under revision.

```bibtex
@article{soltani2025hybridaug,
  title   = {Hybrid Data Augmentation and Object-Based Analysis for LULC Mapping
             with Imbalanced Training Data in Complex Wetlands},
  author  = {Soltani, M.J. and Latifi, H. and Naghavi, H.},
  journal = {Wetlands},
  year    = {2026},
  doi    = {10.1007/s13157-026-02121-z}
}
```

---

## Contact

| | |
|---|---|
| ✉️ Email | [mjavadsoltani@email.kntu.ac.ir](mailto:mjavadsoltani@email.kntu.ac.ir) |
| 💻 GitHub | [MohammadJavadSoltani](https://github.com/MohammadJavadSoltani) |
| 🔗 LinkedIn | [Mohammad-Javad-Soltani](https://www.linkedin.com/in/Mohammad-Javad-Soltani) |
| 🪪 ORCID | [0009-0007-7912-4661](https://orcid.org/0009-0007-7912-4661) |

---

<div align="center">

*Observe · Analyze · Optimize · Impact*  
</div>

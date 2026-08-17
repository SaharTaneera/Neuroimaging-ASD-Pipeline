# Neuroimaging ASD Pipeline (ABIDE fMRI Analysis)

This repository contains an exploratory machine learning and neuroinformatics pipeline designed to analyze resting-state functional MRI (rs-fMRI) data from the **ABIDE PCP** dataset for Autism Spectrum Disorder (ASD) classification.

## 🧠 Pipeline Architecture
1. **Data Ingestion:** Automatically downloads preprocessed functional pipelines (`cpac`, `func_preproc`) from the ABIDE database.
2. **ROI Parcellation:** Utilizes the **Harvard-Oxford Cortical Atlas** via `nilearn` to extract regional BOLD time series.
3. **Functional Connectivity:** Computes Pearson correlation matrices across brain regions of interest and flattens upper-triangle values into feature vectors.
4. **Classification:** Implements a Support Vector Machine (SVM) pipeline to distinguish neurotypical controls from individuals with ASD.

## ⚙️ Installation & Usage
1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/Neuroimaging-ASD-Pipeline.git](https://github.com/your-username/Neuroimaging-ASD-Pipeline.git)
   cd Neuroimaging-ASD-Pipeline

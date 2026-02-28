# Biomedical Signal Processing Labs (MATLAB + Python)

Atrial fibrillation (AF) lab work covering ECG preprocessing, ventricular activity suppression, and baseline AF recurrence modeling using both MATLAB and Python implementations.

## Project Overview

This folder contains:
- MATLAB implementation for **Questions 1–6** (`Question 1 to 6.m`)
- Python deep-learning notebook for **Question 7** (`Question_7.ipynb`): ventricular activity removal (Rva → Ra)
- Python deep-learning notebook for **Question 8** (`Question_8.ipynb`): ANN-based R-peak detection
- Input data in `.mat` format and output visualizations in `.fig` format

The ECG tensors are organized as 12 leads × 15000 samples × 75 subjects, sampled at **256 Hz**.

---

## Files in This Folder

### Code
- `Question 1 to 6.m`  
  MATLAB master script implementing:
  1. 12-lead visualization (raw and atrial-only)
  2. R-peak detection (`findpeaks`)
  3. QRST alignment and mean template
  4. SVD-based ventricular subtraction
  5. Comparison with mean-template subtraction
  6. Baseline AF recurrence classification (SVM with PSD-based features on V1–V6)

- `Question_7.ipynb`  
  Python/TensorFlow workflow for ventricular removal using a residual 1D CNN:
  - load/merge `Rva*` and `Ra*`
  - windowing and normalization
  - train/validation split
  - denoiser training and qualitative prediction plot

- `Question_8.ipynb`  
  Python/TensorFlow workflow for ANN R-peak detection:
  - pseudo-label generation from simple threshold-based detector
  - Gaussian target creation around peaks
  - windowed dataset preparation
  - 1D CNN probability model training
  - stitched inference on a full signal and peak picking

- `requirements.txt`  
  Python environment snapshot used for notebooks (includes TensorFlow, NumPy, SciPy, Matplotlib, Jupyter stack).

### Input Data (`.mat`)
- `Rva1.mat`, `Rva2.mat`, `Rva3.mat`: ventricular + atrial ECG tensors
- `Ra1.mat`, `Ra2.mat`, `Ra3.mat`: atrial-only ECG tensors (ground truth for subtraction tasks)
- `indrecur.mat`, `indnonrecur.mat`: labels/indices for AF recurrence classification

### Generated MATLAB Figures (`.fig`)
- `Task 1_ 12-lead RAW ECG(Xva).fig`
- `Task 1_ 12-lead Atrial-only ECG(Xa).fig`
- `Task 2 R-peaks.fig`
- `Task 3 QRST segments.fig`
- `Task 3 Mean QRST.fig`
- `Task 4 Singular Values.fig`
- `Task 4_ SVD subtraction on first segment.fig`
- `Task 4_ Compare Ground truth Ra vs SVD residual.fig`
- `Task 5 Mean Subtraction vs SVD subtraction.fig`
- `Task 6 AF Recurrence.fig`

### Results Preview (`Images.png`)
- `Images.png/Task 1_ 12-lead Raw ECG (Xva).png`
- `Images.png/Task 1_ 12-lead Atrial-only ECG (Xa).png`
- `Images.png/Task 2_ R-peaks.png`
- `Images.png/Task 3_ QRST segments (R).png`
- `Images.png/Task 3_ Mean QRST.png`
- `Images.png/Task 4_ Singular values.png`
- `Images.png/Task 4_ SVD subtraction on first segment.png`
- `Images.png/Task 4_ Compare ground-truth Ra vs SVD residual.png`
- `Images.png/Task 5_ Mean subtraction vs SVD subtraction.png`
- `Images.png/Task 6_ Confusion Matrix.png`

---

## Method Summary

### MATLAB (Q1–Q6)
- Loads and concatenates dataset splits into 75-subject tensors
- Performs classical ECG processing for R-peak/QRST analysis
- Uses low-rank SVD model to estimate ventricular component and recover atrial residual
- Compares SVD subtraction with simple mean-template subtraction
- Extracts Welch PSD-based features (dominant frequency, bandpower, spectral entropy) from V1–V6
- Trains an RBF SVM baseline with holdout evaluation for AF recurrence

### Python (Q7)
- Builds paired training windows from (`Rva`, `Ra`)
- Trains residual 1D CNN denoiser with MSE + MAE
- Validates using held-out windows and plots predicted vs input vs target

### Python (Q8)
- Creates pseudo-labels for R-peaks from normalized single-lead ECG
- Trains 1D CNN to output per-sample peak probability
- Performs full-signal inference by chunking + stitching
- Detects final peaks using threshold + refractory distance

---

## Environment Setup (Python)

From this folder:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Open `Question_7.ipynb` and `Question_8.ipynb` in Jupyter/VS Code and run cells in order.

---

## Running MATLAB Script

In MATLAB, set current folder to this directory and run:

```matlab
run('Question 1 to 6.m')
```

This generates the task figures listed above.

---

## Notes

- Notebooks assume the `.mat` files are in the same directory.
- For `Question_8.ipynb`, lead II (`lead = 1`) is selected by default for stronger R-peaks.
- The holdout split in MATLAB Q6 is randomized each run.

---

## Repository Hygiene Recommendation

If pushing to GitHub, avoid committing `venv/` (local environment). Add a `.gitignore` with at least:

```gitignore
venv/
__pycache__/
.ipynb_checkpoints/
```

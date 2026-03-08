# Biomedical Signal Processing Labs (MATLAB + Python)

Atrial fibrillation (AF) lab work covering ECG preprocessing, ventricular activity suppression, and baseline AF recurrence modeling using both MATLAB and Python implementations.

## Quick Navigation

- [Project Overview](#project-overview)
- [Files in This Folder](#files-in-this-folder)
- [Dataset Explanation](#dataset-explanation)
- [Results Preview](#results-preview-imagespng)
- [Method Summary](#method-summary)
- [Environment Setup](#environment-setup-python)
- [Running MATLAB Script](#running-matlab-script)
- [Notes](#notes)

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

## Dataset Explanation

### Core ECG tensors
- Each ECG tensor uses shape: **12 leads × 15000 samples × subjects**.
- Sampling frequency is **256 Hz**, so each recording is ~58.6 seconds long (`15000 / 256`).
- Leads follow standard 12-lead order:
  `I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6`.
- The `Rva` signals are raw (band-pass filtered) ECG with both ventricular and atrial activities.
- The `Ra` signals are processed ECG where ventricular activity is removed, leaving atrial activity (AF-focused reference).

### File split convention
- `Rva1.mat`, `Rva2.mat`, `Rva3.mat` are split parts of ventricular-containing ECGs.
- `Ra1.mat`, `Ra2.mat`, `Ra3.mat` are matching split parts of atrial-only ECGs.
- The scripts/notebooks concatenate `1 + 2 + 3` along the subject axis to form full datasets with **75 subjects**.

### Signal meaning
- **Rva***: mixed ECG where ventricular activity is present.
- **Ra***: atrial-only reference signal (used as target/ground truth for ventricular subtraction tasks).

### Labels for recurrence task
- `indrecur.mat` contains subject indices for recurrence class.
- `indnonrecur.mat` contains subject indices for non-recurrence class.
- After electrical cardioversion and a 6-month blanking period, subjects are grouped by AF recurrence outcome.
- Only **63 subjects** are documented in `indrecur`/`indnonrecur` vectors.
- MATLAB Question 6 uses these indices to build labeled samples, then extracts PSD-based features from leads **V1–V6** (`7:12`) for SVM classification.

### Lab task scope (from original manual)
1. 12-lead ECG visualization (MATLAB)
2. Simple R-wave detection (MATLAB)
3. QRST averaging (MATLAB)
4. Ventricular activity subtraction, SVD-based (MATLAB)
5. Comparison with mean QRST subtraction (no SVD)
6. AF recurrence machine learning on `Ra` (use V1–V6)
7. ANN ventricular activity removal on `Rva` with `Ra` as ground truth (use V1–V6)
8. Optional: ANN-based R detection

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
| Preview A | Preview B |
|---|---|
| **Task 1: 12-lead Raw ECG (Xva)**<br>![Task 1 - 12-lead Raw ECG (Xva)](Images.png/Task%201_%2012-lead%20Raw%20ECG%20%28Xva%29.png) | **Task 1: 12-lead Atrial-only ECG (Xa)**<br>![Task 1 - 12-lead Atrial-only ECG (Xa)](Images.png/Task%201_%2012-lead%20Atrial-only%20ECG%20%28Xa%29.png) |
| **Task 2: R-peaks**<br>![Task 2 - R-peaks](Images.png/Task%202_%20R-peaks.png) | **Task 3: QRST segments (R)**<br>![Task 3 - QRST segments (R)](Images.png/Task%203_%20QRST%20segments%20%28R%29.png) |
| **Task 3: Mean QRST**<br>![Task 3 - Mean QRST](Images.png/Task%203_%20Mean%20QRST.png) | **Task 4: Singular values**<br>![Task 4 - Singular values](Images.png/Task%204_%20Singular%20values.png) |
| **Task 4: SVD subtraction on first segment**<br>![Task 4 - SVD subtraction on first segment](Images.png/Task%204_%20SVD%20subtraction%20on%20first%20segment.png) | **Task 4: Compare ground-truth Ra vs SVD residual**<br>![Task 4 - Compare ground-truth Ra vs SVD residual](Images.png/Task%204_%20Compare%20ground-truth%20Ra%20vs%20SVD%20residual.png) |
| **Task 5: Mean subtraction vs SVD subtraction**<br>![Task 5 - Mean subtraction vs SVD subtraction](Images.png/Task%205_%20Mean%20subtraction%20vs%20SVD%20subtraction.png) | **Task 6: Confusion Matrix**<br>![Task 6 - Confusion Matrix](Images.png/Task%206_%20Confusion%20Matrix.png) |

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

## Acknowledgment

Parts of the dataset description and task framing in this README are adapted from the original AF lab manual provided for the course.

---


```gitignore
venv/
__pycache__/
.ipynb_checkpoints/
```

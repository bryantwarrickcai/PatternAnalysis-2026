# 2D U-Net for Prostate Segmentation on HipMRI Slices

**Course:** COMP3710 — Pattern Recognition
**Project:** 2.2 Segmentation — HipMRI 2D Prostate Slices
**Difficulty Band:** [Easy Difficulty]
**Student:** Bryant Warrick Cai | **ID:** 49951790
**Repository:** `recognition/2d_unet_49951790`

---

## 1. Problem Statement

Targeted external-beam radiotherapy for prostate cancer requires millimeter-level delineation of the prostate clinical target volume while tightly sparing adjacent Organs at Risk (OARs) such as the rectum and bladder. Manual contouring is time-consuming and prone to inter-observer variability. 

This project implements an automated 2D segmentation assistant using coronal MRI slices from the HipMRI Study. The core engineering dilemma investigated is the **Spatial Context vs. Clinical Boundary Utility** trade-off: does a standard 2D U-Net with skip connections genuinely eliminate hazardous boundary leakage and slice-transition errors at the prostate base and apex compared to a simpler baseline model, or does it merely smooth easy central slices? We benchmark the U-Net against a baseline CNN without skip connections to quantify the clinical value of spatial context.

## 2. How the Algorithm Works

The primary model is the standard 2D U-Net. It consists of a contracting encoder path and an expanding decoder path. The encoder applies repeated 3×3 convolutions followed by 2×2 max pooling to extract hierarchical features and capture global pelvic context. The decoder upsamples these features using transposed convolutions. Crucially, **skip connections** concatenate high-resolution feature maps from the encoder directly into the decoder. This allows the network to combine coarse semantic information (what/where the prostate is) with fine spatial detail (the exact boundary of the prostate capsule). The final layer is a 1×1 convolution outputting raw logits for 6 classes (background, body, bone, bladder, rectum, prostate).

### Training and Inference Pipeline
1. **Data:** Nifti slices are parsed, paired with ground-truth masks, and center-padded/cropped to 256×160. Patient-level splitting prevents data leakage.
2. **Loss:** A combination of soft Dice loss and Cross-Entropy (weighted 0.5/0.5). Dice loss ensures the model handles extreme class imbalance (the prostate is <2% of the image).
3. **Evaluation:** We report both macro Dice/IoU and specifically the prostate Dice on slices where the prostate is actually present (ignoring empty slices that inflate scores).

## 3. Project Structure

| File | Purpose |
|---|---|
| `modules.py` | U-Net architecture (`DoubleConv`, `Down`, `Up`, `UNet`) built in PyTorch. |
| `baseline.py` | Simple encoder-decoder CNN *without* skip connections for comparison. |
| `dataset.py` | Nifti loading, pairing, patient-split audit, normalization, and DataLoaders. |
| `train.py` | Training loop, combined Dice+CE loss, scheduler, and checkpointing. |
| `predict.py` | Test set inference, per-slice CSV metrics, and 4-panel overlay visualization. |
| `failure_analysis.py` | Automatically extracts and visualizes the worst-performing slices. |
| `profile_model.py` | Profiles parameters, VRAM, and inference latency. |
| `train_all_rangpur.sh` | Slurm batch script for reproducing the full pipeline on the Rangpur HPC cluster. |
| `train_all_direct.sh` | Bash script for running the full pipeline directly on a standard Linux GPU machine. |
| `check_model.py` | Standalone smoke test for model instantiation and forward pass. |

## 4. How to Run

**Dependencies:**
Ensure you have the correct Python environment (see `requirements.txt`):
```bash
pip install -r requirements.txt
```

**Data Setup:**
The dataset must be placed in this directory (`recognition/2d_unet_49951790`) under `HipMRI_Study_open/keras_slices_data`. 

**Option A: Direct GPU Environment**
If running on a standard Linux machine with a GPU, simply execute the provided pipeline script:
```bash
chmod +x train_all_direct.sh
./train_all_direct.sh
```

**Option B: Rangpur HPC (Slurm)**
If running on the university's Rangpur cluster, submit the Slurm batch script:
```bash
sbatch train_all_rangpur.sh
```

---

## 5. Artificial Intelligence Usage Disclosure

In accordance with the course policy on responsible AI use, the following tools were utilized during this project:

- **Tools Used:** DeepSeek
- **Specific Tasks Assisted:**
  - Scaffolding boilerplate code for the PyTorch `Dataset` and `DataLoader` classes.
  - Debugging PyTorch 2.6+ `weights_only=True` checkpoint loading issues.
  - Drafting the Matplotlib visualization functions for the 4-panel error overlays.
- **Verification & Auditing:**
  - **The AI-suggested code was manually reviewed and tested.** 
  - The data loading pipeline was audited against the actual HipMRI data distribution (`dataset.py --discover-labels`) to verify the label mapping (Prostate = 5).
  - The loss function was unit-tested to ensure absent classes in the target did not produce NaN values.
  - The scheduler behavior (ReduceLROnPlateau) was empirically tested to confirm off-by-one patience logic.

---
*(Sections 6 onwards — Feasibility Review, Quantitative Results, Failure Autopsy, and Resource Profiling — will be added in Step 6.2)*

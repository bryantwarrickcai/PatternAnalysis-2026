"""
dataset.py
==========
Data pipeline for the HipMRI 2D prostate segmentation task

This file will contain:
- Filename parsing for case_ABC_week_D_slice_E.nii.gz / seg_* pairs
- Image/mask pairing and patient-level split auditing
- Nifti loading, intensity normalization, one-hot mask encoding
- A torch.utils.data.Dataset subclass and get_dataloaders() helper
- Train-only augmentation (flips, small rotations, intensity jitter)
"""

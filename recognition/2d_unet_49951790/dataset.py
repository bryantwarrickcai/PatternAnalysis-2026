"""
dataset.py
==========
Data pipeline for the HipMRI 2D prostate segmentation task

This module currently contains:
- Filename parsing for case_ABC_week_D_slice_E.nii.gz / seg_* pairs
- Image/mask pairing (one PairedSlice per (patient, week, slice))
- Per-split discovery of paired slices with deterministic ordering

Later steps will add patient-level split auditing, Nifti loading,
intensity normalization, one-hot mask encoding, the torch Dataset
subclass, and get_dataloaders() with train-only augmentation.

Filename convention (see the HipMRI Study data use agreement):
    case_<patient>_week_<week>_slice_<slice>.nii.gz  (image)
    seg_<patient>_week_<week>_slice_<slice>.nii.gz   (mask)
where <patient> is exactly 3 digits, and <week>/<slice> are 1+ digits.
"""

from __future__ import annotations

import argparse
import re
import nibabel as nib
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


IMAGE_FOLDER_TEMPLATE = "keras_slices_{split}"
MASK_FOLDER_TEMPLATE = "keras_slices_seg_{split}"
SPLITS: tuple[str, ...] = ("train", "validate", "test")

_CASE_RE = re.compile(
    r"^case_(?P<patient>\d{3})_week_(?P<week>\d+)_slice_(?P<slice>\d+)\.nii\.gz$"
)
_SEG_RE = re.compile(
    r"^seg_(?P<patient>\d{3})_week_(?P<week>\d+)_slice_(?P<slice>\d+)\.nii\.gz$"
)

# ---------------------------------------------------------------------------
# Label scheme
# ---------------------------------------------------------------------------
# Verified against:
#   - HipMRI data use agreement (HipMRI_Study_open/data_use_agreement.txt):
#       "The body outline, bone, bladder, rectum, and prostate regions
#        are segmented in 3D by an MR physicist ..."
#   - Reference paper for this dataset (MRI image synthesis track):
#       Liu et al., "Manipulating Medical Image Translation with Manifold
#       Disentanglement", arXiv:2011.13615, 2020, Sec. III.D:
#         "Each 3D image is 256x256x128 in resolution and is manually
#          labelled with five foreground classes: body, bone, bladder,
#          rectum and prostate."
#   - Empirical label discovery on keras_slices_seg_train
#     (`python dataset.py --discover-labels`), which produced pixel
#     distributions consistent with the reference (body ~68%, bone ~12%,
#     bladder ~2%, rectum ~0.4%, prostate ~0.4%).
# Nifti headers do NOT store label names (descrip/aux_file/intent_name
# are empty, intent_code = 1002 "label map"). Any labels displayed by a
# web viewer are therefore heuristic guesses, not file content.
#
# Note: the dataset is a **coronal** acquisition (see paper, Sec. III.D:
# "256x128 images from the centre 40 slices of the coronal plane").
# This restricts valid augmentation: left-right flip is anatomically
# safe; superior-inferior flip is NOT.
LABEL_NAMES: dict[int, str] = {
    0: "background",
    1: "body",
    2: "bone",
    3: "bladder",
    4: "rectum",
    5: "prostate",
}
NUM_CLASSES: int = len(LABEL_NAMES)
PROSTATE_LABEL: int = 5  # required by Project 2.2 (Dice >= 0.75 target)


@dataclass(frozen=True, order=True)
class SliceKey:
    """Identifies a single 2D slice across image and mask folders."""

    patient: str
    week: int
    slice_index: int

    def __str__(self) -> str:
        return f"case_{self.patient}_week_{self.week}_slice_{self.slice_index}"


@dataclass(frozen=True)
class PairedSlice:
    """An image/mask pair sharing the same (patient, week, slice) key."""

    key: SliceKey
    image_path: Path
    mask_path: Path


def _parse_filename(pattern: re.Pattern[str], filename: str) -> SliceKey:
    match = pattern.match(filename)
    if match is None:
        raise ValueError(f"Unexpected filename: {filename!r}")
    return SliceKey(
        patient=match.group("patient"),
        week=int(match.group("week")),
        slice_index=int(match.group("slice")),
    )


def parse_case_filename(filename: str) -> SliceKey:
    """Parse ``case_ABC_week_D_slice_E.nii.gz`` into a :class:`SliceKey`."""
    return _parse_filename(_CASE_RE, filename)


def parse_seg_filename(filename: str) -> SliceKey:
    """Parse ``seg_ABC_week_D_slice_E.nii.gz`` into a :class:`SliceKey`."""
    return _parse_filename(_SEG_RE, filename)


def _nifti_files(folder: Path) -> Iterable[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing folder: {folder}")
    for path in folder.iterdir():
        if path.is_file() and path.name.endswith(".nii.gz"):
            yield path


def pair_split(data_root: Path | str, split: str) -> list[PairedSlice]:
    """
    Discover and pair image/mask files for a single split.

    Parameters
    ----------
    data_root:
        Folder that contains ``keras_slices_<split>`` and
        ``keras_slices_seg_<split>`` (e.g. ``.../keras_slices_data``).
    split:
        One of ``"train"``, ``"validate"``, or ``"test"``.

    Returns
    -------
    Sorted list of :class:`PairedSlice`, ordered by
    ``(patient, week, slice_index)``. Raises ``ValueError`` if an
    image has no matching mask, or if two files resolve to the same key.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")

    root = Path(data_root)
    image_dir = root / IMAGE_FOLDER_TEMPLATE.format(split=split)
    mask_dir = root / MASK_FOLDER_TEMPLATE.format(split=split)

    images: dict[SliceKey, Path] = {}
    for path in _nifti_files(image_dir):
        key = parse_case_filename(path.name)
        if key in images:
            raise ValueError(
                f"Duplicate image key {key} in {image_dir}: "
                f"{images[key].name} vs {path.name}"
            )
        images[key] = path

    masks: dict[SliceKey, Path] = {}
    for path in _nifti_files(mask_dir):
        key = parse_seg_filename(path.name)
        if key in masks:
            raise ValueError(
                f"Duplicate mask key {key} in {mask_dir}: "
                f"{masks[key].name} vs {path.name}"
            )
        masks[key] = path

    image_only = sorted(images.keys() - masks.keys())
    mask_only = sorted(masks.keys() - images.keys())
    if image_only:
        raise ValueError(
            f"{len(image_only)} image(s) in split {split!r} have no mask, "
            f"e.g. {image_only[0]}"
        )
    if mask_only:
        raise ValueError(
            f"{len(mask_only)} mask(s) in split {split!r} have no image, "
            f"e.g. {mask_only[0]}"
        )

    return [
        PairedSlice(key=key, image_path=images[key], mask_path=masks[key])
        for key in sorted(images)
    ]


def pair_all_splits(data_root: Path | str) -> dict[str, list[PairedSlice]]:
    """Convenience wrapper returning ``{split: list[PairedSlice]}``."""
    return {split: pair_split(data_root, split) for split in SPLITS}


def patients_in_split(data_root: Path | str, split: str) -> set[str]:
    """Return the unique patient IDs (``ABC``) present in a single split."""
    return {pair.key.patient for pair in pair_split(data_root, split)}


# ---------------------------------------------------------------------------
# Nifti loading
# ---------------------------------------------------------------------------
def _load_nifti_array(path: Path) -> np.ndarray:
    """
    Load a Nifti file as a 2D numpy array.

    The 2D slices in ``keras_slices_data`` are single-slice volumes.
    We squeeze any trivial (size-1) trailing axes that nibabel exposes,
    then require a 2D array. Affine is not returned because we operate
    on 2D slices directly; downstream tasks don't need world coordinates.
    """
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)  # preserves integer dtype for labels
    data = np.squeeze(data)
    if data.ndim != 2:
        raise ValueError(
            f"Expected a 2D slice, got shape {data.shape} for {path.name}"
        )
    return data


def load_image_slice(path: Path) -> np.ndarray:
    """Load an image slice as a 2D float32 array (no normalization)."""
    return _load_nifti_array(path).astype(np.float32, copy=False)


def load_mask_slice(path: Path) -> np.ndarray:
    """
    Load a label slice as a 2D int64 array.

    HipMRI masks can be stored as float Nifti; we round to the nearest
    integer so label values are exact (e.g. 5.0 -> 5). Integer-stored
    masks pass through unchanged.
    """
    raw = _load_nifti_array(path)
    if np.issubdtype(raw.dtype, np.floating):
        raw = np.rint(raw)
    return raw.astype(np.int64, copy=False)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def normalize_image(image: np.ndarray, method: str = "zscore") -> np.ndarray:
    """
    Normalize an image to a stable input range.

    zscore : (x - mean) / std, with a safe denominator.
    minmax : (x - min) / (max - min), with a safe denominator.
    """
    image = image.astype(np.float32, copy=False)
    if method == "zscore":
        mean = float(image.mean())
        std = float(image.std())
        return (image - mean) / max(std, 1e-8)
    if method == "minmax":
        lo = float(image.min())
        hi = float(image.max())
        return (image - lo) / max(hi - lo, 1e-8)
    raise ValueError(f"Unknown normalization method: {method!r}")


def one_hot_encode(mask: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """
    Convert an (H, W) integer label map to an (H, W, C) float32 one-hot stack.

    Labels outside ``[0, num_classes)`` produce all-zero vectors on every
    channel (treated as "ignore" by the loss in train.py).

    Note: not every slice contains every class. On the coronal plane,
    the bladder (label 3) and rectum (label 4) are often absent on any
    given slice, and the prostate (label 5) is absent on the apex/base
    slices. This is expected data, not an error — filtering logic and
    per-class validity handling live in train.py, not here.
    """
    h, w = mask.shape
    onehot = np.zeros((h, w, num_classes), dtype=np.float32)
    for c in range(num_classes):
        onehot[..., c] = (mask == c).astype(np.float32)
    return onehot


def prostate_binary(
    mask: np.ndarray, prostate_label: int = PROSTATE_LABEL
) -> np.ndarray:
    """Return a 2D binary mask (1.0 on prostate, 0.0 elsewhere), float32."""
    return (mask == prostate_label).astype(np.float32)


def preprocess_pair(
    pair: PairedSlice,
    num_classes: int = NUM_CLASSES,
    prostate_label: int = PROSTATE_LABEL,
    norm_method: str = "zscore",
) -> dict[str, np.ndarray]:
    """
    Load and preprocess one PairedSlice.

    Returns a dict with:
      image       : (H, W)      float32  normalized image
      mask_int    : (H, W)      int64    raw labels (for metrics / audit)
      mask_onehot : (H, W, C)   float32  one-hot mask
      prostate    : (H, W)      float32  binary prostate mask
    """
    image = load_image_slice(pair.image_path)
    mask_int = load_mask_slice(pair.mask_path)
    if image.shape != mask_int.shape:
        raise ValueError(
            f"Shape mismatch for {pair.key}: "
            f"image {image.shape} vs mask {mask_int.shape}"
        )
    image = normalize_image(image, method=norm_method)
    mask_onehot = one_hot_encode(mask_int, num_classes=num_classes)
    prostate = prostate_binary(mask_int, prostate_label=prostate_label)
    return {
        "image": image,
        "mask_int": mask_int,
        "mask_onehot": mask_onehot,
        "prostate": prostate,
    }


# ---------------------------------------------------------------------------
# Label discovery (used to verify LABEL_NAMES against real data)
# ---------------------------------------------------------------------------
def discover_labels(
    data_root: Path | str,
    split: str = "train",
    num_samples: int | None = None,
) -> dict[int, int]:
    """
    Scan mask files in a split and return total pixel counts per label value.

    Run this before training and compare against :data:`LABEL_NAMES`.
    Unexpected label values (e.g. ``-1`` or ``6``) will show up here.
    """
    pairs = pair_split(data_root, split)
    if num_samples is not None:
        pairs = pairs[:num_samples]
    counts: dict[int, int] = {}
    for pair in pairs:
        labels, per_label = np.unique(
            load_mask_slice(pair.mask_path), return_counts=True
        )
        for label, count in zip(labels.tolist(), per_label.tolist()):
            key = int(label)
            counts[key] = counts.get(key, 0) + int(count)
    return counts


@dataclass(frozen=True)
class SplitAudit:
    """Result of a patient-level split audit."""

    patient_ids: dict[str, set[str]]  # split -> patient IDs
    pair_counts: dict[str, int]       # split -> number of paired slices
    overlaps: dict[str, set[str]]     # "train∩validate" -> leaked patient IDs

    @property
    def is_clean(self) -> bool:
        """True iff no patient ID appears in more than one split."""
        return all(len(ids) == 0 for ids in self.overlaps.values())


def audit_patient_splits(data_root: Path | str) -> SplitAudit:
    """
    Check for patient-level leakage across train/validate/test.

    A slice split is *clean* when no patient ID (``ABC``) appears in more
    than one split. Because a patient can contribute multiple weeks and
    many slices, any overlap here would inflate validation/test Dice
    relative to true held-out performance.

    Returns a :class:`SplitAudit`; does not raise on overlap. Use
    :func:`assert_no_patient_overlap` when leakage should be fatal
    (e.g. at the start of a training run).
    """
    ids = {split: patients_in_split(data_root, split) for split in SPLITS}
    pair_counts = {split: len(pair_split(data_root, split)) for split in SPLITS}
    overlaps = {
        "train∩validate": ids["train"] & ids["validate"],
        "train∩test": ids["train"] & ids["test"],
        "validate∩test": ids["validate"] & ids["test"],
    }
    return SplitAudit(
        patient_ids=ids, pair_counts=pair_counts, overlaps=overlaps
    )


def assert_no_patient_overlap(data_root: Path | str) -> SplitAudit:
    """
    Raise ``RuntimeError`` if any patient ID appears in more than one split.

    Intended as a startup guard in training scripts so leakage is
    impossible by construction.
    """
    audit = audit_patient_splits(data_root)
    if not audit.is_clean:
        details = "; ".join(
            f"{name}: {sorted(ids)}"
            for name, ids in audit.overlaps.items()
            if ids
        )
        raise RuntimeError(f"Patient-level leakage detected ({details})")
    return audit


def _main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect HipMRI 2D slices: pairing, patient-level audit, "
            "label discovery, and preprocessing smoke test."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("HipMRI_Study_open/keras_slices_data"),
        help="Folder containing keras_slices_* subfolders. Default assumes "
        "the script is run from recognition/2d_unet_49951790/.",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Patient-level leakage audit across train/validate/test.",
    )
    parser.add_argument(
        "--discover-labels",
        action="store_true",
        help="Report pixel counts per label value on the train split.",
    )
    parser.add_argument(
        "--discover-samples",
        type=int,
        default=None,
        help="Limit --discover-labels to the first N masks (faster).",
    )
    parser.add_argument(
        "--preprocess-check",
        action="store_true",
        help="Load one training pair end-to-end and print diagnostics.",
    )
    args = parser.parse_args()

    # --- Pairing summary (always shown) ------------------------------------
    for split in SPLITS:
        pairs = pair_split(args.data_root, split)
        patients = sorted({pair.key.patient for pair in pairs})
        print(f"{split:9s}  pairs={len(pairs):5d}  patients={len(patients):3d}")
        if pairs:
            print(f"  first: {pairs[0].key}")
            print(f"  last : {pairs[-1].key}")

    # --- Patient-level audit ----------------------------------------------
    if args.audit:
        print()
        audit = audit_patient_splits(args.data_root)
        print("Patient-level split audit:")
        for split in SPLITS:
            ids = sorted(audit.patient_ids[split])
            print(
                f"  {split:9s}  {audit.pair_counts[split]:5d} pairs  "
                f"{len(ids):3d} patients  IDs={ids}"
            )
        print()
        if audit.is_clean:
            print("Result: CLEAN - no patient ID appears in more than one split.")
        else:
            print("Result: LEAKAGE DETECTED")
            for name, ids in audit.overlaps.items():
                if ids:
                    print(f"  {name}: {sorted(ids)}")
            raise SystemExit(1)

    # --- Label discovery ---------------------------------------------------
    if args.discover_labels:
        print("\nLabel counts on train masks:")
        counts = discover_labels(
            args.data_root, split="train", num_samples=args.discover_samples
        )
        if not counts:
            print("  no masks found")
        else:
            total = sum(counts.values())
            for label in sorted(counts):
                name = LABEL_NAMES.get(label, "UNKNOWN")
                pct = 100.0 * counts[label] / total if total else 0.0
                print(
                    f"  label {label:3d} ({name:9s}): "
                    f"{counts[label]:>12,d} pixels  ({pct:6.3f}%)"
                )
            unknown = [v for v in counts if v not in LABEL_NAMES]
            if unknown:
                print(
                    f"\n  WARNING: unknown label values {unknown}. "
                    f"Update LABEL_NAMES / NUM_CLASSES / PROSTATE_LABEL."
                )

    # --- Preprocessing smoke test -----------------------------------------
    if args.preprocess_check:
        pairs = pair_split(args.data_root, "train")
        if not pairs:
            print("\nNo training pairs found.")
            return
        out = preprocess_pair(pairs[0])
        print(f"\nPreprocess check ({pairs[0].key}):")
        print(
            f"  image        : shape={out['image'].shape}, "
            f"dtype={out['image'].dtype}, "
            f"min={out['image'].min():.3f}, max={out['image'].max():.3f}"
        )
        uniq = sorted(set(out["mask_int"].ravel().tolist()))
        print(f"  mask_int     : shape={out['mask_int'].shape}, labels={uniq}")
        per_class = out["mask_onehot"].sum(axis=(0, 1)).astype(int).tolist()
        print(f"  mask_onehot  : shape={out['mask_onehot'].shape}, "
              f"per-class pixel counts={per_class}")
        print(
            f"  prostate     : shape={out['prostate'].shape}, "
            f"positive fraction={float(out['prostate'].mean()):.6f}"
        )


if __name__ == "__main__":
    _main()

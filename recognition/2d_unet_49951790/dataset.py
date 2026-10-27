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


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect paired HipMRI 2D slices per split."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("HipMRI_Study_open/keras_slices_data"),
        help="Folder containing keras_slices_* subfolders. Default assumes "
        "the script is run from recognition/2d_unet_49951790/.",
    )
    args = parser.parse_args()

    for split in SPLITS:
        pairs = pair_split(args.data_root, split)
        patients = sorted({pair.key.patient for pair in pairs})
        print(f"{split:9s}  pairs={len(pairs):5d}  patients={len(patients):3d}")
        if pairs:
            print(f"  first: {pairs[0].key}")
            print(f"  last : {pairs[-1].key}")


if __name__ == "__main__":
    _main()

"""
Lung CT Processing Pipeline for LIDC-IDRI
==========================================
Purpose : Load raw DICOM CT series and save NIfTI images for downstream
          U-Net training. Mask generation is handled separately.

Pipeline
--------
1.  Find the axial CT series (most slices, axial orientation check).
2.  Load & sort DICOM slices by true z-position (ImagePositionPatient).
3.  Convert to Hounsfield Units → store as int16 (no normalisation).
4.  Save CT as int16 NIfTI.
"""

import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

LIDC_ROOT   = Path(r"D:\PFE\DATA\manifest-1600709154662\LIDC-IDRI")
OUTPUT_ROOT = Path(r"D:\PFE\DATA\LIDC_processed")

# part1 through part4 are processed independently so each output folder
# can be compressed and uploaded separately.
PARTS = ["part1", "part2", "part3", "part4"]

MIN_SERIES_SLICES = 20  # ignore DICOM folders with fewer slices

# ──────────────────────────────────────────────────────────────────────────────
# DICOM I/O
# ──────────────────────────────────────────────────────────────────────────────

def find_main_series(patient_dir: Path) -> Path | None:
    """
    Return the axial DICOM sub-folder with the most slices (>= MIN_SERIES_SLICES).
    Checks ImageOrientationPatient to skip scout/localizer series.
    """
    best, best_count = None, 0

    for sub in patient_dir.rglob("*"):
        if not sub.is_dir():
            continue
        dcm_files = list(sub.glob("*.dcm"))
        if len(dcm_files) < MIN_SERIES_SLICES:
            continue
        try:
            ds  = pydicom.dcmread(dcm_files[0], stop_before_pixels=True)
            iop = [round(float(v)) for v in ds.ImageOrientationPatient]
            if iop != [1, 0, 0, 0, 1, 0]:  # standard axial orientation
                continue
        except Exception:
            pass  # missing header — still consider the folder
        if len(dcm_files) > best_count:
            best_count = len(dcm_files)
            best = sub

    return best if best_count >= MIN_SERIES_SLICES else None


def load_sorted_slices(series_dir: Path) -> list[pydicom.Dataset]:
    """Read all DICOMs in a folder and sort by true z-position."""
    slices = [pydicom.dcmread(f) for f in sorted(series_dir.glob("*.dcm"))]
    try:
        slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    except AttributeError:
        warnings.warn(f"ImagePositionPatient missing in {series_dir} – falling back to InstanceNumber.")
        slices.sort(key=lambda s: int(getattr(s, "InstanceNumber", 0)))
    return slices


def to_hu(slices: list[pydicom.Dataset]) -> np.ndarray:
    """
    Convert raw pixel data to Hounsfield Units and return as int16.

    HU = pixel_value * RescaleSlope + RescaleIntercept
    Arithmetic in float32 avoids integer overflow; clipped to int16 range
    before casting so the stored values are lossless within that range.
    """
    volume = np.stack([s.pixel_array for s in slices]).astype(np.float32)
    for i, s in enumerate(slices):
        slope     = float(getattr(s, "RescaleSlope",     1))
        intercept = float(getattr(s, "RescaleIntercept", 0))
        volume[i] = volume[i] * slope + intercept
    return np.clip(volume, -32768, 32767).astype(np.int16)


def build_affine(slices: list[pydicom.Dataset]) -> np.ndarray:
    """
    Build a diagonal NIfTI affine from pixel spacing and computed slice spacing.
    Slice spacing is the median of consecutive z-differences from
    ImagePositionPatient — more reliable than the SliceThickness header.
    """
    try:
        px, py = map(float, slices[0].PixelSpacing)
        if len(slices) > 1:
            z_pos = [float(s.ImagePositionPatient[2]) for s in slices]
            diffs = np.abs(np.diff(z_pos))
            dz    = float(np.median(diffs))
            if np.std(diffs) > 0.1 * dz:
                warnings.warn("Irregular slice spacing detected – affine dz is the median spacing.")
        else:
            dz = float(slices[0].SliceThickness)
        return np.diag([px, py, dz, 1.0])
    except Exception:
        warnings.warn("Could not build affine from DICOM headers – using identity.")
        return np.eye(4)


# ──────────────────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────────────────

def process_part(part: str) -> None:
    """Process one data part and write outputs to its own sub-folder."""
    data_root  = LIDC_ROOT / part
    images_out = OUTPUT_ROOT / part / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    patients = sorted(p for p in data_root.iterdir() if p.is_dir())
    print(f"  Found {len(patients)} patient(s)\n")

    ok = skipped = 0

    for patient_dir in patients:
        name = patient_dir.name
        print(f"  Processing {name} ...", end=" ", flush=True)

        # Resume support — skip if output file already exists
        img_path = images_out / f"{name}_img.nii.gz"
        if img_path.exists():
            print("SKIP  (already processed)")
            ok += 1
            continue

        series_dir = find_main_series(patient_dir)
        if series_dir is None:
            print("SKIP  (no valid axial DICOM series found)")
            skipped += 1
            continue

        slices = load_sorted_slices(series_dir)
        volume = to_hu(slices)
        affine = build_affine(slices)

        nib.save(nib.Nifti1Image(volume, affine), img_path)

        print(f"OK  ({len(slices)} slices)")
        ok += 1

    print(f"  Done — {ok} processed, {skipped} skipped.\n")


def main() -> None:
    for part in PARTS:
        data_root = LIDC_ROOT / part
        if not data_root.exists():
            print(f"[{part}] SKIP — folder not found: {data_root}")
            continue
        print(f"{'─' * 60}")
        print(f"[{part}]  {data_root}")
        print(f"{'─' * 60}")
        process_part(part)


if __name__ == "__main__":
    main()

# LUNG-DIGITAL-TWIN

Data was collected from the official website https://www.cancerimagingarchive.net/collection/lidc-idri and was dowloaded using the NBIA data retriever 

## data-proc.py
- Slice sorting
- HU conversion
- Affine matrix construction and NIfTI conversion

https://huggingface.co/datasets/hourouu/LIDC_IDRI_PROCESSED

## mask.py
- Thresholding [-1000, -400]
- Outside air removal
- 2D morphological hole filling
- Column density calculation and valley split
- Morphological closing disk(5)
- Seam heal and morphological closing disk(2) for edge smoothing
- Connected component analysis and 3D morphological smoothing

https://huggingface.co/datasets/hourouu/LIDC_IDRI_MASKS

## resample.py
- Isotropic resampling for CT slices using trilinear interpolation
- Nearest neighbor resampling for masks

https://huggingface.co/datasets/hourouu/LIDC_IDRI_res

## normalize.py
- HU clipping [-1000, 200] to isolate pulmonary regions
- Min-max normalization [0, 1]

https://huggingface.co/datasets/hourouu/LIDC_IDRI_NORMALIZED

## data_prep.py
- Selection of 40% of valid slices (non-empty) capped at 100 slices max and 20 min
- 10% ratio of empty slices
- Resizing to 128x128x128
- Saving to .h5 format

https://huggingface.co/datasets/hourouu/index

## training.py
- **Input:** 128×128 grayscale CT slices, normalised to [0, 1]
- **Output:** 128×128 binary lung mask
- **Dataset:** 108,555 slices from 996 patients (LIDC-IDRI), stored as a single `slices.h5` HDF5 file on HuggingFace
- **Split:** Patient-level (no patient appears in more than one split) — 80% train / 15% val / 5% test
- **Augmentation:** horizontal flip, vertical flip, rotation ±15°, brightness/contrast jitter, Gaussian blur, Gaussian noise
- **Architecture:** UNetV2 — 4 encoder levels (32→64→128→256 channels), bottleneck (512 channels), 4 symmetric decoder levels with skip connections
  

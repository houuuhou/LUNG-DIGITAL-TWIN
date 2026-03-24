# LUNG-DIGITAL-TWIN

## data-proc.py
- Slice sorting
- HU conversion
- Affine matrix construction and NIfTI conversion

## mask.py
- Thresholding [-1000, -400]
- Outside air removal
- 2D morphological hole filling
- Column density calculation and valley split
- Morphological closing disk(5)
- Seam heal and morphological closing disk(2) for edge smoothing
- Connected component analysis and 3D morphological smoothing

## resample.py
- Isotropic resampling for CT slices using trilinear interpolation
- Nearest neighbor resampling for masks

## normalize.py
- HU clipping [-1000, 200] to isolate pulmonary regions
- Min-max normalization [0, 1]

## data_prep.py
- Selection of 40% of valid slices (non-empty) capped at 100 slices max and 20 min
- 10% ratio of empty slices
- Resizing to 128x128x128
- Saving to .h5 format

## training.py
- ...

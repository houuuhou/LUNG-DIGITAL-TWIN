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
- Selection of valid slices (non-empty) capped at 120 slices max and 20 min
- 10% ratio of empty slices
- Resizing to 128x128x128
- Saving to .h5 format

## 2d_Unet.py
- **Input:** 128×128 grayscale CT slices, normalised to [0, 1]
- **Output:** 128×128 binary lung mask
- **Dataset:** 106,238 slices from 996 patients (LIDC-IDRI), stored as a single `slices.h5` HDF5 file on HuggingFace
- **Split:** Patient-level (no patient appears in more than one split) — 80% train / 10% val / 10% test
- **Augmentation** — Spatial (flip, rotate ±25°, elastic, grid distortion) and intensity (brightness/contrast, blur, noise, coarse dropout) transforms applied to the training set only.
- **Architecture:** 4 encoder levels (32→64→128→256 channels), bottleneck (512 channels), 4 symmetric decoder levels with skip connections; each level uses two 3×3 convolutions with Instance Normalisation and ReLU activations.
- **Loss** — Combined 50 % BCE + 50 % Dice loss, which balances pixel-level accuracy with overlap quality.
- **Optimiser** — AdamW (`lr=1e-4`, `weight_decay=1e-4`) with a linear warmup followed by cosine annealing with warm restarts.
https://huggingface.co/datasets/hourouu/model4/tree/main
  
   Here is a visualization of the results obtained by the Unet :
<p  align="center"><img width="555" height="555" alt="image" src="https://github.com/user-attachments/assets/783875a6-ed81-46f0-8c26-bd5629d70076" /></p>


## simulation.py
* Tetrahedral meshes of both lungs loaded from patient-specific `.msh` files
* Displacement boundary conditions applied per cycle; SI: 10–25 mm, ventral: 5 mm, lateral: 2 mm
* Sinusoidal breathing waveform — 1.3 s inspiration / 2.6 s expiration (~15 breaths/min)
* Both lungs simulated independently and synchronised via a shared coordinator
* Per-cycle metrics logged over 7 cycles: tidal volume, VT/TLC ratio, FRC, diaphragm displacement, volumetric strain
* Clinical validation target: VT within [400,500] ml

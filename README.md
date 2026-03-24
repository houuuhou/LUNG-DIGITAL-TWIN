# LUNG-DIGITAL-TWIN
data-proc.py: 
-slice sorting 
-hu conversion
-affine matrix construction and nifti conversion
mask.py:
-threasholding [-1000,-400]
-outside air removal
-2d morphological hole filling
-column density calculation and smart splitting
-disk(5) morphological closing
-seam heal and disk(2) morphological closing for smoothing edges
-connected component check and 3d morphological smoothing
resample.py:
-isotropic resampling for the slices using trilinear interpolation
-nearest neighbor resampling for masks
normalize.py:
-hu clipping [-1000,200] to eliminate bones and soft tissue and isolate pulmunary regions
-minmax normalization [0,1]

data prep:
-select 40% of valid slices (non empty slices) with a cap of 100 slices max and 20 min 
-10% ratio of empty slices 
-resizing to 128x128x128
-saving into .h5 format 

 training:
...

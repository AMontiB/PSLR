# EXPLOITING PRNU AND LINEAR PATTERNS IN FORENSIC CAMERA ATTRIBUTION UNDER COMPLEX LENS DISTORTION CORRECTION

This is the official code implementation of the "ICIP 2022" paper ["EXPLOITING PRNU AND LINEAR PATTERNS IN FORENSIC CAMERA ATTRIBUTION
UNDER COMPLEX LENS DISTORTION CORRECTION"](https://ieeexplore.ieee.org/abstract/document/10096605)

## Requirements

- Download the python libraries of [Camera-fingerprint](https://dde.binghamton.edu/download/camera_fingerprint/) ;
 - if [Camera-fingerprint](https://dde.binghamton.edu/download/camera_fingerprint/) is not already, reorganize the folders such that ```PRNU/CameraFingerprint``` ;
 - Download the Reference Camera Fingerprints [here](https://drive.google.com/drive/folders/1SmTu0IoGCSEWFMNOYfrMtXtCg2DznggN?usp=sharing);
 - at least 9G GPU.
## Set up Virtual-Env
```
conda env create -f environment.yml
```

# Test

## Test a match (H1) hypothesis case
```
python -u main_OFF_H1.py
```

## Test a match (H1) hypothesis case on CPU
```
python -u main_OFF_H1_cpu.py
```

## Test a mis-match (H0) hypothesis case
```
python -u main_OFF_H0.py
```

## NOTE:
You need to edit in ```main_OFF_H1.py```, ```main_OFF_H1_cpu.py``` and ```main_OFF_H0.py```:
- ```Fingeprint_list``` changing it with the path to your Camera Fingerprints
- ```images_set``` changing it with the path to the test images corresponding to your Camera Fingerprint
- ```outfile_name``` changing it with your output file name

# Results of the Paper

Check ["EXPLOITING PRNU AND LINEAR PATTERNS IN FORENSIC CAMERA ATTRIBUTION
UNDER COMPLEX LENS DISTORTION CORRECTION"](https://ieeexplore.ieee.org/abstract/document/10096605)

<p align="center">
  <img src="https://github.com/AMontiB/GPU-PRNU-SIFT/blob/main/figures/ROC.png">
</p>

![tables](https://github.com/AMontiB/GPU-PRNU-SIFT/blob/main/figures/table.png?raw=true)

# Cite Us
If you use this code please cite: 

@inproceedings{montibeller2023exploiting, \
  title={Exploiting PRNU and Linear Patterns in Forensic Camera Attribution under Complex Lens Distortion Correction}, \
  author={Montibeller, Andrea and P{\'e}rez-Gonz{\'a}lez, Fernando}, \
  booktitle={ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, \
  pages={1--5}, \
  year={2023}, \
  organization={IEEE} \
}

# EXPLOITING PRNU AND LINEAR PATTERNS IN FORENSIC CAMERA ATTRIBUTION UNDER COMPLEX LENS DISTORTION CORRECTION

This is the official code implementation of the "ICIP 2022" paper ["EXPLOITING PRNU AND LINEAR PATTERNS IN FORENSIC CAMERA ATTRIBUTION
UNDER COMPLEX LENS DISTORTION CORRECTION"](https://ieeexplore.ieee.org/abstract/document/10096605)

## Requirements

- Download the python libraries of [Camera-fingerprint](https://dde.binghamton.edu/download/camera_fingerprint/) ;
 - if [Camera-fingerprint](https://dde.binghamton.edu/download/camera_fingerprint/) is not already, reorganize the folders such that ```PRNU/CameraFingerprint``` ;
 - Download the Reference Camera Fingerprints [here](https://drive.google.com/drive/folders/1q6FpTvP5FYsgaQf5kbC3vjuT6s8jbmxs?usp=sharing);
 - at least 9G GPU.
## Set up Virtual-Env
```
conda env create -f environment.yml
```

# Test

## Test a match (H1) hypothesis case
```
nohup python -u main_H1.py --videos PATH_TO_VIDEOS --fingerprint PATH_TO_FINGERPRINTS --output PATH_TO_OUTPUT_FOLDER --gpu_dev /gpu:N >| output_H1.log & 
```

## Test a mis-match (H0) hypothesis case
```
nohup python -u main_H0.py --videos PATH_TO_VIDEOS --fingerprint PATH_TO_FINGERPRINTS --output PATH_TO_OUTPUT_FOLDER --gpu_dev /gpu:N >| output_H0.log & 
```

## Run both
Edit and Run ```bash runner.sh```

## NOTE:
You need to edit:
- ```PATH_TO_VIDEOS``` changing it with the path to your dataset
- ```PATH_TO_FINGERPRINTS``` changing it with the path to your reference camera fingerprints
- ```PATH_TO_OUTPUT_FOLDER``` changing it with the path to your output folder
- ```N``` chaging it with your GPU ID

# Results of the Paper

Check ["GPU-accelerated SIFT-aided source identification of stabilized videos"](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=C0v9f-cAAAAJ&citation_for_view=C0v9f-cAAAAJ:UeHWp8X0CEIC)

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

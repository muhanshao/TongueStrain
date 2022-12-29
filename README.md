# Tongue muscle fiber reconstruction and strain analysis

The tongue muscle fiber orientations can be reconstructed from high angular resolution diffusion imaging (HARDI) data. The strain in the line of action during speech production can be analyzed from the fiber orientations and motion fields.

## Required data
 - HARDI image of the tongue (at least 45 b-vectors).
 - Tagged MRI of the tongue acquired during speech.
 - Cine MRI of the tongue acquired during speech (acquired at the same time with tagged MRI).
 - Tongue mask on the HARDI data and on the motion data (tagged and cine).

## Tongue strain analysis pipeline
 - Fiber orientation reconstruction from HARDI data
   * Compute spherical harmonic (SH) coefficients from the original HARDI data.
   * Apply the trained network to the SH coefficients.
   * Manually correct the fiber orientations if necessary.
 - Motion fields estimation from tagged and cine data
 - Transform the fiber orientation from the HARDI space to the motion space using image registration
 - Compute strain tensor based on the estimated motion fields.
 - Project the computed strain tensor onto the transformed fiber orientations to get the strain in the line of action.

## Folders and files description
 - Folder _dirdetect/_ contains tongue muscle fiber orientation reconstruction code and models.
 - Folder _bin/_ contains run scripts for the strain analysis pipeline.
 - Folder _pipeline_code_example/_ contains example scripts for how to run the strain analysis pipeline.

## Required softwares and python packages:
 - Advanced Normalization Tools (ANTs)
 - Nibabel
 - DIPY
 - PyTorch
 - dwave-qbsolv

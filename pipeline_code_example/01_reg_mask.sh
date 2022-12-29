#!/bin/bash

dataDir=$YOUR_DATA_DIR
outDir=$YOUR_OUT_DIR
regScript='../bin/01_SyN_dtimask2cinemask.sh'

# fixed image: cine tongue mask. moving image: dwi tongue mask.
# ATHING
word=athing
mkdir -p $outDir/$word
outDir=$(realpath $outDir/$word/01_registration_mask)
mkdir -p $outDir

fixedImg=$(ls $dataDir/cine/*$word*tonguemask.nii)
outPrefix=$outDir/mask-in-dwi_to_cine
set -x
$regScript $fixedImg $movingImg $outPrefix
set +x

#!/bin/bash

subj=$SUBJECT_ID
dataDir=$YOUR_DATA_DIR
regScript='../bin/01_SyN_cine.sh'
outDir=$YOUR_OUT_DIR
ANTSPATH=$YOUR_ANTS_PATH


run_reg_cine () {
    outDir=$(realpath $outDir/$word/01_registration_cine/reg_01_${tf})
    mkdir -p $outDir
    fixedImg=$(ls $dataDir/*${word}*${tf}_fsl_rai.nii)
    outPrefix=$outDir/${subj}_${word}_01to${tf}
    set -x
    $regScript $fixedImg $movingImg $outPrefix
    set +x
    TransFile1=$(ls $outDir/*0GenericAffine.mat)
    TransFile2=$(ls $outDir/*1Warp.nii.gz)
    outMask=$outDir/${subj}_${word}_mask_01to${tf}_reg.nii.gz
    $ANTSPATH/antsApplyTransforms -d 3 \
				  -i $inMask \
				  -r $fixedImg \
				  --float \
				  -n NearestNeighbor \
				  -o $outMask \
				  -t $TransFile2 \
				  -t $TransFile1

}

# ATHING
word=athing
movingImg=$(ls $dataDir/*${word}*01_fsl_rai.nii)
inMask=$(ls $dataDir/*${word}*01_fsl_rai_tonguemask.nii)
mkdir -p $outDir/$word/01_registration_cine
######
tf=07
run_reg_cine
######
tf=14
run_reg_cine
######
tf=18
run_reg_cine
######
tf=24
run_reg_cine

#!/bin/bash


subj=$SUBJECT_ID  # Subject ID
#############
outDir=$YOUR_OUT_DIR
scriptDir='../bin/'

mkdir -p $outDir/fiber_recon
fdwi=$HARDI_DATA_PATH
fmask=$TONGUE_MASK_PATH
fbval=$B_VAL_PATH
fbvec=$B_VEC_PATH
fsh=$outDir/${subj}_eddy_SH_order8.nii.gz
foutpre=$outDir/fiber_recon/${subj}_eddy_epoch150

set -x
$scriptDir/02_run_pred_cnn.py --fdwi $fdwi \
			      --fmask $fmask \
			      --fbval $fbval \
			      --fbvec $fbvec \
			      --fsh $fsh \
			      --foutpre $foutpre
set +x

fdir1=$(ls ${foutpre}_dir1_pred.nii*)
fdir2=$(ls ${foutpre}_dir2_pred.nii*)
set -x
$scriptDir/02_run_qubo.py --fdir1 $fdir1 \
			  --fdir2 $fdir2 \
			  --fmask $fmask \
			  --foutpre $foutpre
set +x

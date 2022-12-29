#!/bin/bash


subj=$SUBJECT_ID
scriptDir='../bin/'
motionDir=$YOUR_MOTION_FIELD_DIR
outDir=$YOUR_OUT_DIR

## Projection:
## From "uh" to each other key time frame (Lagranging)

run_proj_strain () {
    fiberDir=$(realpath $outDir/$word/03v3_reorient_fiber/trans_01_${tfref})
    fFiberDir1=$(ls $fiberDir/*dir1_tensor_trans_reorient_todir.nii.gz)
    fFiberDir2=$(ls $fiberDir/*dir2_tensor_trans_reorient_todir.nii.gz)
    fMask=$(ls $fiberDir/*tongue_mask*.nii*)
    foutpre=$outDir/${subj}_${word}_lagra_align

    set -x
    $scriptDir/04v2_strain_proj.py  --tflist $tflist \
				    --niidir $niiDir \
				    --fdir1 $fFiberDir1 \
				    --fdir2 $fFiberDir2 \
				    --fmask $fMask \
				    --foutpre $foutpre
    set +x
}


## ATHING realign TF
word=athing
niiDir=$motionDir/${subj}_${word}_uncrop_field/lagra_align
outDir=$(realpath $outDir/${word}/04_proj_strain_aligntf)
mkdir -p $outDir
tfref=07
tfstart=1
tfend=20
tf1=$(printf "%02d" $tfstart)
for ((i=$((tfstart+1));i<=$tfend;i++)); do
    tf2=$(printf "%02d" $i)
    tflist=${tf1}${tf2}
    # echo $tflist
    run_proj_strain
done

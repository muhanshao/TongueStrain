#!/bin/bash


subj=$SUBJECT_ID
scriptDir='../bin/'
dataDir=$YOUR_DATA_DIR
outDir=$YOUR_OUT_DIR
fiberDir=$(realpath $outDir/fiber_recon)
fdir1=$(ls $fiberDir/*dir1_pred_match_patch3.nii.gz)
fdir2=$(ls $fiberDir/*dir2_pred_match_patch3.nii.gz)
fmask=$(ls $dataDir/dwi/*mask*correct.nii*)
foutpre=$fiberDir/${subj}_eddy_cnn

#### Reorient the fiber direction from DTI to cine (5 key TFs)
#### Transformation from cine to cine:
#### Registration in $word/01_registration_cine


## Fiber to tensor
set -x
$scriptDir/03_dir2tensor.py --fdir1 $fdir1 \
			    --fdir2 $fdir2 \
			    --fmask $fmask \
			    --foutpre $foutpre
set +x


reorient_tf01 () {
    outDirtf=$outDir/trans_01_${tf2}
    mkdir $outDirtf
    refImg=$(ls $regDir/*reg.nii*)
    foutpre=$outDirtf/${subj}_eddy_cnn_${word}_01_${tf2}

    set -x
    $scriptDir/03v3_run_reorient.sh $fiberDir $refImg \
				  $regDir $outDirtf
    set +x

    ftensor1=$(ls $outDirtf/*dir1_tensor_trans.nii.gz)
    ftensor2=$(ls $outDirtf/*dir2_tensor_trans.nii.gz)
    fmask1=$(ls $regDir/*reg.nii.gz)
    fmask2=$(ls $outDirtf/*dir2_mask-to-cine.nii.gz)
    set -x
    $scriptDir/03_tensor2dir.py --ftensor1 $ftensor1 \
				--ftensor2 $ftensor2 \
				--fmask1 $fmask1 \
				--fmask2 $fmask2 \
				--frefimg $refImg \
				--foutpre $foutpre
    set +x
    ln -s $fmask1 $outDirtf/${subj}_eddy_tongue_mask-to-cine_${word}_${tf2}.nii.gz
    mv $fmask2 $outDirtf/${subj}_eddy_dir2_mask-to-cine_${word}_${tf2}.nii.gz
}


reorient_anytf () {
    regDir2=${regDir2Root}/reg_01_${tf2}
    refImg=$(ls $regDir/*reg.nii.gz)

    outDirtf=$outDir/trans_01_${tf2}
    mkdir $outDirtf
    foutpre=$outDirtf/${subj}_eddy_cnn_${word}_01_${tf2}
    # echo $regDir
    set -x
    ${scriptDir}/03v3_run_reorient.sh $fiberDir $refImg \
    		$regDir $outDirtf \
    		$regDir2 $fmask
    set +x

    ftensor1=$(ls $outDirtf/*dir1_tensor_trans.nii.gz)
    ftensor2=$(ls $outDirtf/*dir2_tensor_trans.nii.gz)
    fmask1=$(ls $outDirtf/*tongue_mask-to-cine.nii*)
    fmask2=$(ls $outDirtf/*dir2_mask-to-cine.nii.gz)
    set -x
    ${scriptDir}/03_tensor2dir.py --ftensor1 $ftensor1 \
    		--ftensor2 $ftensor2 \
    		--fmask1 $fmask1 \
    		--fmask2 $fmask2 \
    		--frefimg $refImg \
    		--foutpre $foutpre
    set +x
    mv $fmask1 $outDirtf/${subj}_eddy_tongue_mask-to-cine_${word}_${tf2}.nii.gz
    mv $fmask2 $outDirtf/${subj}_eddy_dir2_mask-to-cine_${word}_${tf2}.nii.gz
}

## ATHING
word=athing
regDir=$(realpath $outDir/$word/01_registration_mask)
regDir2Root=$(realpath $outDir/$word/01_registration_cine)
outDir=$(realpath $outDir/$word/03v3_reorient_fiber)
mkdir -p $outDir
#######
tf2=01
reorient_tf01
#######
tf2=07
reorient_anytf
#######
tf2=14
reorient_anytf
#######
tf2=18
reorient_anytf
#######
tf2=24
reorient_anytf

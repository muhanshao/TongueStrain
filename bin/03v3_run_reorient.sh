#!/bin/bash


## Combine *Warp.nii.gz (deformable) and *.mat (rigid matrix)
fiberDir=$(realpath $1)
refImg=$(realpath $2)
regDir=$(realpath $3)
outDir=$(realpath $4)
regDir2=$(realpath $5)
dwiMask=$(realpath $6)

Img=$(ls $fiberDir/*_dir1_tensor.nii.gz)
Imgbase=$(basename $Img)
ImgTrans=$outDir/$(echo $Imgbase | sed "s/\.nii\.gz$/_trans.nii.gz/")
ImgReorient=$outDir/$(echo $Imgbase | sed "s/\.nii\.gz$/_trans.nii.gz/")

Img2=$(ls $fiberDir/*_dir2_tensor.nii.gz)
Img2base=$(basename $Img2)
ImgTrans2=$outDir/$(echo $Img2base | sed "s/\.nii\.gz$/_trans.nii.gz/")
ImgReorient2=$outDir/$(echo $Img2base | sed "s/\.nii\.gz$/_trans.nii.gz/")

dtCombTrans=$outDir/dtCombinedWarp.nii.gz

TransFile1=$(ls $regDir/*0GenericAffine.mat)
TransFile2=$(ls $regDir/*1Warp.nii.gz)
TransFile3=$(ls $regDir2/*0GenericAffine.mat)
TransFile4=$(ls $regDir2/*1Warp.nii.gz)

$ANTSPATH/antsApplyTransforms -d 3 \
			      -r $refImg \
			      --float \
			      -o [$dtCombTrans, 1] \
			      -t $TransFile4 \
			      -t $TransFile3 \
			      -t $TransFile2 \
			      -t $TransFile1

## Apply the transform to the mask of the dir2
Mask2=$(ls $fiberDir/*dir2_mask.nii.gz)
Mask2base=$(basename $Mask2)
Mask2Trans=$outDir/$(echo $Mask2base | sed "s/mask/mask-to-cine/")
$ANTSPATH/antsApplyTransforms -i $Mask2 \
			      --float \
			      -d 3 \
			      -r $refImg \
			      -n NearestNeighbor \
			      -o $Mask2Trans \
			      -t $dtCombTrans

## Apply the transform to the dwi mask
dwiMaskTrans=$outDir/$(echo $(basename $Mask2Trans) | sed "s/dir2/tongue/")
$ANTSPATH/antsApplyTransforms -i $dwiMask \
			      --float \
			      -d 3 \
			      -r $refImg \
			      -n NearestNeighbor \
			      -o $dwiMaskTrans \
			      -t $dtCombTrans

################### Transform dir1 #########################
## Apply the transform to the DT image
$ANTSPATH/antsApplyTransforms -i $Img \
			      --float \
			      -d 3 \
			      -r $refImg \
			      --input-image-type 2 \
			      -n NearestNeighbor \
			      -o $ImgTrans \
			      -t $dtCombTrans

## Reorient tensor
$ANTSPATH/ReorientTensorImage 3 $ImgTrans $ImgReorient $dtCombTrans

################### Transform dir2 #########################
## Apply the transform to the DT image
$ANTSPATH/antsApplyTransforms -i $Img2 \
			      --float \
			      -d 3 \
			      -r $refImg \
			      --input-image-type 2 \
			      -n NearestNeighbor \
			      -o $ImgTrans2 \
			      -t $dtCombTrans

### Reorient tensor
$ANTSPATH/ReorientTensorImage 3 $ImgTrans2 $ImgReorient2 $dtCombTrans

#!/bin/bash

if [ $# -lt "1" ]; then 
	echo "Usage: 
	run_SyN.sh FixedImage MovingImage OutputPrefix

FixedImage	is the full path of the fixed image volume  
		* Note: Can not be relative path like ./
MovingImage     is the full path of the fixed image volume  
		* Note: Can not be relative path like ./
OutputPrefix    is the directory and prefix of the output (full path)


"
	exit
fi

export FIXED=$1
export MOVING=$2
export OUTPUTPRE=$3


$ANTSPATH/antsRegistration --verbose 1 --dimensionality 3 --float 1 \
	--output [$OUTPUTPRE,${OUTPUTPRE}_reg.nii.gz,${OUTPUTPRE}_inverse.nii.gz] \
	--interpolation NearestNeighbor --use-histogram-matching 1 \
	--winsorize-image-intensities [0.005,0.995] \
	--initial-moving-transform [$FIXED,$MOVING,1] \
	--transform Rigid[0.1] \
	--metric MI[$FIXED,$MOVING,1,32,Regular,0.25] \
	--convergence [1000x500x250x100,1e-6,10] \
	--shrink-factors 8x4x2x1 \
	--smoothing-sigmas 3x2x1x0vox \
	--transform SyN[0.1,3,0] \
	--metric CC[$FIXED,$MOVING,1,4] \
	--convergence [20x20x20x20,1e-6,10] \
	--shrink-factors 6x4x2x1 \
	--smoothing-sigmas 3x2x1x0vox \
	> ${OUTPUTPRE}.txt


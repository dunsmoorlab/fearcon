#!/bin/bash

#For each subject, I need to replace the reg/example_func2standard with the identity matrix and then the standard.nii.gz with an MNI corrected orig.nii


for subI in 01 02 03 04 05 06 07 08 09 10 12 13 14 15 16 17 18 19 20 21
	do
	for runI in 1 2 3
	do
		for conI in all early late
		do
			destDir=/work/05426/ach3377/lonestar/fc/Sub0${subI}/model/run00${runI}/${conI}.feat
			rm ${destDir}/reg/*.mat
			rm ${destDir}/reg/standard.nii.gz
			cp /work/IRC/ls5/opt/apps/fsl-5.0.10/data/standard/MNI152_T1_1mm_brain.nii.gz ${destDir}/reg/standard.nii.gz
			cp /work/IRC/ls5/opt/apps/fsl-5.0.10/etc/flirtsch/ident.mat ${destDir}/reg/example_func2standard.mat
			cp /work/IRC/ls5/opt/apps/fsl-5.0.10/etc/flirtsch/ident.mat ${destDir}/reg/standard2example_func.mat
		done

	done
done

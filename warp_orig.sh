#!/bin/bash


# for subI in 01 02 03 04 05 06 07 08 09 10 12 13 14 15 16 17 18 19 20 21
# do
# 	subDir=/work/05426/ach3377/lonestar/fc/Sub0${subI}

# 	for runI in 1 2 3 4 5 6 7 8 9
# 	do
# 		echo "applywarp --in=${subDir}/bold/orig/pp_run00${runI}.nii.gz --ref=/work/IRC/ls5/opt/apps/fsl-5.0.10/data/standard/MNI152_T1_1mm_brain.nii.gz --out=${subDir}/bold/std/std_pp_run00${runI}.nii.gz --premat=${subDir}/reg/func2struct.mat --warp=${subDir}/reg/struct2std_warp.nii.gz" >> pp_warp_Sub0${subI}.txt		
# 	done
# 	chmod u+x pp_warp_Sub0${subI}.txt
# done


scripts=/home1/05426/ach3377/CodeBase/level_1_scripts/

for subI in 01 02 03 04 05 06 07 08 09 10 12 13 14 15 16 17 18 19 20 21
do
	for runI in 1 2 3
		for conI in all early late
		do
			echo "feat ${scripts}/Sub0${subI}_run00${runI}_${conI}.fsf" >> level1_Sub0${subI}.txt		
		done
	chmod u+x level1_Sub0${subI}.txt
done
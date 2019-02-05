#!/bin/bash


mkdir -p "/work/05426/ach3377/lonestar/FearCon/reg_convert"


export DEST_DIR="/work/05426/ach3377/lonestar/FearCon/reg_convert"

declare -a SUBS=("01" "02" "03" "04" "05" "06" "07" "08" "09" "10" "12" "13" "14" "15" "16" "17" "18" "19" "20")
declare -a RUNS=("1" "2" "3" "4" "5" "6" "7" "8" "9")
for s in "${SUBS[@]}"
do
	export SUBJ=Sub0"$s"
	export OUTPUT_DIR="$DEST_DIR"/"$SUBJ"
	mkdir -p "$OUTPUT_DIR"
	export TEMP_DIR="/work/05426/ach3377/lonestar/FearCon"/"$SUBJ"/"tacc_temp"/

	echo "$SUBJ"
	echo "$TEMP_DIR"
	echo "$OUTPUT_DIR"

	for i in "${RUNS[@]}"
	do
		echo "run00""$i"
		c3d_affine_tool -itk "$TEMP_DIR"run00"$i"-refvol_0GenericAffine.txt -ref "$TEMP_DIR"be_refvol.nii.gz -src "$TEMP_DIR"be_avg_mc_run00"$i".nii.gz -ras2fsl -o "$OUTPUT_DIR"/run00"$i"-refvol.mat
	done
done	
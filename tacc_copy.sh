#!/bin/bash


#if there is more than 1 flag, add it after the s: and follow the same format for defining flags in `case`
while getopts 's:' flag; do
	case "${flag}" in
		s)
			export SUBJ="${OPTARG}"
	esac
done
echo $SUBJ

export TEMP_DIR="$SUBJECTS_DIR"/"$SUBJ"/tacc_temp/

export DEST_DIR="/work/05426/ach3377/lonestar/FearCon"/"$SUBJ"

export REG_DIR="/work/05426/ach3377/lonestar/FearCon/reg_convert"

export LOCAL_DIR="/Users/ach3377/Db_lpl/STUDY/FearCon/reg_convert"


# scp -r "$TEMP_DIR" ach3377@ls5.tacc.utexas.edu:"$DEST_DIR"

#this put them in CodeBase for some Reason IDK
scp -r ach3377@ls5.tacc.utexas.edu:"$REG_DIR" "$LOCAL_DIR"


#scp -r ach3377@ls5.tacc.utexas.edu:"/work/05426/ach3377/lonestar/group_glm/run002/all/CS+.gfeat/" "/Users/ach3377/Desktop/CS+.gfeat"

#scp -r ach3377@ls5.tacc.utexas.edu:"/work/05426/ach3377/lonestar/fc/Sub020/model/run002/all.feat/" "/Users/ach3377/Desktop/"

#scp -r ach3377@ls5.tacc.utexas.edu:"/work/05426/ach3377/lonestar/group_glm/run004/all.gfeat/" "/Users/ach3377/Desktop/"

#scp -r ach3377@ls5.tacc.utexas.edu:"/work/05426/ach3377/lonestar/fc/Sub001/model/run004/full_tt.feat/" "/Users/ach3377/Desktop/"
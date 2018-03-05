import pandas as pd
import numpy as np
import itertools as it
import os
import sys
from glob import glob


def get_data_dir():
	#three cases: bash, windows, school mac
	if sys.platform == 'linux':
		return '/mnt/c/Users/ACH/Google Drive/FC_FMRI_DATA/'
	elif sys.platform == 'win32':
		return 'C:\\Users\\ACH\\Dropbox (LewPeaLab)\\STUDY\\FearCon\\'
	else:
		return '/Users/ach3377/Db_lpl/STUDY/FearCon/'

data_dir = get_data_dir()

def get_sub_args():
	return [int(subs[3:]) for subs in os.listdir(data_dir) if 'Sub' in subs and 'fs' not in subs]

sub_args = get_sub_args()

working_subs = [1,2,3,4,5,6,7,8,9,10,12,13,14,15,16]

def get_subj_dir(subj):

	return os.path.join(data_dir, 'Sub{0:0=3d}'.format(subj))

def get_bold_dir(subj_dir):

	return os.path.join(subj_dir,'bold' + os.sep)


#only returns run_dir if phase is provided
#self.subj_dir, self.bold_dir, self.run_dir = init_dirs(subj,phase=None) 
def init_dirs(subj,phase=None):
	subj_dir = get_subj_dir(subj)
	bold_dir = get_bold_dir(subj_dir)
	if phase is not None:
		run_dir = os.path.join(bold_dir, nifti_paths[phase][:11])
		return subj_dir, bold_dir, run_dir
	else:
		return subj_dir, bold_dir

def fsub(subj):

	return 'Sub{0:0=3d}'.format(subj)

dataz = 'arr_0'

cv_iter = [2,5,10,50,100,150,200,300,500,1000,1500,2000]

dc2nix_run_key = {
	'BASELINE': 'run001',
	'FEAR_CONDITIONING': 'run002',
	'EXTINCTION': 'run003',
	'EXTINCTION_RECALL': 'run004',
	'MEMORY_RUN_1': 'run005',
	'MEMORY_RUN_2': 'run006',
	'MEMORY_RUN_3': 'run007',
	'LOCALIZER_1': 'run008',
	'LOCALIZER_2': 'run009',
	'MPRAGE': 'struct',
}

py_run_key = {
	'baseline': 'run001',
	'fear_conditioning': 'run002',
	'extinction': 'run003',
	'extinction_recall': 'run004',
	'memory_run_1': 'run005',
	'memory_run_2': 'run006',
	'memory_run_3': 'run007',
	'localizer_1': 'run008',
	'localizer_2': 'run009'
}

hdr_shift = {'start': 3}
hdr_shift['end'] = 6 - hdr_shift['start']

#cond2 = {
#	0: 'rest',
#	1: 'CS-',
#	2: 'CS+',
#	3: 'scene',
#}

#these dictionaries are for FC_timing
#this dictionary lets us be more precise in our code, as experimentally some runs have the same structure
#IMPORTANT: need to manually set which phase is currently the 'tag phase' for day 1
day2phase = {
	'baseline': 'day1',
	'fear_conditioning': 'day1',
	'extinction': 'day1_tag',
	'extinction_recall': 'day2_er' ,
	'memory_run_1': 'day2_mem',
	'memory_run_2': 'day2_mem',
	'memory_run_3': 'day2_mem',
	'localizer_1': 'day2_loc',
	'localizer_2': 'day2_loc',
}

#this is a dictionary in format of phase:[initialITI, finalITI, number of ITIs(not counting first and last)]
#all time in miliseconds
time2add = {
	'day1': [4000,2000,48],
	'day1_tag': [4000,2000,48],
	'day2_er': [8000,2000,24],
	'day2_mem': [4000,2000,80],
	#localizer is a little different [initialITI, stimdur, ITI, IMBI, IBI]
	'day2_loc': [4000, 1500, 500, 6000, 12000]
}

#the localizer stim categories
LocStims = {
	'anim': 'animal',
	'tool': 'tool',
	'indo': 'indoor',
	'loca': 'outdoor',
	'scra': 'scrambled',
}


phase2location = {
	'baseline': 'day1/run001/zd_mc_run001.npy',
	'fear_conditioning': 'day1/run002/zd_mc_run002.npy',
	'extinction': 'day1/run003/zd_mc_run003.npy',
	'extinction_recall': 'day2/run004/zd_mc_run004.npy' ,
	'memory_run_1': 'day2/run005/zd_mc_run005.npy',
	'memory_run_2': 'day2/run006/zd_mc_run006.npy',
	'memory_run_3': 'day2/run007/zd_mc_run007.npy',
	'localizer_1': 'day2/run008/zd_mc_run008.npy',
	'localizer_2': 'day2/run009/zd_mc_run009.npy',
}


mvpa_prepped = {
	'baseline': 'day1/run001/prepped_run001.npz',
	'fear_conditioning': 'day1/run002/prepped_run002.npz',
	'extinction': 'day1/run003/prepped_run003.npz',
	'extinction_recall': 'day2/run004/prepped_run004.npz' ,
	'memory_run_1': 'day2/run005/prepped_run005.npz',
	'memory_run_2': 'day2/run006/prepped_run006.npz',
	'memory_run_3': 'day2/run007/prepped_run007.npz',
	'localizer_1': 'day2/run008/prepped_run008.npz',
	'localizer_2': 'day2/run009/prepped_run009.npz',
}

PPA_prepped = {
	'baseline': 'day1/run001/PPA_prepped_run001.npz',
	'fear_conditioning': 'day1/run002/PPA_prepped_run002.npz',
	'extinction': 'day1/run003/PPA_prepped_run003.npz',
	'extinction_recall': 'day2/run004/PPA_prepped_run004.npz' ,
	'memory_run_1': 'day2/run005/PPA_prepped_run005.npz',
	'memory_run_2': 'day2/run006/PPA_prepped_run006.npz',
	'memory_run_3': 'day2/run007/PPA_prepped_run007.npz',
	'localizer_1': 'day2/run008/PPA_prepped_run008.npz',
	'localizer_2': 'day2/run009/PPA_prepped_run009.npz',
}

PPA_fs_prepped = {
	'baseline': 'day1/run001/PPA_fs_prepped_run001.npz',
	'fear_conditioning': 'day1/run002/PPA_fs_prepped_run002.npz',
	'extinction': 'day1/run003/PPA_fs_prepped_run003.npz',
	'extinction_recall': 'day2/run004/PPA_fs_prepped_run004.npz' ,
	'memory_run_1': 'day2/run005/PPA_fs_prepped_run005.npz',
	'memory_run_2': 'day2/run006/PPA_fs_prepped_run006.npz',
	'memory_run_3': 'day2/run007/PPA_fs_prepped_run007.npz',
	'localizer_1': 'day2/run008/PPA_fs_prepped_run008.npz',
	'localizer_2': 'day2/run009/PPA_fs_prepped_run009.npz',
}


nifti_paths = {
	'baseline': 'day1/run001/mc_run001.nii.gz',
	'fear_conditioning': 'day1/run002/mc_run002.nii.gz',
	'extinction': 'day1/run003/mc_run003.nii.gz',
	'extinction_recall': 'day2/run004/mc_run004.nii.gz' ,
	'memory_run_1': 'day2/run005/mc_run005.nii.gz',
	'memory_run_2': 'day2/run006/mc_run006.nii.gz',
	'memory_run_3': 'day2/run007/mc_run007.nii.gz',
	'localizer_1': 'day2/run008/mc_run008.nii.gz',
	'localizer_2': 'day2/run009/mc_run009.nii.gz',
}

phase2rundir = {
	'baseline': 'day1/run001/',
	'fear_conditioning': 'day1/run002/',
	'extinction': 'day1/run003/',
	'extinction_recall': 'day2/run004/' ,
	'memory_run_1': 'day2/run005/',
	'memory_run_2': 'day2/run006/',
	'memory_run_3': 'day2/run007/',
	'localizer_1': 'day2/run008/',
	'localizer_2': 'day2/run009/',
}

n_trials = {
	'baseline': range(0,48),
	'fear_conditioning': range(0,48),
	'extinction': range(0,48),
	'extinction_recall': range(0,24) ,
	'memory_run_1': range(0,80),
	'memory_run_2': range(0,80),
	'memory_run_3': range(0,80),
	'localizer_1': range(0,160),
	'localizer_2': range(0,160),
}
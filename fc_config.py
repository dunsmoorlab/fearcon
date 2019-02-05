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
		return os.path.expanduser('~') + os.sep + 'Db_lpl/STUDY/FearCon/'

data_dir = get_data_dir()

def get_sub_args(P=False):
	 
	subs = [int(subs[3:]) for subs in os.listdir(data_dir) if 'Sub' in subs and 'fs' not in subs]
	
	#the ptsd subs all have 3 digits
	if P:
		sub_args = [subs for subs in subs if len(str(subs)) > 2]
		sub_args.sort()
		return sub_args
	
	sub_args = [subs for subs in subs if len(str(subs)) < 3]
	sub_args.sort()
	return sub_args

sub_args = get_sub_args()
# sub_args = [1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,17,18,19,20,21]
p_sub_args = get_sub_args(P=True)
# p_sub_args = [101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,120,121]
all_sub_args = sub_args + p_sub_args

no107 = [101,102,103,104,105,106,108,109,110,111,112,113,114,115,116,117,118,120,121]

all_no107 = sub_args + p_sub_args
all_no107 = [sub for sub in all_no107 if sub is not 107]

working_subs = [101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,120,121]
# working_subs = [101,102,103,104,105,106,108,109,110,111,112,113,114,115,]

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

def pretty_graph(ax, xlab=None, ylab=None, main=None, legend=None, grid=False):

	#these arguments have to be set because you should always label
	ax.set_title(main, size='xx-large')
	ax.set_xlabel(xlab, size = 'x-large')
	ax.set_ylabel(ylab, size = 'x-large')

	#if you give it a legend title then it plots a legend
	if legend:
	    ax.legend(loc='upper right')

	#turn on the grid if you want to
	if grid:
	    ax.grid(which='major')


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

dc2nix_run_key_1 = {
	'Baseline': 'run001',
	'Fear_Conditioning': 'run002',
	'Extinction': 'run003',
	'Extinction_Recall': 'run004',
	'Memory_Run_1': 'run005',
	'Memory_Run_2': 'run006',
	'Memory_Run_3': 'run007',
	'Localizer_1': 'run008',
	'Localizer_2': 'run009',
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

hdr_shift = {'start': 2}
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

day_dict = {
	'day1':{
		'level1':[
				['baseline', 'fear_conditioning','mid1_1'],
				['fear_conditioning', 'extinction','mid1_2']],
		'level2':[['mid1_1', 'mid1_2','day1']]
	},
	

	'day2':{
		'level1':[
				['extinction_recall', 'memory_run_1','mid1_1'],
				['memory_run_1', 'memory_run_2','mid1_2'],
				['memory_run_2', 'memory_run_3','mid1_3'],
				['memory_run_3','localizer_1','mid1_4'],
				['localizer_1', 'localizer_2','mid1_5']],
		'level2':[
				['mid1_1', 'mid1_2', 'mid2_1'],
				['mid1_2', 'mid1_3', 'mid2_2'],
				['mid1_3','mid1_4', 'mid2_3'],
				['mid1_4', 'mid1_5', 'mid2_4']],
		'level3':[
				['mid2_1','mid2_2', 'mid3_1'],
				['mid2_2','mid2_3', 'mid3_2'],
				['mid2_3','mid2_4', 'mid3_3']],
		'level4':[
				['mid3_1','mid3_2', 'mid4_1'],
				['mid3_2','mid3_3', 'mid4_2']],
		'level5':[
				['mid4_1', 'mid4_2', 'day2']]
	}
}

fsl_betas = {
	'baseline': 'day1/run001/fsl_betas/run001_beta.nii.gz',
	'fear_conditioning': 'day1/run002/fsl_betas/run002_beta.nii.gz',
	'extinction': 'day1/run003/fsl_betas/run003_beta.nii.gz',
	'extinction_recall': 'day2/run004/fsl_betas/run004_beta.nii.gz' ,
	#'memory_run_1': 'day2/run005/fsl_betas/run005_beta.nii.gz',
	#'memory_run_2': 'day2/run006/fsl_betas/run006_beta.nii.gz',
	#'memory_run_3': 'day2/run007/fsl_betas/run007_beta.nii.gz',
	#'localizer_1': 'day2/run008/fsl_betas/run008_beta.nii.gz',
	#'localizer_2': 'day2/run009/fsl_betas/run009_beta.nii.gz',
}

fsl_betas_std = {
	'baseline': 'day1/run001/fsl_betas/run001_beta_std.nii.gz',
	'fear_conditioning': 'day1/run002/fsl_betas/run002_beta_std.nii.gz',
	'extinction': 'day1/run003/fsl_betas/run003_beta_std.nii.gz',
	'extinction_recall': 'day2/run004/fsl_betas/run004_beta_std.nii.gz' ,
	#'memory_run_1': 'day2/run005/fsl_betas/run005_beta_std.nii.gz',
	#'memory_run_2': 'day2/run006/fsl_betas/run006_beta_std.nii.gz',
	#'memory_run_3': 'day2/run007/fsl_betas/run007_beta_std.nii.gz',
	#'localizer_1': 'day2/run008/fsl_betas/run008_beta_std.nii.gz',
	#'localizer_2': 'day2/run009/fsl_betas/run009_beta_std.nii.gz',
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

mvpa_masked_prepped = {
	'baseline': 'day1/run001/m_pp_run001.npz',
	'fear_conditioning': 'day1/run002/m_pp_run002.npz',
	'extinction': 'day1/run003/m_pp_run003.npz',
	'extinction_recall': 'day2/run004/m_pp_run004.npz' ,
	'memory_run_1': 'day2/run005/m_pp_run005.npz',
	'memory_run_2': 'day2/run006/m_pp_run006.npz',
	'memory_run_3': 'day2/run007/m_pp_run007.npz',
	'localizer_1': 'day2/run008/m_pp_run008.npz',
	'localizer_2': 'day2/run009/m_pp_run009.npz',
}

combined_prepped = {
	'baseline': 'day1/run001/vtc_ppa_run001.npz',
	'fear_conditioning': 'day1/run002/vtc_ppa_run002.npz',
	'extinction': 'day1/run003/vtc_ppa_run003.npz',
	'extinction_recall': 'day2/run004/vtc_ppa_run004.npz' ,
	'memory_run_1': 'day2/run005/vtc_ppa_run005.npz',
	'memory_run_2': 'day2/run006/vtc_ppa_run006.npz',
	'memory_run_3': 'day2/run007/vtc_ppa_run007.npz',
	'localizer_1': 'day2/run008/vtc_ppa_run008.npz',
	'localizer_2': 'day2/run009/vtc_ppa_run009.npz',
}

beta_combined_prepped = {
	'baseline': 'day1/run001/b_vtc_ppa_run001.npz',
	'fear_conditioning': 'day1/run002/b_vtc_ppa_run002.npz',
	'extinction': 'day1/run003/b_vtc_ppa_run003.npz',
	'extinction_recall': 'day2/run004/b_vtc_ppa_run004.npz' ,
	'memory_run_1': 'day2/run005/b_vtc_ppa_run005.npz',
	'memory_run_2': 'day2/run006/b_vtc_ppa_run006.npz',
	'memory_run_3': 'day2/run007/b_vtc_ppa_run007.npz',
	'localizer_1': 'day2/run008/b_vtc_ppa_run008.npz',
	'localizer_2': 'day2/run009/b_vtc_ppa_run009.npz',
}

ppa_prepped = {
	'baseline': 'day1/run001/ppa_run001.npz',
	'fear_conditioning': 'day1/run002/ppa_run002.npz',
	'extinction': 'day1/run003/ppa_run003.npz',
	'extinction_recall': 'day2/run004/ppa_run004.npz' ,
	'memory_run_1': 'day2/run005/ppa_run005.npz',
	'memory_run_2': 'day2/run006/ppa_run006.npz',
	'memory_run_3': 'day2/run007/ppa_run007.npz',
	'localizer_1': 'day2/run008/ppa_run008.npz',
	'localizer_2': 'day2/run009/ppa_run009.npz',
}

beta_masked_prepped = {
	'baseline': 'day1/run001/fsl_betas/b_m_pp_run001.npz',
	'fear_conditioning': 'day1/run002/fsl_betas/b_m_pp_run002.npz',
	'extinction': 'day1/run003/fsl_betas/b_m_pp_run003.npz',
	'extinction_recall': 'day2/run004/fsl_betas/b_m_pp_run004.npz' ,
	'memory_run_1': 'day2/run005/fsl_betas/b_m_pp_run005.npz',
	'memory_run_2': 'day2/run006/fsl_betas/b_m_pp_run006.npz',
	'memory_run_3': 'day2/run007/fsl_betas/b_m_pp_run007.npz',
	'localizer_1': 'day2/run008/fsl_betas/b_m_pp_run008.npz',
	'localizer_2': 'day2/run009/fsl_betas/b_m_pp_run009.npz',
}

amygdala_beta = {
	'baseline': 'day1/run001/amygdala_run001.npz',
	'fear_conditioning': 'day1/run002/amygdala_run002.npz',
	'extinction': 'day1/run003/amygdala_run003.npz',
	'extinction_recall': 'day2/run004/amygdala_run004.npz' ,
	'memory_run_1': 'day2/run005/amygdala_run005.npz',
	'memory_run_2': 'day2/run006/amygdala_run006.npz',
	'memory_run_3': 'day2/run007/amygdala_run007.npz',
	'localizer_1': 'day2/run008/amygdala_run008.npz',
	'localizer_2': 'day2/run009/amygdala_run009.npz',
}

dACC_beta = {
	'baseline': 'day1/run001/dACC_run001.npz',
	'fear_conditioning': 'day1/run002/dACC_run002.npz',
	'extinction': 'day1/run003/dACC_run003.npz',
	'extinction_recall': 'day2/run004/dACC_run004.npz' ,
	'memory_run_1': 'day2/run005/dACC_run005.npz',
	'memory_run_2': 'day2/run006/dACC_run006.npz',
	'memory_run_3': 'day2/run007/dACC_run007.npz',
	'localizer_1': 'day2/run008/dACC_run008.npz',
	'localizer_2': 'day2/run009/dACC_run009.npz',
}

hippocampus_beta = {
	'baseline': 'day1/run001/hippocampus_run001.npz',
	'fear_conditioning': 'day1/run002/hippocampus_run002.npz',
	'extinction': 'day1/run003/hippocampus_run003.npz',
	'extinction_recall': 'day2/run004/hippocampus_run004.npz' ,
	'memory_run_1': 'day2/run005/hippocampus_run005.npz',
	'memory_run_2': 'day2/run006/hippocampus_run006.npz',
	'memory_run_3': 'day2/run007/hippocampus_run007.npz',
	'localizer_1': 'day2/run008/hippocampus_run008.npz',
	'localizer_2': 'day2/run009/hippocampus_run009.npz',
}

vmPFC_beta = {
	'baseline': 'day1/run001/vmPFC_run001.npz',
	'fear_conditioning': 'day1/run002/vmPFC_run002.npz',
	'extinction': 'day1/run003/vmPFC_run003.npz',
	'extinction_recall': 'day2/run004/vmPFC_run004.npz' ,
	'memory_run_1': 'day2/run005/vmPFC_run005.npz',
	'memory_run_2': 'day2/run006/vmPFC_run006.npz',
	'memory_run_3': 'day2/run007/vmPFC_run007.npz',
	'localizer_1': 'day2/run008/vmPFC_run008.npz',
	'localizer_2': 'day2/run009/vmPFC_run009.npz',
}

beta_ppa_prepped = {
	'baseline': 'day1/run001/fsl_betas/b_ppa_run001.npz',
	'fear_conditioning': 'day1/run002/fsl_betas/b_ppa_run002.npz',
	'extinction': 'day1/run003/fsl_betas/b_ppa_run003.npz',
	'extinction_recall': 'day2/run004/fsl_betas/b_ppa_run004.npz' ,
	'memory_run_1': 'day2/run005/fsl_betas/b_ppa_run005.npz',
	'memory_run_2': 'day2/run006/fsl_betas/b_ppa_run006.npz',
	'memory_run_3': 'day2/run007/fsl_betas/b_ppa_run007.npz',
	'localizer_1': 'day2/run008/fsl_betas/b_ppa_run008.npz',
	'localizer_2': 'day2/run009/fsl_betas/b_ppa_run009.npz',
}


nifti_paths = {
	'baseline': 'day1/run001/pp_run001.nii.gz',
	'fear_conditioning': 'day1/run002/pp_run002.nii.gz',
	'extinction': 'day1/run003/pp_run003.nii.gz',
	'extinction_recall': 'day2/run004/pp_run004.nii.gz' ,
	'memory_run_1': 'day2/run005/pp_run005.nii.gz',
	'memory_run_2': 'day2/run006/pp_run006.nii.gz',
	'memory_run_3': 'day2/run007/pp_run007.nii.gz',
	'localizer_1': 'day2/run008/pp_run008.nii.gz',
	'localizer_2': 'day2/run009/pp_run009.nii.gz',
}

std_paths = {
	'baseline': 'day1/run001/std_pp_run001.nii.gz',
	'fear_conditioning': 'day1/run002/std_pp_run002.nii.gz',
	'extinction': 'day1/run003/std_pp_run003.nii.gz',
	'extinction_recall': 'day2/run004/std_pp_run004.nii.gz' ,
	'memory_run_1': 'day2/run005/std_pp_run005.nii.gz',
	'memory_run_2': 'day2/run006/std_pp_run006.nii.gz',
	'memory_run_3': 'day2/run007/std_pp_run007.nii.gz',
	'localizer_1': 'day2/run008/std_pp_run008.nii.gz',
	'localizer_2': 'day2/run009/std_pp_run009.nii.gz',
}



avg_mc_paths = {
	'baseline': 'day1/run001/avg_mc_run001.nii.gz',
	'fear_conditioning': 'day1/run002/avg_mc_run002.nii.gz',
	'extinction': 'day1/run003/avg_mc_run003.nii.gz',
	'extinction_recall': 'day2/run004/avg_mc_run004.nii.gz' ,
	'memory_run_1': 'day2/run005/avg_mc_run005.nii.gz',
	'memory_run_2': 'day2/run006/avg_mc_run006.nii.gz',
	'memory_run_3': 'day2/run007/avg_mc_run007.nii.gz',
	'localizer_1': 'day2/run008/avg_mc_run008.nii.gz',
	'localizer_2': 'day2/run009/avg_mc_run009.nii.gz',
}

be_avg_mc_paths = {
	'baseline': 'day1/run001/be_avg_mc_run001.nii.gz',
	'fear_conditioning': 'day1/run002/be_avg_mc_run002.nii.gz',
	'extinction': 'day1/run003/be_avg_mc_run003.nii.gz',
	'extinction_recall': 'day2/run004/be_avg_mc_run004.nii.gz' ,
	'memory_run_1': 'day2/run005/be_avg_mc_run005.nii.gz',
	'memory_run_2': 'day2/run006/be_avg_mc_run006.nii.gz',
	'memory_run_3': 'day2/run007/be_avg_mc_run007.nii.gz',
	'localizer_1': 'day2/run008/be_avg_mc_run008.nii.gz',
	'localizer_2': 'day2/run009/be_avg_mc_run009.nii.gz',
}


raw_paths = {
	'baseline': 'day1/run001/run001.nii.gz',
	'fear_conditioning': 'day1/run002/run002.nii.gz',
	'extinction': 'day1/run003/run003.nii.gz',
	'extinction_recall': 'day2/run004/run004.nii.gz' ,
	'memory_run_1': 'day2/run005/run005.nii.gz',
	'memory_run_2': 'day2/run006/run006.nii.gz',
	'memory_run_3': 'day2/run007/run007.nii.gz',
	'localizer_1': 'day2/run008/run008.nii.gz',
	'localizer_2': 'day2/run009/run009.nii.gz',
}

temp_mc_paths = {
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

fc_n_trials = {
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

beta_n_trials = {
	'baseline': 48,
	'fear_conditioning': 48,
	'extinction': 48,
	'extinction_recall': 24,
	'extinction_recall_start':25,
	'memory_run_1': 80,
	'memory_run_2': 80,
	'memory_run_3': 80,
	'localizer_1': 24,
	'localizer_2': 24,
}
import os
import numpy as np
import pandas as pd
import pingouin as pg
from scipy.stats import ttest_ind
from nilearn.input_data import NiftiMasker

from fc_config import *

def posthocT():
	nvox = 77779
	voxel_dir = os.path.join(data_dir,'group_ER','voxel_dfs')
	phases = ['baseline','fear_conditioning','extinction']
	cons = ['CS+','CS-']
	stats = ['t','p']
	
	out = {phase:{con:{stat:np.zeros(nvox) for stat in stats} for con in cons} for phase in phases}

	for i in range(nvox):
		vdf = pd.read_csv(os.path.join(voxel_dir,'voxel_%s.csv'%(i)))
		vdf = vdf.groupby(['subject','encode','trial_type']).mean()['rsa'].reset_index()
		vdf['group'] = vdf.subject.apply(lgroup)

		#lets hardcode group pairwise first
		for phase in phases:
			for con in cons:
				Cdat = vdf.rsa[vdf.encode == phase][vdf.trial_type == con][vdf.group == 'control'].values
				Pdat = vdf.rsa[vdf.encode == phase][vdf.trial_type == con][vdf.group == 'ptsd'].values
				for s,stat in enumerate(stats):
					out[phase][con][stat][i] = ttest_ind(Cdat,Pdat)[s]
		
		if i%1000 == 0:print(i)#some marker of progress
	
	#save the output
	outdf = pd.DataFrame([])
	for phase in phases:
		for con in cons:
			for stat in stats:
				outdf[phase+'_'+con+'_'+stat] = out[phase][con][stat]
	outdf.to_csv(os.path.join(data_dir,'group_ER','stats','group_posthoc.csv'))

def reconstruct_stats():
    masker = NiftiMasker(mask_img='/Users/ach3377/Desktop/standard/MNI152_T1_3mm_brain_mask.nii.gz')
    masker.fit()
    #['no_mem_ANOVA','group_posthoc']
    test = 'group_posthoc'
    df = pd.read_csv(os.path.join(data_dir,'group_ER','stats','%s.csv'%(test)),index_col=0)
    # df = 1 - df
    outdir = os.path.join(data_dir,'group_ER','stats','%s_imgs'%(test))
    if not os.path.exists(outdir):os.mkdir(outdir)
    for col in df.columns:
        if 'p' in col: df[col] = 1 - df[col]
        img = masker.inverse_transform(df[col].values)
        nib.save(img,os.path.join(outdir,'%s.nii.gz'%(col)))

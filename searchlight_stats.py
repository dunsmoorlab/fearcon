import os
import numpy as np
import pandas as pd
import pingouin as pg
from scipy.stats import ttest_ind
from nilearn.input_data import NiftiMasker
from fc_config import *
from glm_timing import glm_timing

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
    test = 'no_mem_ANOVA'
    df = pd.read_csv(os.path.join(data_dir,'group_ER','stats','%s.csv'%(test)),index_col=0)
    df = 1 - df
    outdir = os.path.join(data_dir,'group_ER','stats','%s_imgs'%(test))
    if not os.path.exists(outdir):os.mkdir(outdir)
    for col in df.columns:
        # if 'p' in col: df[col] = 1 - df[col]
        img = masker.inverse_transform(df[col].values)
        nib.save(img,os.path.join(outdir,'%s.nii.gz'%(col)))
        
def create_mem_dfs():
    mem_phases = ['memory_run_1','memory_run_2','memory_run_3']
    df = {}
    for sub in all_sub_args:
        sub_df = {}

        for phase in mem_phases: sub_df[phase] = glm_timing(sub,phase).mem_events()

        sub_df = pd.concat(sub_df.values())

        sub_df = sub_df[sub_df.memcond.isin(['Old'])].reset_index(drop=True)
        sub_df = sub_df.drop(columns=['onset','duration','memcond'])
        
        sub_df['subject'] = sub
        
        if sub in sub_args: sub_df['group'] = 'control'
        if sub in p_sub_args: sub_df['group'] = 'ptsd'

        sub_df.acc = (sub_df.acc == 'H').astype(int)
        sub_df.hc_acc = (sub_df.hc_acc == 'H').astype(int)
        
        df[sub] = sub_df

    df = pd.concat(df.values()).reset_index(drop=True)
    df['rsa'] = 0

    #knowing which subs to exlude from memory anovas
    lc_remove = np.ndarray(0)
    hc_remove = np.ndarray(0)
    for sub in all_sub_args:
        for phase in df.encode.unique():
            for con in df.trial_type.unique():
                for mem_cond in [0,1]:
                    if np.where(df.acc[df.subject == sub][df.encode == phase][df.trial_type == con] == mem_cond)[0].shape[0] < 5:
                        lc_remove = np.append(lc_remove,sub)
                    if np.where(df.hc_acc[df.subject == sub][df.encode == phase][df.trial_type == con] == mem_cond)[0].shape[0] < 5:    
                        hc_remove = np.append(hc_remove,sub)
    lc_remove = np.unique(lc_remove)
    hc_remove = np.unique(hc_remove)

    #loading data, actual work here
    data = np.load(os.path.join(data_dir,'group_ER','all_subs_std_item_ER.npy'))
    for i in range(data.shape[2]):
        vdf = df.copy()
        vdf.rsa = data[:,:,i].flatten()
        
        outstr = os.path.join(data_dir,'group_ER','voxel_dfs','voxel_%s.csv'%(i))
        vdf.to_csv(outstr,index=False)
import os
import numpy as np
import pandas as pd
import pingouin as pg
import nibabel as nib
from scipy.stats import ttest_ind, ttest_rel
from nilearn.input_data import NiftiMasker
from fc_config import *
from glm_timing import glm_timing
from joblib import Parallel, delayed
import multiprocessing

def vox_df_posthocT():
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

def reconstruct_stats(group,test,anova=False):
    masker = NiftiMasker(mask_img='/Users/ach3377/Desktop/standard/MNI152_T1_3mm_brain_mask.nii.gz')
    masker.fit()
    #['no_mem_ANOVA','group_posthoc']
    # test = 'no_mem_ANOVA'
    df = pd.read_csv(os.path.join(data_dir,'group_ER','pingouin_stats','%s_%s.csv'%(group,test)),index_col=0)
    if anova: df = 1 - df
    outdir = os.path.join(data_dir,'group_ER','pingouin_stats','imgs',group,test)
    if not os.path.exists(outdir):os.makedirs(outdir)
    for col in df.columns:
        if not anova and 'p' in col: df[col] = 1 - df[col]
        img = masker.inverse_transform(df[col].values)
        nib.save(img,os.path.join(outdir,'%s_%s.nii.gz'%(test,col)))
def create_mem_dfs(group):
    mem_phases = ['memory_run_1','memory_run_2','memory_run_3']
    subjects = {'control':sub_args,
                'ptsd':p_sub_args,
                'all':all_sub_args}

    df = {}
    for sub in subjects[group]:
        sub_df = {}

        for phase in mem_phases: sub_df[phase] = glm_timing(sub,phase).mem_events()

        sub_df = pd.concat(sub_df.values())

        sub_df = sub_df[sub_df.memcond.isin(['Old'])].reset_index(drop=True)
        sub_df = sub_df.drop(columns=['onset','duration','memcond'])
        
        sub_df['subject'] = sub
        
        if group == 'all':
            if sub in sub_args: sub_df['group'] = 'control'
            if sub in p_sub_args: sub_df['group'] = 'ptsd'

        sub_df.acc = (sub_df.acc == 'H').astype(int)
        sub_df.hc_acc = (sub_df.hc_acc == 'H').astype(int)
        
        df[sub] = sub_df

    df = pd.concat(df.values()).reset_index(drop=True)
    df['rsa'] = 0
    df.to_csv(os.path.join(data_dir,'group_ER','%s_template_df.csv'%(group)))
    # #knowing which subs to exlude from memory anovas
    # lc_remove = np.ndarray(0)
    # hc_remove = np.ndarray(0)
    # for sub in all_sub_args:
    #     for phase in df.encode.unique():
    #         for con in df.trial_type.unique():
    #             for mem_cond in [0,1]:
    #                 if np.where(df.acc[df.subject == sub][df.encode == phase][df.trial_type == con] == mem_cond)[0].shape[0] < 5:
    #                     lc_remove = np.append(lc_remove,sub)
    #                 if np.where(df.hc_acc[df.subject == sub][df.encode == phase][df.trial_type == con] == mem_cond)[0].shape[0] < 5:    
    #                     hc_remove = np.append(hc_remove,sub)
    # lc_remove = np.unique(lc_remove)
    # hc_remove = np.unique(hc_remove)

    #loading data, actual work here
    # data = np.load(os.path.join(data_dir,'group_ER','all_subs_std_item_ER.npy'))
    # for i in range(data.shape[2]):
    #     vdf = df.copy()
    #     vdf.rsa = data[:,:,i].flatten()
        
    #     outstr = os.path.join(data_dir,'group_ER','voxel_dfs','voxel_%s.csv'%(i))
    #     vdf.to_csv(outstr,index=False)


def apply_mask(target=None):#meant for TACC
        #pass in an roi and a nifti image to return the data located in the roi mask
        mask = nib.load(std_3mm_brain_mask)
        coor = np.where(mask.get_data() == 1)
        values = target.get_data()[coor]
        if values.ndim > 1:
            values = np.transpose(values) #swap axes to get feature X sample
        return values


def merge_sl_res(group='control'):#meant for TACC
    subjects = {'control':sub_args,
                'ptsd':p_sub_args,
                'all':all_sub_args}
    nvox    = 77779
    ntrial  = 144
    results = np.zeros((len(subjects[group]),ntrial,nvox))
    for i,sub in enumerate(subjects[group]):
        subj = meta(sub)
        results[i] = apply_mask(target=nib.load(os.path.join(subj.rsa,'std_item_ER.nii.gz')))
    save_str = os.path.join(WORK,'group_rsa','%s_std_item_ER.npy'%(group))
    np.save(save_str,results)

def compute_stats(group='control',test='cs_comp'):
    
    df = pd.read_csv(os.path.join(data_dir,'group_ER','%s_template_df.csv'%(group)),index_col=0)
    data = np.load(os.path.join(data_dir,'group_ER','%s_std_item_ER.npy'%(group)))
    nvox = data.shape[2]
    save_str = os.path.join(data_dir,'group_ER','pingouin_stats','%s_%s.csv'%(group,test))

    n_cpus = multiprocessing.cpu_count()

    def twoway_rm(i):
        vdf = df.copy()
        vdf.rsa = data[:,:,i].flatten()
        anova = pg.rm_anova(data=vdf,dv='rsa',within=['encode','trial_type'],subject='subject')#repeated measures anova
        if i%1000 == 0:print(i)#some marker of progress
        return anova['p-unc'].values
    
    def group_phase_comp(i):
        vdf = df.copy()
        vdf.rsa = data[:,:,i].flatten()
        vdf = vdf.set_index(['encode','trial_type','subject'])
        vdf = (vdf.rsa.loc['fear_conditioning'] - vdf.rsa.loc['extinction']).reset_index()
        vdf['group'] = vdf.subject.apply(lgroup)
        anova = pg.mixed_anova(data=vdf,dv='rsa',within='trial_type',between='group',subject='subject')
        if i%1000 == 0:print(i)#some marker of progress
        return anova['p-unc'].values

    def group_cs_comp(i):
        vdf = df.copy()
        vdf.rsa = data[:,:,i].flatten()
        vdf = vdf.set_index(['trial_type','encode','subject'])
        vdf = (vdf.rsa.loc['CS+'] - vdf.rsa.loc['CS-']).reset_index()
        vdf['group'] = vdf.subject.apply(lgroup)
        anova = pg.mixed_anova(data=vdf,dv='rsa',within='encode',between='group',subject='subject')
        if i%1000 == 0:print(i)#some marker of progress
        return anova['p-unc'].values

    def fear_ext(i):
        vdf = df.copy()
        vdf.rsa = data[:,:,i].flatten()
        vdf = vdf.groupby(['encode','trial_type','subject']).mean()
        t, p = ttest_rel(vdf.rsa.loc[('fear_conditioning','CS+')],vdf.rsa.loc[('extinction','CS+')])
        return [t,p]

    def fear_ext_cs(i):
        vdf = df.copy()
        vdf.rsa = data[:,:,i].flatten()
        vdf = vdf.groupby(['trial_type','encode','subject']).mean()
        vdf = vdf.rsa.loc['CS+'] - vdf.rsa.loc['CS-']
        t, p = ttest_rel(vdf.loc['fear_conditioning'],vdf.loc['extinction'])
        return [t,p]

    jobs = {'rm'     :twoway_rm,
            'group_phase_comp':group_phase_comp,
            'group_cs_comp':group_cs_comp,
            'fear_ext':fear_ext,
            'fear_ext_cs':fear_ext_cs}
    
    tests = {'rm'              :['encode','trial_type','intrx'],
             'group_phase_comp':['group','trial_type','intrx'],
             'group_cs_comp'   :['group','encode','intrx'],
             'fear_ext'        :['t','p'],
             'fear_ext_cs'     :['t','p']}

    res = Parallel(n_jobs=n_cpus)(delayed(jobs[test])(i) for i in range(data.shape[2]))

    out = pd.DataFrame(res,columns=tests[test])
    
    out.to_csv(save_str)




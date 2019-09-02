import os
import numpy as np
import pandas as pd
import nibabel as nib
from sys import platform
import copy
from subprocess import Popen

from glob import glob
from scipy.signal import detrend
from scipy.stats import zscore
from nilearn.masking import apply_mask
from nilearn import signal
from shutil import copyfile
import sys
from fc_config import *
from nilearn.input_data import NiftiMasker

from scipy.stats import sem, ttest_ind, ttest_rel
import numpy as np
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from preprocess_library import *
from glm_timing import *

def collect_ev(imgs='tr',nvox='all',SC=True,S_DS=True,rmv_scram=True,rmv_ind=False,verbose=False,
				split=False,con='CS+',binarize=False,save_dict=None,res=None,pres=None):

	if res == None:
		res = group_decode(imgs=imgs, k=nvox,p=False, save_dict=save_dict, rmv_ind=rmv_ind, SC=SC, binarize=binarize, S_DS=S_DS, rmv_scram=rmv_scram, verbose=verbose)
	if pres == None:
		pres = group_decode(imgs=imgs, k=nvox, p=True, save_dict=save_dict, rmv_ind=rmv_ind, SC=SC, binarize=binarize, S_DS=S_DS, rmv_scram=rmv_scram, verbose=verbose)
	
	res.exp_event(con=con,split=split) # this is necessary for both image formats
	# res.vis_exp_event()
	if save_dict == beta_ppa_prepped or save_dict == hippocampus_beta or save_dict == hipp_no_ppa_beta or save_dict == beta_nz_ppa_prepped: 
		res.beta_exp_stats(vis=False)
	else:
		res.exp_stats(vis=False)
	
	if con == 'CS+': res.more_exp_stats()

	context_ev = pd.concat([res.no_df,res.exp_df])

	pres.exp_event(con=con,split=split)
	# pres.vis_exp_event()
	if save_dict == beta_ppa_prepped or save_dict == hippocampus_beta or save_dict == hipp_no_ppa_beta or save_dict == beta_nz_ppa_prepped:
		pres.beta_exp_stats(vis=False)
	else:
		pres.exp_stats(vis=False)
	
	if con == 'CS+': pres.more_exp_stats()

	p_context_ev = pd.concat([pres.no_df,pres.exp_df])
	
	if binarize: comp_cond = 'scrambled'
	else: comp_cond = 'rest'
	# comp_cond = 'rest'
	#take the scence evidene for the first 4 TR of extinciton recall
	idx = pd.IndexSlice
	if save_dict == beta_ppa_prepped  or save_dict == beta_nz_ppa_prepped:
		start_tr = 0
		end_tr = 0
	else:
		start_tr = 0
		end_tr = 3
	cev = {}
	c_init = {}
	res.exp_ev_df.set_index(['subject','condition','trial','tr'],inplace=True)
	res.exp_ev_df.sort_index(level=['subject','condition','trial','tr'],inplace=True)
	for sub in sub_args:
		cev[sub] = res.exp_ev_df['evidence'].loc[idx[sub,'scene',:,start_tr:end_tr]].groupby(['trial']).mean().mean()
		c_init[sub] = res.group_df['scene'].loc['extinction_recall'].loc[sub][0:4].mean()
		# cev_rel[sub] = np.mean(res.group_df['scene'].loc['extinction_recall'].loc[sub][0:4] - res.group_df[comp_cond].loc['extinction_recall'].loc[sub][0:4])
	res.exp_ev_df.reset_index(inplace=True)
	
	# ptsd group
	p_cev = {}
	p_init = {}
	pres.exp_ev_df.set_index(['subject','condition','trial','tr'],inplace=True)
	pres.exp_ev_df.sort_index(level=['subject','condition','trial','tr'],inplace=True)
	for sub in p_sub_args:
		p_cev[sub] = pres.exp_ev_df['evidence'].loc[idx[sub,'scene',:,start_tr:end_tr]].groupby(['trial']).mean().mean()
		p_init[sub] = pres.group_df['scene'].loc['extinction_recall'].loc[sub][0:4].mean()
#		p_cev_rel[sub] = np.mean(pres.group_df['scene'].loc['extinction_recall'].loc[sub][0:4] - pres.group_df[comp_cond].loc['extinction_recall'].loc[sub][0:4])
	pres.exp_ev_df.reset_index(inplace=True)
	


	# output structure
	out = pd.DataFrame([],index=all_sub_args,columns=['ev','Response','Group'])
	out_init = pd.DataFrame([],index=all_sub_args,columns=['ev','Response','Group'])
	for sub in out.index:
		if sub in sub_args: 
			out['ev'].loc[sub] = cev[sub]
			out_init.loc[sub] = c_init[sub]
		else:
			out['ev'].loc[sub] = p_cev[sub]
			out_init.loc[sub] = p_init[sub]

	for no in [1,2,5,7,9,10,13,14,15,16,20,23,24,25,26,101,102,104,105,106,107,109,110,113,116,118,125]:
		out['Response'].loc[no] = 'no'
		out_init['Response'].loc[no] = 'no'		
	
	for exp in [3,4,6,8,12,17,18,19,21,103,108,111,112,114,115,117,120,121,122,123,124]:
		out['Response'].loc[exp] = 'exp'
		out_init['Response'].loc[exp] = 'exp'
	
	for control in sub_args:
		out['Group'].loc[control] = 'Control'
		out_init['Group'].loc[control] = 'Control'
	
	for ptsd in p_sub_args:
		out['Group'].loc[ptsd] = 'PTSD'
		out_init['Group'].loc[ptsd] = 'PTSD'

	out.reset_index(inplace=True)
	out = out.rename_axis({'index':'subject'},axis=1)

	out_init.reset_index(inplace=True)
	out_init = out_init.rename_axis({'index':'subject'},axis=1)

	out.to_csv(os.path.join(data_dir,'graphing','signal_change','mvpa_ev.csv'),index=False)
	out_init.to_csv(os.path.join(data_dir,'graphing','signal_change','mvpa_init_ev.csv'),index=False)
	# out = pd.DataFrame([],index=all_sub_args,columns=['ev','Response','Group'])
	# for sub in out.index:
	# 	if sub < 99: out['ev'].loc[sub] = cev_rel[sub]
	# 	else: out['ev'].loc[sub] = p_cev_rel[sub]

	# for no in [1,2,5,7,9,10,13,14,15,16,20,101,102,104,105,106,107,109,110,113,116,118]: out['Response'].loc[no] = 'no'
	# for exp in [3,4,6,8,12,17,18,19,21,103,108,111,112,114,115,117,120,121]: out['Response'].loc[exp] = 'exp'
	# for control in sub_args: out['Group'].loc[control] = 'Control'
	# for ptsd in p_sub_args: out['Group'].loc[ptsd] = 'PTSD'

	# out.reset_index(inplace=True)
	# out = out.rename_axis({'index':'subject'},axis=1)

	# out.to_csv(os.path.join(data_dir,'graphing','signal_change','mvpa_rel_ev.csv'),index=False)
#fig, ax1 = plt.subplots()
#ax1 = sns.scatterplot(x='ev',y='pe',data=out,hue='resp')
#plt.show()
def collect_psc(run=0):


	peg = {}
	for sub in all_sub_args:
		peg[sub] = {}
		subj = meta(sub)
		# for roi in ['amygdala','mOFC','vmPFC','hippocampus','dACC']:
		for roi in ['amygdala_beta','mOFC_beta','vmPFC_beta','hippocampus_beta','dACC_beta','insula_beta']:
			peg[sub][roi] = {}
 		# for roi in ['vmPFC']:
			# fq = pd.read_csv(os.path.join(subj.model_dir,'GLM/run004/','v2_psc.feat',roi,'report.txt'),sep=' ',header=None)
			fq = pd.read_csv(os.path.join('/Volumes/DunsmoorRed/all_psc',subj.fsub,'v2_psc.feat',roi,'report.txt'),sep=' ',header=None)
			# fq = pd.read_csv(os.path.join('/Volumes/DunsmoorRed/run00%s/feats'%(run),subj.fsub,'lin_run00%s.feat'%(run),roi,'report.txt'),sep=' ',header=None)
			# fq = pd.read_csv(os.path.join('/Users/ach3377/Desktop/fc',subj.fsub,'ev_corr.feat',roi,'report.txt'),sep=' ',header=None)
			
			fq.rename_axis({1:'stats_image',2:'nvox',3:'min',4:'10',5:'mean',6:'median',7:'90',8:'max',9:'stddev'},axis='columns',inplace=True)
			# for i,img in enumerate(['junk1','junk2','junk3','jnk4','CSp_pe','CSm_pe','CSp_CSm','CSm_CSp']):
			for i,img in enumerate(['early_CSp_pe','early_CSm_pe','late_CSp_pe','late_CSm_pe','early_CSp_cope','early_CSm_cope','early_CSp_CSm','early_CSm_CSp','late_CSp_cope','late_CSm_cope','late_CSp_CSm','late_CSm_CSp']):
			# for i,img in enumerate(['early_CSp_pe','early_CSm_pe','late_CSp_pe','late_CSm_pe','early_CSp_cope','early_CSp_CSm','late_CSp_cope','late_CSp_CSm','all_CSp_cope','all_CSp_CSm']):
				peg[sub][roi][img] = fq['mean'][i]

	peg = pd.DataFrame.from_dict({(roi,sub): peg[sub][roi]
									for sub in peg.keys()
									for roi in peg[sub].keys()},
									orient='index')
	peg = peg.reset_index().rename_axis({'level_0':'roi','level_1':'sub'},axis=1)
	peg.to_csv(os.path.join(data_dir,'graphing','signal_change','run00%s_beta_values.csv'%(run)),sep=',',index=False)

	# peg.to_csv(os.path.join(data_dir,'graphing','signal_change','group_mask_psc_values.csv'),sep=',',index=False)

# def pe_analysis(phase='extinction_recall',roi=None):

# 	peg = pd.DataFrame([],index=sub_args,columns=['start','CSp','CSm','resp'])

# 	for sub in sub_args:
# 		subj = meta(sub)
# 		ev = glm_timing(sub,phase).phase_events(er_start=True,con=True)

# 		# beta = np.load(os.path.join(subj.bold_dir,'day2','run002','%s_b_pp_run004.npz'%(roi)))[dataz]
# 		beta = np.load(os.path.join(subj.bold_dir,'day1','run002','%s_run002.npz'%(roi)))[dataz]
# 		start = []
# 		csp = []
# 		csm = []
		
# 		# for trial in range(beta.shape[0]):
# 		for trial in range(9):
# 			if ev[trial] == 'start': start.append(beta[trial].mean())
# 			elif ev[trial] == 'CS+': csp.append(beta[trial].mean())
# 			elif ev[trial] == 'CS-': csm.append(beta[trial].mean())

# 		peg['start'][sub] = np.array(start).mean()
# 		peg['CSp'][sub] = np.array(csp).mean()
# 		peg['CSm'][sub] = np.array(csm).mean()

# 	print(ttest_rel(peg['CSp'],peg['CSm']))
# 	print('CS+ mean: %s'%(np.array(csp).mean()))
# 	print('CS- mean: %s'%(np.array(csm).mean()))

# 	for no in [1,2,5,7,9,10,13,14,15,16,20]: peg['resp'][no] = 'no'
# 	for exp in [3,4,6,8,12,17,18,19,21]: peg['resp'][exp] = 'exp'

# 	peg.to_csv(os.path.join(data_dir,'graphing','signal_change','group_pe.csv'))

def run_featquery(sub=None,roi=None,run=0):
	print(sub)
	subj = meta(sub)
	feat_dest = '/Volumes/DunsmoorRed/run00%s/feats'%(run)
	os.system('cp %s %s'%(os.path.join(subj.roi,'%s_mask.nii.gz'%(roi)),os.path.join('/Volumes/DunsmoorRed/all_psc',subj.fsub,'%s_mask.nii.gz'%(roi))))
	os.system('featquery 1 /Volumes/DunsmoorRed/all_psc/%s/v2_psc.feat 12 stats/pe1 stats/pe2 stats/pe3 stats/pe4 stats/cope1 stats/cope2 stats/cope3 stats/cope4 stats/cope5 stats/cope6 stats/cope7 stats/cope8 %s_beta /Volumes/DunsmoorRed/all_psc/%s/%s_mask.nii.gz'%(subj.fsub,roi,subj.fsub,roi))
	# os.system('cp %s %s'%(os.path.join(subj.roi,'%s_mask.nii.gz'%(roi)),os.path.join(feat_dest,subj.fsub,'%s_mask.nii.gz'%(roi))))
	# os.system('featquery 1 %s/%s/lin_run00%s.feat 8 stats/pe1 stats/pe2 stats/pe3 stats/pe4 stats/cope1 stats/cope2 stats/cope3 stats/cope4 %s_beta %s/%s/%s_mask.nii.gz'%(feat_dest,subj.fsub,run,roi,feat_dest,subj.fsub,roi))


def manual_psc(subs=all_sub_args):
	copes = os.path.join(data_dir,'rsa_copes')
	rois = ['amygdala','mOFC','vmPFC','hippocampus','PPA','dACC']
	labels = ['baseline','fear','ext','early_fear','late_fear','early_ext','late_ext','early_rnw','late_rnw']
	conds = ['csp','csp_csm']
	df = pd.DataFrame(index=pd.MultiIndex.from_product((subs,rois,labels,conds)),columns=['beta'])

	for sub in subs:
		print(sub)
		subj = meta(sub)
		for roi in rois:
			masker = NiftiMasker(mask_img=os.path.join(subj.roi,'%s_mask.nii.gz'%(roi)),memory='nilearn_cache',memory_level=2)
			for label in labels:
				for cond in conds:
					cope = masker.fit_transform(os.path.join(copes,subj.fsub,'%s_%s.nii.gz'%(label,cond)))
					df['beta'].loc[sub,roi,label,cond] = cope.mean()
	df.reset_index(inplace=True)
	df = df.rename(columns={'level_0':'subject','level_1':'roi','level_2':'label','level_3':'con'})
	df['group'] = np.repeat(['control','ptsd'],2592)
	df.to_csv(os.path.join(data_dir,'graphing','signal_change','manual_betas.csv'))
	return df


from mvpa_analysis import *
from fc_config import data_dir
from scipy.stats import sem, ttest_ind
import numpy as np
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import pandas as pd
import seaborn as sns
from scipy.stats import sem
from scipy.stats import linregress as lm
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from fc_config import *
from wesanderson import wes_palettes
from signal_change import *
import sys
from scipy.signal import exponential

imgs='beta'
nvox = 'all'
SC=False
S_DS=False
rmv_scram=False
rmv_ind=False #also collapses across animals/tools
rmv_rest=False
verbose=True
split=False
con = 'CS+'
binarize=True
save_dict=beta_ppa_prepped

conds = ['scene','scrambled']
res = group_decode(conds=conds, imgs=imgs, k=nvox, save_dict=save_dict, rmv_rest=rmv_rest, binarize=binarize, SC=SC, S_DS=S_DS, rmv_scram=rmv_scram, rmv_ind=rmv_ind, verbose=verbose)
pres = group_decode(conds=conds, imgs=imgs, k=nvox, save_dict=save_dict, rmv_rest=rmv_rest, binarize=binarize, SC=SC, S_DS=S_DS, rmv_scram=rmv_scram, rmv_ind=rmv_ind, p=True, verbose=verbose)

collect_ev(res=res,pres=pres)

if imgs == 'tr': iNdEx = ['subject','trial','tr']
else: iNdEx = ['subject','trial']
cev_corr = res.exp_ev_df.loc[np.where(res.exp_ev_df.condition == 'scene')[0]] 
for i in cev_corr.index:
	if cev_corr['response'].loc[i] == 'expect':
		cev_corr['response'].loc[i] = 1
	elif cev_corr['response'].loc[i] == 'no':
		cev_corr['response'].loc[i] = 0
cev_corr = cev_corr.set_index(iNdEx)


pev_corr = pres.exp_ev_df.loc[np.where(pres.exp_ev_df.condition == 'scene')[0]] 
for i in pev_corr.index:
	if pev_corr['response'].loc[i] == 'expect':
		pev_corr['response'].loc[i] = 1
	elif pev_corr['response'].loc[i] == 'no':
		pev_corr['response'].loc[i] = 0
pev_corr = pev_corr.set_index(iNdEx)

# decay = [1,1,1,1]
# decay = exponential(5,0,-(4-1)/np.log(.25),False)
decay = exponential(5,center=0,tau=-(5-1)/np.log(.1),sym=False)[0:4]

exp_corr = pd.DataFrame([],index=all_sub_args,columns=['response','evidence','Group'])
for sub in sub_args:
	exp_corr['response'].loc[sub] = np.mean(cev_corr['response'].loc[sub] * decay)
	exp_corr['evidence'].loc[sub] = np.mean(cev_corr['evidence'].loc[sub] * decay)
	exp_corr['Group'].loc[sub] = 'Control'

for sub in p_sub_args:
	exp_corr['response'].loc[sub] = np.mean(pev_corr['response'].loc[sub] * decay)
	exp_corr['evidence'].loc[sub] = np.mean(pev_corr['evidence'].loc[sub] * decay)
	exp_corr['Group'].loc[sub] = 'PTSD'

exp_corr['response'] = exp_corr['response'].astype(float)
exp_corr['evidence'] = exp_corr['evidence'].astype(float)

exp_corr.to_csv('/Users/ach3377/Db_lpl/STUDY/FearCon/graphing/signal_change/ev_exp.csv',header=True,index=False)

formula = 'evidence ~ response + Group'
model = ols(formula, exp_corr).fit()
aov_table = anova_lm(model)
print(aov_table)


ec = sns.lmplot(x='response', y='evidence', data=exp_corr, col='Group', hue='Group',
				palette=[wes_palettes['Chevalier'][0],wes_palettes['FantasticFox'][3]])

c_ec = lm(exp_corr['response'].loc[sub_args],exp_corr['evidence'].loc[sub_args])
p_ec = lm(exp_corr['response'].loc[p_sub_args],exp_corr['evidence'].loc[p_sub_args])
print(c_ec)
print(p_ec)
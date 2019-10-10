import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
from fc_config import *
from wesanderson import wes_palettes
from signal_change import *
from beta_rsa import *
from mvpa_analysis import *
from scipy.stats import sem, ttest_ind
from scipy.stats import sem
from scipy.stats import linregress as lm

sns.set_context('poster')
sns.set_style('whitegrid')


imgs='beta'
SC=False
S_DS=False
rmv_scram=False
rmv_ind=False #also collapses across animals/tools
rmv_rest=False
verbose=False
split=False
con = 'CS+'
binarize=True
save_dict=beta_ppa_prepped
nvox = 'all'

conds = ['scene','scrambled']
res = group_decode(conds=conds, imgs=imgs, k=nvox, save_dict=save_dict, rmv_rest=rmv_rest, binarize=binarize, SC=SC, S_DS=S_DS, rmv_scram=rmv_scram, rmv_ind=rmv_ind, verbose=verbose)
pres = group_decode(conds=conds, imgs=imgs, k=nvox, save_dict=save_dict, rmv_rest=rmv_rest, binarize=binarize, SC=SC, S_DS=S_DS, rmv_scram=rmv_scram, rmv_ind=rmv_ind, p=True, verbose=verbose)

collect_ev(res=res,pres=pres,save_dict=save_dict,split=split)

ev = pd.read_csv(os.path.join(data_dir,'graphing','signal_change','mvpa_ev.csv'))
ev = ev.rename(columns={'ev':'Context Reinstatement'})
psc = pd.read_csv(os.path.join(data_dir,'graphing','signal_change','beta_values.csv'))
psc['Group'] = np.tile(np.repeat(['control','ptsd'],24),4)
epsc = pd.read_csv(os.path.join(data_dir,'graphing','signal_change','run003_beta_values.csv'))
gev = ev.copy()


##########Early ctx group comparison###########
fig, gctx = plt.subplots()	
gctx = sns.swarmplot(x='Group',y='Context Reinstatement',data=gev, size=10, alpha=.9,
	palette=[wes_palettes['Darjeeling2'][1],wes_palettes['FantasticFox'][-1]])
sns.pointplot(x='Group',y='Context Reinstatement',data=gev,
				join=False,capsize=.2,palette=['black','black'])
gctx.set_ylim(0,.8)
plt.title('')
gctx.set_xlabel('')
gctx.set_ylabel('')

##########ROI ctx correlations#################
control_psc = 
'''
steps to prep maps for pyCortex:
threshold both, multiple one by -1, add them togher
'''




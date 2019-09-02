import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
from fc_config import *
from fc_behavioral import *
from scr_analysis import scr_stats
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
psc['Group'] = np.tile(np.repeat(('control','ptsd'),24),5)
epsc = pd.read_csv(os.path.join(data_dir,'graphing','signal_change','run003_beta_values.csv'))
gev = ev.copy()


##########Early ctx group comparison###########
fig, gctx = plt.subplots()	
gctx = sns.swarmplot(x='Group',y='Context Reinstatement',data=gev, size=10, alpha=.9,
	palette=[wes_palettes['Chevalier'][0],wes_palettes['FantasticFox'][-1]])
sns.pointplot(x='Group',y='Context Reinstatement',data=gev,
				join=False,capsize=.1,palette=['black','black'])
gctx.set_ylim(0,.8)
plt.title('Early CS+ Context Reinstatement')
gctx.set_xtitle('')
gctx.set_ytitle('')

##########ROI ctx correlations#################
control_psc = psc[psc.Group == 'control']
ptsd_psc = psc[psc.Group == 'ptsd']

control_psc = pd.concat((control_psc[control_psc.roi == 'mOFC_beta'],control_psc[control_psc.roi == 'hippocampus_beta']))
ptsd_psc = pd.concat((ptsd_psc[ptsd_psc.roi == 'mOFC_beta'],ptsd_psc[ptsd_psc.roi == 'hippocampus_beta']))

control_psc['ev'] = np.tile(ev['Context Reinstatement'][ev.Group == 'Control'],2)
ptsd_psc['ev'] = np.tile(ev['Context Reinstatement'][ev.Group == 'PTSD'],2)

# gpsc = pd.concat((control_psc,ptsd_psc),axis=0)

# glm = sns.lmplot(x='ev', y='early_CSp_CSm', data=gpsc, col='roi', hue='Group',
# 				col_order=['hippocampus_beta','amygdala_beta'],legend=False,
# 				palette=[wes_palettes['Chevalier'][0],wes_palettes['FantasticFox'][-1]])

clm = sns.lmplot(x='ev', y='early_CSp_CSm', data=control_psc, col='roi', hue='Group',
				col_order=['mOFC_beta','hippocampus_beta'],legend=False,
				palette=[wes_palettes['Chevalier'][0]])
clm.axes[0,0].set_xlim(0,.8)

plm = sns.lmplot(x='ev', y='early_CSp_CSm', data=ptsd_psc, col='roi', hue='Group',
				col_order=['mOFC_beta','hippocampus_beta'],legend=False,
				palette=[wes_palettes['FantasticFox'][-1]])
plm.axes[0,0].set_xlim(.1,.8)

A = pd.concat([control_psc,ptsd_psc])
alm = sns.lmplot(x='ev',y='early_CSp_CSm',data=A,col='roi',hue='Group',
				col_order=['mOFC_beta','hippocampus_beta'],legend=False,
				palette=[wes_palettes['Chevalier'][0],wes_palettes['FantasticFox'][-1]])

#########Expectancy plot#######################

save_dict=ppa_prepped
res = group_decode(conds=conds, imgs=imgs, k=nvox, save_dict=save_dict, rmv_rest=rmv_rest, binarize=binarize, SC=SC, S_DS=S_DS, rmv_scram=rmv_scram, rmv_ind=rmv_ind, verbose=verbose)
pres = group_decode(conds=conds, imgs=imgs, k=nvox, save_dict=save_dict, rmv_rest=rmv_rest, binarize=binarize, SC=SC, S_DS=S_DS, rmv_scram=rmv_scram, rmv_ind=rmv_ind, p=True, verbose=verbose)
collect_ev(res=res,pres=pres,save_dict=save_dict,split=split)
trial = 1

res.exp_stats(trial_=trial,vis=False)
pres.exp_stats(trial_=trial,vis=False)
df = pd.DataFrame([], columns=['Group','Response','Relative Context Reinstatement'])
n_con = res._no_[trial].shape[0] + res._expect_[trial].shape[0]
n_p = pres._no_[trial].shape[0] + pres._expect_[trial].shape[0]
df['Group']	= np.concatenate((np.repeat(['Control'],n_con), np.repeat(['PTSD'],n_p)))
df['Response']	= np.concatenate((np.repeat(['Do not expect'],res._no_[trial].shape[0]), np.repeat(['Expect'],res._expect_[trial].shape[0]),
								np.repeat(['Do not expect'],pres._no_[trial].shape[0]), np.repeat(['Expect'],pres._expect_[trial].shape[0])))
df['Relative Context Reinstatement'] = np.concatenate((res._no_[trial],res._expect_[trial],pres._no_[trial],pres._expect_[trial]))
df['outcome'] = 0


fig, exp = plt.subplots()
exp = sns.swarmplot(x='Group',y='Relative Context Reinstatement',hue='Response',data=df,size=10,hue_order=['Expect','Do not expect'],
						dodge=True,palette=[wes_palettes['Royal1'][0],wes_palettes['Darjeeling2'][1]])
sns.pointplot(x='Group',y='Relative Context Reinstatement',hue='Response',data=df,hue_order=['Expect','Do not expect'],
	join=False,palette=['black','black'],capsize=.1,dodge=.45)
exp.set_ylim(-.1,1.1)
exp.legend_.remove()

########Behavior##############
#Expectancy
cexp = shock_expectancy()
pexp = shock_expectancy(p=True)
cond = ['CS+']
fig3, bexp = plt.subplots()
bexp.plot(range(1,13), cexp.prop_df['avg'].loc['extinction_recall'].loc[cond], marker='o', color=wes_palettes['Chevalier'][0], markersize=6, linewidth=4)

bexp.plot(range(1,13), pexp.prop_df['avg'].loc['extinction_recall'].loc[cond], marker='o', color=wes_palettes['FantasticFox'][-1], markersize=6, linewidth=4)
bexp.set_xticks(np.arange(2,13,2))
bexp.set_yticks(np.arange(.2,1.2,.2)) 
bexp.set_ylim([0,1])
bexp.set_xlim([.5,12.5])

#SCR

scr = scr_stats().scr
scr = scr[scr.condition == 'CS+']
gscr = sns.pointplot(x='trial',y='scr',hue='group',data=scr,
	join=True,palette=[wes_palettes['Chevalier'][0],wes_palettes['FantasticFox'][-1]],capsize=.1,dodge=.45)
gscr.legend_.remove()


scr = scr_stats().scr
scr.set_index(['trial'],inplace=True)
scr.drop(index=range(5,13),inplace=True)
scr.reset_index(inplace=True)
scr.set_index(['subject','trial'],inplace=True)
csp = scr[scr.condition == 'CS+']
csm = scr[scr.condition == 'CS-']

csp = csp.groupby(level=0).mean().reset_index()
csm = csm.groupby(level=0).mean().reset_index()

diff = pd.DataFrame()
diff['subject'] = all_sub_args
diff['diff_scr'] = csp.scr-csm.scr
diff['group'] = np.repeat(['control','ptsd'],24)
diff.to_csv(os.path.join(data_dir,'graphing','SCR','diff_e_rnw.csv'),index=False)
########RSA###################
roi = 'mOFC'
con = 'csp'
split = False
label = 'e__e_r'

if split: rdf = pd.read_csv(os.path.join(data_dir,'graphing','beta_rsa','split_%s_%s.csv'%(roi,con)))
else: rdf = pd.read_csv(os.path.join(data_dir,'graphing','beta_rsa','%s_%s.csv'%(roi,con)))
rdf = rdf[rdf.label == label]
rdf.reset_index(inplace=True,drop=True)
rdf['response'] = gev.Response

sns.pointplot(x='group',y='z',hue='response',data=rdf,dodge=True)

rdf.set_index('sub',inplace=True)
mOFC_out = [5,23,103,107,116]
for sub in mOFC_out:rdf.drop(index=sub,axis=0,inplace=True)
rdf.reset_index(inplace=True)

fig, rsa = plt.subplots()
rsa = sns.swarmplot(x='group',y='z',hue='group',data=rdf,size=10,
					palette=[wes_palettes['Chevalier'][0],wes_palettes['FantasticFox'][-1]])
sns.pointplot(x='group',y='z',data=rdf,join=False,capsize=.1,
				palette=['black','black'])
rsa.legend_.remove()
plt.tight_layout()
print(ttest_ind(rdf.z[rdf.group=='control'],rdf.z[rdf.group=='ptsd']))

########BETAS###################
roi = 'amygdala'
group = 'control'

betas = pd.read_csv(os.path.join(data_dir,'graphing','signal_change','manual_betas.csv'))
betas=betas[betas.group == group]

# betas=betas[betas.roi == roi]
sns.catplot(x='label',y='beta',hue='con',row='roi',data=betas,kind='point',dodge=True)

sns.pointplot(x='label',y='beta',hue='con',data=betas,dodge=True)






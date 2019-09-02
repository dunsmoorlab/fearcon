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
from scipy.signal import exponential

roi = 'PPA'
con = 'csp'
split = False
label = 'e__e_r'


if split: rdf = pd.read_csv(os.path.join(data_dir,'graphing','beta_rsa','split_%s_%s.csv'%(roi,con)))
else: rdf = pd.read_csv(os.path.join(data_dir,'graphing','beta_rsa','%s_%s.csv'%(roi,con)))
rdf = rdf[rdf.label == label]
rdf.reset_index(inplace=True,drop=True)
rdf = rdf.rename(columns={'z':'ev'})
rdf.to_csv('/Users/ach3377/Desktop/ppa_rsa.csv')
rdf = rdf.rename(columns={'ev':'z'})
# fig, rsa = plt.subplots()
# rsa = sns.swarmplot(x='group',y='z',hue='group',data=rdf,size=10,
# 					palette=[wes_palettes['Chevalier'][0],wes_palettes['FantasticFox'][-1]])
# sns.pointplot(x='group',y='z',data=rdf,join=False,capsize=.1,
# 				palette=['black','black'])
# rsa.legend_.remove()
# plt.tight_layout()
# print(ttest_ind(rdf.z[rdf.group=='control'],rdf.z[rdf.group=='ptsd']))

# rdf['response'] = gev.Response
ev = pd.read_csv(os.path.join(data_dir,'graphing','signal_change','mvpa_ev.csv'))
ev['z'] = rdf.z

sns.lmplot(x='ev',y='z',col='Group',hue='Group',data=ev,size=10,palette=[wes_palettes['Chevalier'][0],wes_palettes['FantasticFox'][-1]])

psc = pd.read_csv(os.path.join(data_dir,'graphing','signal_change','beta_values.csv'))
psc['Group'] = np.tile(np.repeat(('control','ptsd'),24),5)

control_psc = psc[psc.Group == 'control']
ptsd_psc = psc[psc.Group == 'ptsd']

control_psc = pd.concat((control_psc[control_psc.roi == 'mOFC_beta'],control_psc[control_psc.roi == 'hippocampus_beta']))
ptsd_psc = pd.concat((ptsd_psc[ptsd_psc.roi == 'mOFC_beta'],ptsd_psc[ptsd_psc.roi == 'hippocampus_beta']))
control_psc.roi = np.repeat(['vmPFC_beta','hippocampus_beta'],24)
ptsd_psc.roi = np.repeat(['vmPFC_beta','hippocampus_beta'],24)

control_psc['z'] = np.tile(ev['z'][ev.Group == 'Control'],2)
ptsd_psc['z'] = np.tile(ev['z'][ev.Group == 'PTSD'],2)

clm = sns.lmplot(x='z', y='early_CSp_CSm', data=control_psc, col='roi', hue='Group',
				col_order=['vmPFC_beta','hippocampus_beta'],legend=False,
				palette=[wes_palettes['Chevalier'][0]])

plm = sns.lmplot(x='z', y='early_CSp_CSm', data=ptsd_psc, col='roi', hue='Group',
				col_order=['vmPFC_beta','hippocampus_beta'],legend=False,
				palette=[wes_palettes['FantasticFox'][-1]])



sns.pointplot(x='group',y='z',hue='response',data=rdf,dodge=True)

# rdf.set_index('sub',inplace=True)
# mOFC_out = [5,23,103,107,116]
# for sub in mOFC_out:rdf.drop(index=sub,axis=0,inplace=True)
# rdf.reset_index(inplace=True)



res.exp_ev_df.set_index(['subject','condition','trial','tr'],inplace=True)
res.exp_ev_df.sort_index(level=['subject','condition','trial','tr'],inplace=True)
pres.exp_ev_df.set_index(['subject','condition','trial','tr'],inplace=True)
pres.exp_ev_df.sort_index(level=['subject','condition','trial','tr'],inplace=True)
idx = pd.IndexSlice

if imgs == 'tr': iNdEx = ['subject','trial','tr']
else: iNdEx = ['subject','trial']
# cev_corr = res.exp_ev_df.loc[np.where(res.exp_ev_df.condition == 'scene')[0]] 
cev_corr = pd.DataFrame([],index=pd.MultiIndex.from_product((sub_args,range(1,5)),names=['subject','trial']),columns=['evidence','response'])
for sub in sub_args:
	for trial in range(1,5):
		cev_corr['evidence'].loc[(sub,trial)] = res.exp_ev_df.loc[idx[sub,'scene',trial,1:3],'evidence'].mean()
		cev_corr['response'].loc[(sub,trial)] = res.exp_ev_df.loc[idx[sub,'scene',trial,0],'response']	
cev_corr.reset_index(inplace=True)
cev_corr.set_index(['subject'])

for i in cev_corr.index:
	if cev_corr['response'].loc[i] == 'expect':
		cev_corr['response'].loc[i] = 1
	elif cev_corr['response'].loc[i] == 'no':
		cev_corr['response'].loc[i] = 0
cev_corr = cev_corr.set_index(iNdEx)


# pev_corr = pres.exp_ev_df.loc[np.where(pres.exp_ev_df.condition == 'scene')[0]] 
pev_corr = pd.DataFrame([],index=pd.MultiIndex.from_product((p_sub_args,range(1,5)),names=['subject','trial']),columns=['evidence','response'])
for sub in p_sub_args:
	for trial in range(1,5):
		pev_corr['evidence'].loc[(sub,trial)] = pres.exp_ev_df.loc[idx[sub,'scene',trial,1:3],'evidence'].mean()
		pev_corr['response'].loc[(sub,trial)] = pres.exp_ev_df.loc[idx[sub,'scene',trial,0],'response']	
pev_corr.reset_index(inplace=True)
pev_corr.set_index(['subject'])

for i in pev_corr.index:
	if pev_corr['response'].loc[i] == 'expect':
		pev_corr['response'].loc[i] = 1
	elif pev_corr['response'].loc[i] == 'no':
		pev_corr['response'].loc[i] = 0
pev_corr = pev_corr.set_index(iNdEx)


# N = 4
# end_value = .25
# decay = exponential(N,0,-(N-1)/np.log(end_value),False)
# decay = [1,.75,.5,.25]
decay = [1,1,1,1]

exp_corr = pd.DataFrame([],index=all_sub_args,columns=['response','z','Group'])
for sub in sub_args:
	exp_corr['response'].loc[sub] = np.mean(cev_corr['response'].loc[sub] * decay)
	# exp_corr['evidence'].loc[sub] = np.mean(cev_corr['evidence'].loc[sub] * decay)
	exp_corr['Group'].loc[sub] = 'Control'

for sub in p_sub_args:
	exp_corr['response'].loc[sub] = np.mean(pev_corr['response'].loc[sub] * decay)
	# exp_corr['evidence'].loc[sub] = np.mean(pev_corr['evidence'].loc[sub] * decay)
	exp_corr['Group'].loc[sub] = 'PTSD'
exp_corr.z = rdf.z.values


exp_corr['response'] = exp_corr['response'].astype(float)
exp_corr['z'] = exp_corr['z'].astype(float)

ec = sns.lmplot(x='response', y='z', data=exp_corr, col='Group', hue='Group',
				palette=[wes_palettes['Chevalier'][0],wes_palettes['FantasticFox'][-1]])

c_ec = lm(exp_corr['response'].loc[sub_args],exp_corr['z'].loc[sub_args])
p_ec = lm(exp_corr['response'].loc[p_sub_args],exp_corr['z'].loc[p_sub_args])
print(c_ec)
print(p_ec)


meadian_var = 'response'
c_l = []
c_h = []
for sub in sub_args:
	if exp_corr[meadian_var][sub] <= np.median(exp_corr[meadian_var][sub_args]):
		c_l.append(sub)
	else:
		c_h.append(sub)
p_l = []
p_h = []
for sub in p_sub_args:
	if exp_corr[meadian_var][sub] <= np.median(exp_corr[meadian_var][p_sub_args]):
		p_l.append(sub)
	else:
		p_h.append(sub)

print(ttest_ind(exp_corr['z'][c_l], exp_corr['z'][c_h]))
print(ttest_ind(exp_corr['z'][p_l], exp_corr['z'][p_h]))
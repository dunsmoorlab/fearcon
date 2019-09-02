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
from beta_rsa import *

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
save_dict=ppa_prepped
nvox = 'all'
#if save_dict != ppa_prepped and nvox == 'all': sys.exit()
# conds=['stim','scene','scrambled']
# conds=['csplus','csminus','scene','scrambled']
# conds=['csplus','csminus','scene','rest']
conds = ['scene','scrambled']
res = group_decode(conds=conds, imgs=imgs, k=nvox, save_dict=save_dict, rmv_rest=rmv_rest, binarize=binarize, SC=SC, S_DS=S_DS, rmv_scram=rmv_scram, rmv_ind=rmv_ind, verbose=verbose)
pres = group_decode(conds=conds, imgs=imgs, k=nvox, save_dict=save_dict, rmv_rest=rmv_rest, binarize=binarize, SC=SC, S_DS=S_DS, rmv_scram=rmv_scram, rmv_ind=rmv_ind, p=True, verbose=verbose)

collect_ev(res=res,pres=pres,save_dict=save_dict,split=split)
# for trial in [1,2,3,4]:
for trial in [1]:
	res.exp_stats(trial_=trial,vis=False)
	pres.exp_stats(trial_=trial,vis=False)
	print(res.ind_df)
	print(pres.ind_df)
	df = pd.DataFrame([], columns=['Group','Response','Relative Context Reinstatement'])
	n_con = res._no_[trial].shape[0] + res._expect_[trial].shape[0]
	n_p = pres._no_[trial].shape[0] + pres._expect_[trial].shape[0]

	df['Group']	= np.concatenate((np.repeat(['Control'],n_con), np.repeat(['PTSD'],n_p)))
	df['Response']	= np.concatenate((np.repeat(['Do not expect'],res._no_[trial].shape[0]), np.repeat(['Expect'],res._expect_[trial].shape[0]),
									np.repeat(['Do not expect'],pres._no_[trial].shape[0]), np.repeat(['Expect'],pres._expect_[trial].shape[0])))

	df['Relative Context Reinstatement'] = np.concatenate((res._no_[trial],res._expect_[trial],pres._no_[trial],pres._expect_[trial]))
	df['outcome'] = 0

	no = [res._no_[trial].mean(),pres._no_[trial].mean()]
	no_err = [sem(res._no_[trial]), sem(pres._no_[trial])]

	yes = [res._expect_[trial].mean(), pres._expect_[trial].mean()]
	yes_err = [sem(res._expect_[trial]), sem(pres._expect_[trial])]

	fig, ax55 = plt.subplots()

	sns.set_context('poster')
	sns.set_style('whitegrid',{'axes.facecolor':'.9','figure.facecolor':'.9'})	
	ax55 = sns.swarmplot(x='Group',y='Relative Context Reinstatement',hue='Response',data=df, size=10, alpha=.9,
						dodge=True,palette=[wes_palettes['FantasticFox'][2],wes_palettes['Royal1'][0]])
	plt.title('Renewal Test: 1st CS+ Trial')

	sns.pointplot(x='Group',y='Relative Context Reinstatement',hue='Response',data=df,
		join=False,palette=['black','black'],capsize=.1,dodge=.45)
	# ax55.legend_.remove()
	plt.tight_layout()
	plt.savefig(os.path.join(data_dir,'graphing','sfn18','exp_splot'),dpi=300)

	fig, ax56 = plt.subplots()
	ax56 = sns.swarmplot(x='Group',y='Relative Context Reinstatement',hue='Group',data=df, size=10, alpha=.9,
						dodge=False,palette=[wes_palettes['Chevalier'][0],wes_palettes['Royal1'][1]])
	plt.title('Renewal Test: 1st CS+ Trial')

	sns.pointplot(x='Group',y='Relative Context Reinstatement',hue='Group',data=df,
		join=False,palette=['black','black'],capsize=.1,dodge=.2)
	# ax56.legend_.remove()

	# legend_elements = [Patch(facecolor=wes_palettes['FantasticFox'][2],edgecolor='#535353',label='Do not expect'),
						# Patch(facecolor=wes_palettes['Royal1'][0],edgecolor='#535353',label='Expect')]
	# plt.legend(handles=legend_elements,title='Do you expect a shock?',bbox_to_anchor=(1,.5))
	plt.tight_layout()


	ax55.legend_.remove()

	df.outcome.loc[np.where(df.Response == 'no')[0]] = 1

	df = df.rename(columns={'Relative Context Reinstatement':'baseline'})

	# aov(cr ~ (condition * phase * group) + Error(subject/(condition*phase)), data=cr_dat)
	formula = 'baseline ~ C(Group) + C(Response) + C(Group):C(Response)'
	model = ols(formula, df).fit()
	aov_table = anova_lm(model, typ=2)
	print(trial,aov_table)

	control = np.concatenate((res._no_[trial],res._expect_[trial]))
	ptsd = np.concatenate((pres._no_[trial],pres._expect_[trial]))
	print(ttest_ind(control,ptsd))

	print(trial,'c_no v c_exp', ttest_ind(res._no_[trial], res._expect_[trial]))
	print(trial,'c_no v p_no', ttest_ind(res._no_[trial], pres._no_[trial]))
	print(trial,'c_no v p_exp', ttest_ind(res._no_[trial], pres._expect_[trial]))


ev = pd.read_csv(os.path.join(data_dir,'graphing','signal_change','mvpa_ev.csv'))
ev = ev.rename(columns={'ev':'Context Reinstatement'})
psc = pd.read_csv(os.path.join(data_dir,'graphing','signal_change','beta_values.csv'))
epsc = pd.read_csv(os.path.join(data_dir,'graphing','signal_change','run003_beta_values.csv'))
gev = ev.copy()

evmeans = [gev['Context Reinstatement'].loc[gev.Group == 'Control'].mean(),gev['Context Reinstatement'].loc[gev.Group == 'PTSD'].mean()]
evsems = [sem(gev['Context Reinstatement'].loc[gev.Group == 'Control']),sem(gev['Context Reinstatement'].loc[gev.Group == 'PTSD'])]

sns.set_context('poster')
sns.set_style('whitegrid',{'axes.facecolor':'.9','figure.facecolor':'.9'})
fig, ax99 = plt.subplots()	
ax99 = sns.swarmplot(x='Group',y='Context Reinstatement',data=gev, size=10, alpha=.9,
	palette=[wes_palettes['Chevalier'][0],wes_palettes['Royal1'][1]])

sns.pointplot(x='Group',y='Context Reinstatement',data=gev,
				join=False,capsize=.1,palette=['black','black'])
# ax99.set_ylim(0,1)
plt.title('Early CS+ Context Reinstatement')

tres = ttest_ind(gev['Context Reinstatement'].loc[np.where(gev.Group == 'Control')[0]], gev['Context Reinstatement'].loc[np.where(gev.Group == 'PTSD')[0]])
print(tres)
# fig,ax69 = plt.subplots()
# ax69 = sns.boxplot(x='Group',y='Context Reinstatement',data=gev)

psc = psc.set_index(['roi'])
epsc = epsc.set_index(['roi'])
#####################################################
vm = psc.loc['mOFC_beta']
evm = epsc.loc['mOFC_beta']

vm.reset_index(inplace=True)
evm.reset_index(inplace=True)

v = pd.concat([vm,ev],axis=1)
# v = v.rename(columns={'early_CSp_CSm':'R β Parameter Estimate'})
v = v.rename(columns={'early_CSp_CSm':'β Parameter Estimate'})
em = pd.concat([evm,ev],axis=1)
em = evm.rename(columns={'CSp_CSm':'β Parameter Estimate'})

# b = pd.concat([v,em],axis=1)
gb = sns.lmplot(x='β Parameter Estimate', y='Context Reinstatement', data=em, col='Group', hue='Group',
				palette=[wes_palettes['Chevalier'][0],wes_palettes['Royal1'][1]],
				height=8, aspect=10/8)

# v = v.set_index(['sub'])
# v = v.drop(labels=9)
# v = v.reset_index()
gv = sns.lmplot(x='Context Reinstatement', y='β Parameter Estimate', data=v, col='Group', hue='Group',
				palette=[wes_palettes['Chevalier'][0],wes_palettes['Royal1'][1]],
				height=8, aspect=10/8)
cv = v.loc[np.where(v.Group == 'Control')[0]]
# cv = cv.drop(labels=8)
pv = v.loc[np.where(v.Group == 'PTSD')[0]]

# cc = sns.lmplot(x='Context Reinstatement',y='β Parameter Estimate', data=cv, hue='Group',palette=[wes_palettes['Chevalier'][0]],height=8,aspect=10/8)
# plt.title('Context reinstatement correlates with vmPFC activity')
# pc = sns.lmplot(x='Context Reinstatement',y='β Parameter Estimate', data=pv, hue='Group',palette=[wes_palettes['Royal1'][1]],height=8,aspect=10/8)
# plt.title('vmPFC')

cvlm = lm(cv['Context Reinstatement'],cv['β Parameter Estimate'])
pvlm = lm(pv['Context Reinstatement'],pv['β Parameter Estimate'])
print('control vmpfc = %s'%(cvlm.pvalue))
print('ptsd vmpfc = %s'%(pvlm.pvalue))
print(ttest_ind(cv['β Parameter Estimate'],pv['β Parameter Estimate']))
plt.figure(figsize=(8,8))
sw1 = sns.swarmplot(x='Group',y='β Parameter Estimate',hue='Group',data=v, size=10, alpha=.9,
 palette=[wes_palettes['Chevalier'][0],wes_palettes['Royal1'][1]])
sns.pointplot(x='Group',y='β Parameter Estimate',data=v, palette=['black','black'],capsize=.1)
sw1.set_ylim(-60,60)
plt.title('vmPFC')
sw1.legend_.remove()
plt.tight_layout()


#####################################################
amyg = psc.loc['amygdala_beta']
amyg.reset_index(inplace=True)

a = pd.concat([amyg,ev],axis=1)
a = a.rename(columns={'early_CSp_CSm':'β Parameter Estimate'})

ca = a.loc[np.where(a.Group == 'Control')[0]]
pa = a.loc[np.where(a.Group == 'PTSD')[0]]
ga = sns.lmplot(x='Context Reinstatement', y='β Parameter Estimate', data=a, col='Group', hue='Group',
				palette=[wes_palettes['Chevalier'][0],wes_palettes['Royal1'][1]],
				height=8, aspect=10/8)

# cc = sns.lmplot(x='Context Reinstatement',y='β Parameter Estimate', data=ca, hue='Group',palette=[wes_palettes['Chevalier'][0]],height=8,aspect=10/8)
# plt.title('Amygdala')
# pc = sns.lmplot(x='Context Reinstatement',y='β Parameter Estimate', data=pa, hue='Group',palette=[wes_palettes['Royal1'][1]],height=8,aspect=10/8)
# plt.title('Amygdala')


calm = lm(ca['Context Reinstatement'],ca['β Parameter Estimate'])
palm = lm(pa['Context Reinstatement'],pa['β Parameter Estimate'])
calm2 = lm(ca['Context Reinstatement'],ca['β Parameter Estimate'])
palm2 = lm(pa['Context Reinstatement'],pa['β Parameter Estimate'])
print('control amyg = %s'%(calm.pvalue))
print('ptsd amyg = %s'%(palm.pvalue))
ttest_ind(ca['β Parameter Estimate'],pa['β Parameter Estimate'])
plt.figure(figsize=(8,8))
sw1 = sns.swarmplot(x='Group',y='β Parameter Estimate',hue='Group',data=a, size=10, alpha=.9,
 palette=[wes_palettes['Chevalier'][0],wes_palettes['Royal1'][1]])
sns.pointplot(x='Group',y='β Parameter Estimate',data=a, palette=['black','black'],capsize=.1)
sw1.set_ylim(-60,60)
plt.title('Amygdala')
sw1.legend_.remove()
plt.tight_layout()


#####################################################
hipp = psc.loc['hippocampus_beta']
hipp.reset_index(inplace=True)

h = pd.concat([hipp,ev],axis=1)
h = h.rename(columns={'early_CSp_CSm':'β Parameter Estimate'})

ch = h.loc[np.where(h.Group == 'Control')[0]]
ph = h.loc[np.where(h.Group == 'PTSD')[0]]
gh = sns.lmplot(x='Context Reinstatement',y='β Parameter Estimate',data=h, col='Group', hue='Group',
				palette=[wes_palettes['Chevalier'][0],wes_palettes['Royal1'][1]],
				height=8, aspect=10/8)
# cc = sns.lmplot(x='Context Reinstatement',y='β Parameter Estimate', data=ch, hue='Group',palette=[wes_palettes['Chevalier'][0]],height=8,aspect=10/8)
# plt.title('Hippocampus')
# pc = sns.lmplot(x='Context Reinstatement',y='β Parameter Estimate', data=ph, hue='Group',palette=[wes_palettes['Royal1'][1]],height=8,aspect=10/8)
# plt.title('Hippocampus')

chlm = lm(ch['Context Reinstatement'],ch['β Parameter Estimate'])
phlm = lm(ph['Context Reinstatement'],ph['β Parameter Estimate'])
print('control hipp = %s'%(chlm.pvalue))
print('ptsd hipp = %s'%(phlm.pvalue))
ttest_ind(ch['β Parameter Estimate'],ph['β Parameter Estimate'])

plt.figure(figsize=(8,8))
sw1 = sns.swarmplot(x='Group',y='β Parameter Estimate',hue='Group',data=h, size=10, alpha=.9,
 palette=[wes_palettes['Chevalier'][0],wes_palettes['Royal1'][1]])
sns.pointplot(x='Group',y='β Parameter Estimate',data=h, palette=['black','black'],capsize=.1)
sw1.set_ylim(-60,60)
plt.title('Hippocampus')
sw1.legend_.remove()
plt.tight_layout()


#####################################################
roi = vmPFC_beta
fs = 300

c = group_rsa(subs=sub_args,save_dict=roi,fs=fs)
p = group_rsa(subs=p_sub_args,save_dict=roi,fs=fs)



for f in [True,False]:
	c.sub_stats(f=f)
	p.sub_stats(f=f)

	# print(ttest_ind(c.stats['baseline'],p.stats['baseline']))
	# print(ttest_ind(c.stats['fear_conditioning'],p.stats['fear_conditioning']))
	# print(ttest_ind(c.stats['extinction'],p.stats['extinction']))
	# print(ttest_ind(c.stats['renewal'],p.stats['renewal']))
	print(ttest_ind(c.stats['r_e'],p.stats['r_e']))
	print(ttest_ind(c.stats['r_f'],p.stats['r_f']))
# cdif = c.stats['extinction'] - c.stats['fear_conditioning']
# pdif = p.stats['extinction'] - p.stats['fear_conditioning']
# ttest_ind(cdif,pdif)

ex = pd.DataFrame(np.concatenate((c.stats['r_e'],p.stats['r_e'])),columns=['Mean Pattern Similarity'])
ex['Group'] = ''
ex['Group'].loc[0:24] = 'Control'
ex['Group'].loc[24:48] = 'PTSD'
sns.set_context('poster')
sns.set_style('whitegrid',{'axes.facecolor':'.9','figure.facecolor':'.9'})
plt.figure(figsize=(8,8))
sw1 = sns.swarmplot(x='Group',y='Mean Pattern Similarity',hue='Group',data=ex, size=10, alpha=.9,
 	palette=[wes_palettes['Chevalier'][0],wes_palettes['Royal1'][1]])
sns.pointplot(x='Group',y='Mean Pattern Similarity',data=ex, palette=['black','black'],capsize=.1)
sw1.set_ylim(-.015,.055)
plt.title('vmPFC')
sw1.legend_.remove()
plt.tight_layout()
# sw1.errorbar(x=[0.15,1.15],y=[c.stats['extinction'].mean(),p.stats['extinction'].mean()],yerr=[sem(c.stats['extinction']),sem(p.stats['extinction'])],
# 	fmt='o',color='#212121',markersize=8,capsize=6)

#####################################################################
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


N = 4
end_value = .25
decay = exponential(N,0,-(N-1)/np.log(end_value),False)
# decay = [1,.75,.5,.25]
# decay = [1,1,1,1]

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

ec = sns.lmplot(x='response', y='evidence', data=exp_corr, col='Group', hue='Group',
				palette=[wes_palettes['Chevalier'][0],wes_palettes['FantasticFox'][3]])

c_ec = lm(exp_corr['response'].loc[sub_args],exp_corr['evidence'].loc[sub_args])
p_ec = lm(exp_corr['response'].loc[p_sub_args],exp_corr['evidence'].loc[p_sub_args])
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

print(ttest_ind(exp_corr['evidence'][c_l], exp_corr['evidence'][c_h]))
print(ttest_ind(exp_corr['evidence'][p_l], exp_corr['evidence'][p_h]))

#######################################################################################
demo = pd.read_csv(os.path.join(data_dir,'Demographics_Survey','summary_stats.csv'))
demo['sub'] = 0
for i in demo.index: demo['sub'][i] = int(demo.Subject[i][-3:])
demo.set_index(['sub'],inplace=True)
# Sub110 does not have IUS survey data
# demo = demo.drop(110,'index')
#p_args = [sub for sub in p_sub_args if sub is not 110]
#ttest_ind(demo.IUS[sub_args], demo.IUS[p_args])

ev = pd.read_csv(os.path.join(data_dir,'graphing','signal_change','mvpa_ev.csv'))
ev.set_index(['subject'],inplace=True)
# ev.drop(110,'index',inplace=True)

ev = pd.concat([ev,demo],axis=1)

gs = sns.lmplot(x='ev',y='BAI',data=ev, col='Group', hue='Group',
				palette=[wes_palettes['Chevalier'][0],wes_palettes['Royal1'][1]],
				height=8, aspect=10/8)


###############################################################################################
ev = pd.read_csv(os.path.join(data_dir,'graphing','signal_change','mvpa_ev.csv'))
ev.set_index(['subject'],inplace=True)
scr = pd.read_csv(os.path.join(data_dir,'Group_SCR','extinction_recall','avg_4.csv'))
scr.set_index(['subject'],inplace=True)

scr['ev'] = ev.ev
gs = sns.lmplot(x='ev',y='scr',data=scr, col='group', hue='group',
				palette=[wes_palettes['Chevalier'][0],wes_palettes['Royal1'][1]],
				height=8, aspect=10/8)

gs = sns.lmplot(x='vm',y='scr',data=scr, col='group', hue='group',
				palette=[wes_palettes['Chevalier'][0],wes_palettes['Royal1'][1]],
				height=8, aspect=10/8)
###############################################################################################
for roi in ['mOFC','vmPFC','amygdala','hippocampus','dACC']:
	dat = psc.loc[]

vm.reset_index(inplace=True)

v = pd.concat([vm,ev],axis=1)
# v = v.rename(columns={'early_CSp_CSm':'β Parameter Estimate'})
v = v.rename(columns={'CSp_CSm':'β Parameter Estimate'})

v = v.set_index(['sub'])
# v = v.drop(labels=9)
v = v.reset_index()
gv = sns.lmplot(x='Context Reinstatement', y='β Parameter Estimate', data=v, col='Group', hue='Group',
				palette=[wes_palettes['Chevalier'][0],wes_palettes['Royal1'][1]],
				height=8, aspect=10/8)
cv = v.loc[np.where(v.Group == 'Control')[0]]
# cv = cv.drop(labels=8)
pv = v.loc[np.where(v.Group == 'PTSD')[0]]

# cc = sns.lmplot(x='Context Reinstatement',y='β Parameter Estimate', data=cv, hue='Group',palette=[wes_palettes['Chevalier'][0]],height=8,aspect=10/8)
# plt.title('Context reinstatement correlates with vmPFC activity')
# pc = sns.lmplot(x='Context Reinstatement',y='β Parameter Estimate', data=pv, hue='Group',palette=[wes_palettes['Royal1'][1]],height=8,aspect=10/8)
# plt.title('vmPFC')

cvlm = lm(cv['Context Reinstatement'],cv['β Parameter Estimate'])
pvlm = lm(pv['Context Reinstatement'],pv['β Parameter Estimate'])
print('control vmpfc = %s'%(cvlm.pvalue))
print('ptsd vmpfc = %s'%(pvlm.pvalue))
print(ttest_ind(cv['β Parameter Estimate'],pv['β Parameter Estimate']))
plt.figure(figsize=(8,8))
sw1 = sns.swarmplot(x='Group',y='β Parameter Estimate',hue='Group',data=v, size=10, alpha=.9,
 palette=[wes_palettes['Chevalier'][0],wes_palettes['Royal1'][1]])
sns.pointplot(x='Group',y='β Parameter Estimate',data=v, palette=['black','black'],capsize=.1)
sw1.set_ylim(-60,60)
plt.title('vmPFC')
sw1.legend_.remove()
plt.tight_layout()




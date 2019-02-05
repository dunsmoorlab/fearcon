from mvpa_analysis import group_decode, decode#, vis_cond_phase, get_bar_stats, phase_bar_plot, vis_event_res
from fc_config import data_dir
from scipy.stats import sem, ttest_ind
import numpy as np
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import pandas as pd
import seaborn as sns
from scipy.stats import sem
import statsmodels.api as sm
import matplotlib.pyplot as plt
from fc_config import *

sns.set_style('whitegrid')
sns.set_style('ticks')
sns.set_style(rc={'axes.linewidth':'2'})
plt.rcParams['xtick.labelsize'] = 18 
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['axes.labelsize'] = 22
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['legend.labelspacing'] = .25


imgs='tr'
nvox = 'all'
SC=True
S_DS=True
rmv_scram=True
verbose=True
split=False
con = 'CS+'
binarize=False
save_dict=ppa_prepped


res = group_decode(imgs=imgs, k=nvox, save_dict=save_dict, SC=SC, S_DS=S_DS, rmv_scram=rmv_scram, verbose=verbose)
pres = group_decode(imgs=imgs, k=nvox, save_dict=save_dict, SC=SC, S_DS=S_DS, rmv_scram=rmv_scram, p=True, verbose=verbose)

res.exp_event(con=con,split=split)
res.vis_exp_event()
res.exp_stats(vis=False)
if con == 'CS+': res.more_exp_stats()
res.exp_stats(vis=False)
for trial in [1]:
	res.exp_stats(trial_=trial,vis=False)
	print(res.pair_df)
	print(res.ind_df)

pres.exp_event(con='CS+',split=split)
pres.vis_exp_event()
pres.exp_stats(vis=False)
if con == 'CS+': pres.more_exp_stats()
pres.exp_stats(vis=False)
for trial in [1]:
	pres.exp_stats(trial_=trial,vis=False)
	print(pres.pair_df)
	print(pres.ind_df)


for trial in [1]:
	df = pd.DataFrame([], columns=['group','response','baseline'])
	n_con = res._no_[trial].shape[0] + res._expect_[trial].shape[0]
	n_p = pres._no_[trial].shape[0] + pres._expect_[trial].shape[0]

	df['group']	= np.concatenate((np.repeat(['control'],n_con), np.repeat(['ptsd'],n_p)))
	df['response']	= np.concatenate((np.repeat(['no'],res._no_[trial].shape[0]), np.repeat(['expect'],res._expect_[trial].shape[0]),
									np.repeat(['no'],pres._no_[trial].shape[0]), np.repeat(['expect'],pres._expect_[trial].shape[0])))

	df['baseline'] = np.concatenate((res._no_[trial],res._expect_[trial],pres._no_[trial],pres._expect_[trial]))
	df['outcome'] = 0
	df.outcome.loc[np.where(df.response == 'no')[0]] = 1
	# aov(cr ~ (condition * phase * group) + Error(subject/(condition*phase)), data=cr_dat)
	formula = 'baseline ~ C(group) + C(response) + C(group):C(response)'
	model = ols(formula, df).fit()
	aov_table = anova_lm(model, typ=2)
	print(trial,aov_table)


	control = np.concatenate((res._no_[trial],res._expect_[trial]))
	ptsd = np.concatenate((pres._no_[trial],pres._expect_[trial]))
	print(ttest_ind(control,ptsd))

	print(trial,'c_no v c_exp', ttest_ind(res._no_[trial], res._expect_[trial]))
	print(trial,'c_no v p_no', ttest_ind(res._no_[trial], pres._no_[trial]))
	print(trial,'c_no v p_exp', ttest_ind(res._no_[trial], pres._expect_[trial]))

	print(trial,res.pair_df)
	print(trial,pres.pair_df)


	fig, ax1 = plt.subplots()
	ind2 = np.arange(2)    # the x locations for the groups
	width = 0.4         # the width of the bars
	sns.set_style('whitegrid')
	sns.set_style('ticks')

	no = [res._no_[trial].mean(),pres._no_[trial].mean()]
	no_err = [sem(res._no_[trial]), sem(pres._no_[trial])]

	yes = [res._expect_[trial].mean(), pres._expect_[trial].mean()]
	yes_err = [sem(res._expect_[trial]), sem(pres._expect_[trial])]

	ind2 = np.array((.45,.55))       # the x locations for the groups
	width = 0.01         # the width of the bars
	hw = width/2
	fig, ax1 = plt.subplots()
	p1 = ax1.errorbar(ind2, no, yerr=no_err, marker='o',
						 color = 'seagreen', markersize=10, linewidth=3)
	p2 = ax1.errorbar(ind2 + width, yes, yerr=yes_err, marker='o',
						 color='maroon', markersize=10, linewidth=3,linestyle='--')
	# ind2 + (width*.25)
	legend = ax1.legend((p1[0], p2[0]), ('No', 'Yes'),fontsize='xx-large',title='Expect a shock?',loc='upper right')
	legend.get_title().set_fontsize('18') #legend 'Title' fontsize
	ax1.set_ylim([-.4,1])
	ax1.set_xticks(ind2 + width / 2)
	ax1.set_xticklabels(('Control','PTSD'))
	ax1.set_title('Differences in Relative Scene Evidence')
	ax1.set_ylabel('Relateive Evidence (Scene-Rest)')
	ax1.set_xlabel('Group')
	ax1.set_xlim(xmin=.425,xmax=.585)

	fig.set_size_inches(8, 6.5)





tr = 4
c_no, c_yes = res.no_df.loc[:,0:tr].mean(axis=1), res.exp_df.loc[:,0:tr].mean(axis=1)
p_no, p_yes = pres.no_df.loc[:,0:tr].mean(axis=1), pres.exp_df.loc[:,0:tr].mean(axis=1)
df2 = pd.DataFrame([], columns=['group','response','evidence'])
n_con2 = c_no.shape[0] + c_yes.shape[0]
n_p = p_no.shape[0] + p_yes.shape[0]

df2['group']	= np.concatenate((np.repeat(['control'],n_con), np.repeat(['ptsd'],n_p)))
df2['response']	= np.concatenate((np.repeat(['no'],c_no.shape[0]), np.repeat(['expect'], c_yes.shape[0]),
								np.repeat(['no'],p_no.shape[0]), np.repeat(['expect'], p_yes.shape[0])))

df2['evidence'] = np.concatenate((c_no, c_yes, p_no, p_yes))
df2['outcome'] = 0
df2.outcome.loc[np.where(df.response == 'no')[0]] = 1

formula2 = 'evidence ~ C(group) + C(response) + C(group):C(response)'
model2 = ols(formula2, df2).fit()
aov_table2 = anova_lm(model2, typ=2)
print(aov_table2)

print('c_no v c_exp TR', ttest_ind(c_no, c_yes))
print('c_no v p_no TR', ttest_ind(c_no, p_no))
print('c_no v p_exp TR', ttest_ind(c_no, p_yes))

no = [c_no.mean(), p_no.mean()]
no_err = [c_no.sem(), p_no.sem()]
yes = [c_yes.mean(), p_yes.mean()]
yes_err = [c_yes.sem(), p_yes.sem()]

ind2 = np.array((.45,.55))       # the x locations for the groups
width = 0.01         # the width of the bars
hw = width/2
fig, ax1 = plt.subplots()
p1 = ax1.errorbar(ind2, no, yerr=no_err, marker='o',
					 color = 'seagreen', markersize=10, linewidth=3)
p2 = ax1.errorbar(ind2 + width, yes, yerr=yes_err, marker='o',
					 color='maroon', markersize=10, linewidth=3,linestyle='--')
# ind2 + (width*.25)
legend = ax1.legend((p1[0], p2[0]), ('No', 'Yes'),fontsize='xx-large',title='Expect a shock?',loc='upper right')
legend.get_title().set_fontsize('18') #legend 'Title' fontsize
ax1.set_ylim([0,1])
ax1.set_xticks(ind2 + width / 2)
ax1.set_xticklabels(('Control','PTSD'))
ax1.set_title('Differences in Initial Scene Evidence')
ax1.set_ylabel('Scene Evidence')
ax1.set_xlabel('Group')
ax1.set_xlim(xmin=.425,xmax=.585)

fig.set_size_inches(8, 6.5)
plt.tight_layout()

sns.set_style('whitegrid')
sns.set_style('ticks')
sns.set_style(rc={'axes.linewidth':'2'})
plt.rcParams['xtick.labelsize'] = 10 
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.labelspacing'] = .25

trials = [1,2]
trial = 0
nrows = len(trials)

xaxis_tr = [-2,-1,0,1,2,3]

fig, ax = plt.subplots(nrows,2, sharex='col',sharey='row')
# for cond, color in zip(['csplus','scene','rest'], [plt.cm.Set1.colors[0],'seagreen',plt.cm.Set1.colors[-1]]):	
for cond, color in zip(['scene','rest'], ['seagreen',plt.cm.Set1.colors[-1]]):	
	if cond == 'csplus':
		labelcond = 'CS+'
	else:
		labelcond = cond

	ax[0][0].plot(xaxis_tr, res.err_['ev'].loc['no'].loc[trials[trial]].loc[cond], color=color, marker='o', markersize=5, label=labelcond)
	
	ax[0][0].fill_between(xaxis_tr, res.err_['ev'].loc['no'].loc[trials[trial]].loc[cond] - res.err_['err'].loc['no'].loc[trials[trial]].loc[cond],
	 res.err_['ev'].loc['no'].loc[trials[trial]].loc[cond] + res.err_['err'].loc['no'].loc[trials[trial]].loc[cond],
	 alpha=.5, color=color)

	ax[0][1].plot(xaxis_tr, res.err_['ev'].loc['expect'].loc[trials[trial]].loc[cond], label='%s'%(labelcond), color=color, marker='o', markersize=5)
	
	ax[0][1].fill_between(xaxis_tr, res.err_['ev'].loc['expect'].loc[trials[trial]].loc[cond] - res.err_['err'].loc['expect'].loc[trials[trial]].loc[cond],
	 res.err_['ev'].loc['expect'].loc[trials[trial]].loc[cond] + res.err_['err'].loc['expect'].loc[trials[trial]].loc[cond],
	 alpha=.5, color=color)
	
	ax[0][0].set_title('Control; Did not expect a shock (N=%s)'%(res.nexp['no'][trials[trial]]))
	ax[0][1].set_title('Control; Expected a shock (N=%s)'%(res.nexp['expect'][trials[trial]]))

	#ptsd
	ax[1][0].plot(xaxis_tr, pres.err_['ev'].loc['no'].loc[trials[trial]].loc[cond], color=color, marker='o', markersize=5, label=labelcond)
	
	ax[1][0].fill_between(xaxis_tr, pres.err_['ev'].loc['no'].loc[trials[trial]].loc[cond] - pres.err_['err'].loc['no'].loc[trials[trial]].loc[cond],
	 pres.err_['ev'].loc['no'].loc[trials[trial]].loc[cond] + pres.err_['err'].loc['no'].loc[trials[trial]].loc[cond],
	 alpha=.5, color=color)

	ax[1][1].plot(xaxis_tr, pres.err_['ev'].loc['expect'].loc[trials[trial]].loc[cond], label='%s'%(labelcond), color=color, marker='o', markersize=5)
	
	ax[1][1].fill_between(xaxis_tr, pres.err_['ev'].loc['expect'].loc[trials[trial]].loc[cond] - pres.err_['err'].loc['expect'].loc[trials[trial]].loc[cond],
	 pres.err_['ev'].loc['expect'].loc[trials[trial]].loc[cond] + pres.err_['err'].loc['expect'].loc[trials[trial]].loc[cond],
	 alpha=.5, color=color)
	
	ax[1][0].set_title('PTSD; Did not expect a shock (N=%s)'%(pres.nexp['no'][trials[trial]]))
	ax[1][1].set_title('PTSD; Expected a shock (N=%s)'%(pres.nexp['expect'][trials[trial]]))

ax[1][0].set_ylim([0,1])
ax[1][0].set_xlabel('TR (away from stimulus onset)')
ax[1][1].set_xlabel('TR (away from stimulus onset)')

ax[0][0].set_ylabel('Classifier Evidence')
ax[1][0].set_ylabel('Classifier Evidence')

fig.set_size_inches(10, 6)
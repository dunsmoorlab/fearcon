from mvpa_analysis_old import *
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

# sns.set_style('whitegrid')
# sns.set_style('ticks')
# sns.set_style(rc={'axes.linewidth':'2'})
# plt.rcParams['xtick.labelsize'] = 18 
# plt.rcParams['ytick.labelsize'] = 16
# plt.rcParams['axes.labelsize'] = 22
# plt.rcParams['axes.titlesize'] = 24
# plt.rcParams['legend.labelspacing'] = .25



imgs='tr'
nvox = 'all'
SC=True
S_DS=True
rmv_scram=True
rmv_ind=False
rmv_rest=False
verbose=False
split=False
con = 'CS+'
binarize=False
save_dict=ppa_prepped


res = group_decode(imgs=imgs, k=nvox, save_dict=save_dict, rmv_rest=True, binarize=binarize, SC=SC, S_DS=S_DS, rmv_scram=rmv_scram, rmv_ind=rmv_ind, verbose=verbose)
pres = group_decode(imgs=imgs, k=nvox, save_dict=save_dict, rmv_rest=True, binarize=binarize, SC=SC, S_DS=S_DS, rmv_scram=rmv_scram, rmv_ind=rmv_ind, p=True, verbose=verbose)

collect_ev(res=res,pres=pres)

res.ev_out()
pres.ev_out()

res.tavg['group'] = 'control'
pres.tavg['group'] = 'ptsd'
idx=pd.IndexSlice
res.tavg.evidence = res.tavg.evidence.astype('float')
pres.tavg.evidence = pres.tavg.evidence.astype('float')
c = res.tavg.loc[idx['CS+',1:8,:],:].groupby('subject').mean()
p = pres.tavg.loc[idx['CS+',1:8,:],:].groupby('subject').mean()
ttest_ind(c,p)

cbic = [i for k,i in res.bic.items()]
pbic = [i for k,i in pres.bic.items()]

res.tavg.reset_index(inplace=True)
pres.tavg.reset_index(inplace=True)
out = pd.concat([res.tavg,pres.tavg])


# out.to_csv(os.path.join(data_dir,'Group_GLM','beta_pm_mvpa_ev.csv'),sep=',',index=False)

# out.to_csv(os.path.join(data_dir,'Group_GLM','tr-1_0_pm_mvpa_ev.csv'),sep=',',index=False)
# out.to_csv(os.path.join(data_dir,'Group_GLM','tr_0_pm_mvpa_ev.csv'),sep=',',index=False)
# out.to_csv(os.path.join(data_dir,'Group_GLM','tr_0-1_pm_mvpa_ev.csv'),sep=',',index=False)
# out.to_csv(os.path.join(data_dir,'Group_GLM','tr_0_relative_pm_mvpa_ev.csv'),sep=',',index=False)


#{res.vis_cond_phase(phase=phase, title='ppa_roi') for phase in decode.decode_runs}
#{res.vis_event_res(phase, title='PPA roi') for phase in decode.decode_runs}
res.exp_event(con=con,split=split)
res.vis_exp_event()
res.exp_stats(vis=False)
if con == 'CS+': res.more_exp_stats()
res.exp_stats(vis=False)
for trial in [1]:
	res.exp_stats(trial_=trial)
	print(res.pair_df)
	print(res.ind_df)

# {res.vis_cond_phase(phase=phase, title='new_beta_sc_ds') for phase in decode.decode_runs}
# {res.vis_event_res(phase, title='Train on NEW Betas, scene_ds, k=1000') for phase in decode.decode_runs}
pres.exp_event(con='CS+',split=split)
pres.vis_exp_event()
pres.exp_stats(vis=False)
if con == 'CS+': pres.more_exp_stats()
pres.exp_stats(vis=False)
for trial in [1]:
	pres.exp_stats(trial_=trial)
	print(pres.pair_df)
	print(pres.ind_df)

for trial in [1]:
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


	sns.set_context('poster')
	sns.set_style('whitegrid',{'axes.facecolor':'.9','figure.facecolor':'.9'})	
	ax55 = sns.violinplot(x='Group',y='Relative Context Reinstatement',hue='Response',data=df,height=6.5,aspect=8/6.5,
							cut=0,split=False,scale='count',inner='box',saturation=.75,dodge=True,
							palette=[wes_palettes['Royal1'][1],wes_palettes['Royal1'][0]])
	plt.title('Renewal Test: 1st CS+ Trial')
	# plt.legend(handles=legend_elements,title='Do you expect a shock?',loc='upper center')
	ax55.set_ylim(-1,1.5)
	# ax55.errorbar(x=[-.125,.875],y=no,yerr=no_err,fmt='o',color='#212121',markersize=8,capsize=6)
	# ax55.errorbar(x=[.125,1.125],y=yes,yerr=yes_err,fmt='o',color='#212121',markersize=8,capsize=6)
	#plt.plot(yes)
	sns.stripplot(x='Group',y='Relative Context Reinstatement',hue='Response',data=df,
		palette=['#f2f2f2','#f2f2f2'],dodge=.2,jitter=.15,alpha=.75,size=6)
	ax55.legend_.remove()

	legend_elements = [Patch(facecolor='#b45c3e',edgecolor='#535353',label='Do not expect'),
						Patch(facecolor=wes_palettes['Royal1'][0],edgecolor='#535353',label='Expect')]
	plt.legend(handles=legend_elements,title='Do you expect a shock?',loc='center left',bbox_to_anchor=(1,.5))
	plt.subplots_adjust(right=0.7)
	

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

	print(trial,res.pair_df)
	print(trial,pres.pair_df)


	
	ind2 = np.arange(2)    # the x locations for the groups
	width = 0.4         # the width of the bars
	sns.set_style('whitegrid')
	sns.set_style('ticks')



	#ind2 = np.array((.45,.55))       # the x locations for the groups
	#width = 0.01         # the width of the bars
	hw = width/2
	fig, ax1 = plt.subplots()
	p1 = ax1.bar(ind2, no, width, yerr=no_err, color='seagreen',error_kw=dict(lw=2, capsize=5, capthick=2))
	p2 = ax1.bar(ind2 + width, yes, width, yerr=yes_err, color='indianred',error_kw=dict(lw=2, capsize=5, capthick=2))
	

	#p1 = ax1.errorbar(ind2, no, yerr=no_err, marker='o',
	#					 color = 'seagreen', markersize=10, linewidth=3)
	#p2 = ax1.errorbar(ind2 + width, yes, yerr=yes_err, marker='o',
	#					 color='maroon', markersize=10, linewidth=3,linestyle='--')
	#ind2 + (width*.25)
	legend = ax1.legend((p1[0], p2[0]), ('No', 'Yes'),fontsize='xx-large',title='Expect a shock?',loc='upper right')
	legend.get_title().set_fontsize('18') #legend 'Title' fontsize
	ax1.set_ylim([-.4,1])
	ax1.set_xticks(ind2 + width / 2)
	ax1.set_xticklabels(('Control','PTSD'))
	ax1.set_title('Differences in Relative Scene Evidence')
	ax1.set_ylabel('Relateive Evidence (Scene-Rest)')
	ax1.set_xlabel('Group')
	#ax1.set_xlim(xmin=.425,xmax=.585)

	fig.set_size_inches(8, 6.5)
	fig.savefig(os.path.join(data_dir,'graphing','cnirc','trial_1_group_relative.png'), dpi=300)
def old_stuff():
	tr = 4
	c_no, c_yes = res.no_df.loc[:,0:tr].mean(axis=1), res.exp_df.loc[:,0:tr].mean(axis=1)
	p_no, p_yes = pres.no_df.loc[:,0:tr].mean(axis=1), pres.exp_df.loc[:,0:tr].mean(axis=1)
	c = np.concatenate((c_no,c_yes))
	p = np.concatenate((p_no,p_yes))
	# print(list(np.repeat('removed 2 subs from control',20)))

	df2 = pd.DataFrame([], columns=['Group','Response','Context Reinstatement'])
	n_con2 = c_no.shape[0] + c_yes.shape[0]
	n_p = p_no.shape[0] + p_yes.shape[0]

	df2['Group']	= np.concatenate((np.repeat(['Control'],c_no.shape[0] + c_yes.shape[0]), np.repeat(['PTSD'],n_p)))
	df2['Response']	= np.concatenate((np.repeat(['Do not expect'],c_no.shape[0]), np.repeat(['Expect'], c_yes.shape[0]),
									np.repeat(['Do not expect'],p_no.shape[0]), np.repeat(['Expect'], p_yes.shape[0])))

	df2['Context Reinstatement'] = np.concatenate((c_no, c_yes, p_no, p_yes))
	df2['outcome'] = 0
	df2.outcome.loc[np.where(df2.Response == 'no')[0]] = 1

	df2 = df2.rename(columns={'Context Reinstatement':'evidence'})
	formula2 = 'evidence ~ C(Group) + C(Response) + C(Group):C(Response)'
	model2 = ols(formula2, df2).fit()
	aov_table2 = anova_lm(model2, typ=2)
	print(aov_table2)



	print('c_no v c_exp TR', ttest_ind(c_no, c_yes))
	print('c_no v p_no TR', ttest_ind(c_no, p_no))
	print('c_no v p_exp TR', ttest_ind(c_no, p_yes))


	# fig, ax1 = plt.subplots()
	# ind2 = [0,.5]    # the x locations for the groups
	# width = 0.1         # the width of the bars
	# sns.set_style('whitegrid')
	# sns.set_style('ticks')
	# p1 = ax1.bar(ind2[0], c_no.mean(), width, yerr=c_no.sem(),
	# 	color='mediumblue', alpha=.7)
	# p2 = ax1.bar(ind2[0]+width, c_yes.mean(), width, yerr=c_yes.sem(),
	# 	edgecolor='maroon', color='w', linewidth=3)
	# p3 = ax1.bar(ind2[1], p_no.mean(), width, yerr=p_no.sem(),
	# 	color='mediumblue', alpha=.7)
	# p4 = ax1.bar(ind2[1]+width, p_yes.mean(), width, yerr=p_yes.sem(),
	# 	edgecolor='maroon', color='w', linewidth=3)

	no = [c_no.mean(), p_no.mean()]
	no_err = [c_no.sem(), p_no.sem()]
	yes = [c_yes.mean(), p_yes.mean()]
	yes_err = [c_yes.sem(), p_yes.sem()]

	# ind2 = np.array((.45,.55))       # the x locations for the groups
	# width = 0.01         # the width of the bars
	# hw = width/2
	fig, ax1 = plt.subplots()
	p1 = ax1.bar(ind2, no, width, yerr=no_err, color='seagreen',error_kw=dict(lw=2, capsize=5, capthick=2))
	p2 = ax1.bar(ind2 + width, yes, width, yerr=yes_err, color='indianred',error_kw=dict(lw=2, capsize=5, capthick=2))

	# p1 = ax1.errorbar(ind2, no, yerr=no_err, marker='o',
	# 					 color = 'seagreen', markersize=10, linewidth=3)
	# p2 = ax1.errorbar(ind2 + width, yes, yerr=yes_err, marker='o',
	# 					 color='maroon', markersize=10, linewidth=3,linestyle='--')
	# ind2 + (width*.25)
	legend = ax1.legend((p1[0], p2[0]), ('No', 'Yes'),fontsize='xx-large',title='Expect a shock?',loc='upper right')
	legend.get_title().set_fontsize('18') #legend 'Title' fontsize
	ax1.set_ylim([0,1])
	ax1.set_xticks(ind2 + width / 2)
	ax1.set_xticklabels(('Control','PTSD'))
	ax1.set_title('Differences in Initial Scene Evidence')
	ax1.set_ylabel('Scene Evidence')
	ax1.set_xlabel('Group')
	# ax1.set_xlim(xmin=.425,xmax=.585)

	fig.set_size_inches(8, 6.5)
	plt.tight_layout()
	# fig.savefig('C:\\Users\\ACH\\Desktop\\group_scene.png', dpi=300)
	fig.savefig(os.path.join(data_dir,'graphing','cnirc','group_4tr_ev.png'), dpi=300)





	sns.set_style('whitegrid')
	sns.set_style('ticks')
	sns.set_style(rc={'axes.linewidth':'2'})
	plt.rcParams['xtick.labelsize'] = 14 
	plt.rcParams['ytick.labelsize'] = 14
	plt.rcParams['axes.labelsize'] = 16
	plt.rcParams['axes.titlesize'] = 16
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


	# ax[0][1].legend(loc='upper center', bbox_to_anchor=(0.5, 1), frameon=True, ncol=3,borderaxespad=1)#,prop={'size': 8})
	# ax[0][1].legend(bbox_to_anchor=(0, .02, 1., 1), frameon=True, loc=3, ncol=3,borderaxespad=0,prop={'size': 8})
	# ax[0][0].legend(loc='upper center', bbox_to_anchor=(0.5, 1), frameon=True, ncol=3,borderaxespad=1)#,prop={'size': 8})
	fig.savefig(os.path.join(data_dir,'graphing','cnirc','1st_csp_trial_combined.png'), dpi=500)


ev = pd.read_csv(os.path.join(data_dir,'graphing','signal_change','mvpa_ev.csv'))
# ev = pd.read_csv(os.path.join(data_dir,'graphing','signal_change','mvpa_rel_ev.csv'))
ev = ev.rename(columns={'ev':'Context Reinstatement'})
psc = pd.read_csv(os.path.join(data_dir,'graphing','signal_change','psc_values.csv'))

gev = ev.copy()
gev = gev.rename(columns={'ev':'Context Reinstatement'})

sns.set_context('poster')
sns.set_style('whitegrid',{'axes.facecolor':'.9','figure.facecolor':'.9'})
fig, ax99 = plt.subplots()	
ax99 = sns.boxplot(x='Group',y='Context Reinstatement',data=gev,
	palette=[wes_palettes['Chevalier'][0],wes_palettes['FantasticFox'][3]])

ax99.set_ylim(0,1)
sns.stripplot(x='Group',y='Context Reinstatement',data=gev,
	palette=['red','red'],dodge=.2,jitter=.05,alpha=.5,size=6)
plt.title('Initial Context Reinstatement')

tres = ttest_ind(gev['Context Reinstatement'].loc[np.where(gev.Group == 'Control')[0]], gev['Context Reinstatement'].loc[np.where(gev.Group == 'PTSD')[0]])
# fig,ax69 = plt.subplots()
# ax69 = sns.boxplot(x='Group',y='Context Reinstatement',data=gev)

psc = psc.set_index(['roi'])
vm = psc.loc['vmPFC_beta']
vm.reset_index(inplace=True)

v = pd.concat([vm,ev],axis=1)
v = v.rename(columns={'early_CSp_cope':'% Signal Change'})

# v = v.set_index(['sub'])
# v = v.drop(labels=3)
# v = v.reset_index()

gv = sns.lmplot(x='Context Reinstatement', y='% Signal Change', data=v, col='Group', hue='Group',
				palette=[wes_palettes['Chevalier'][0],wes_palettes['FantasticFox'][3]])
# plt.legend(title='Do you expect a shock?',loc='lower center')
# plt.title('Renewal Test: 1st CS+ Trial')
cv = v.loc[np.where(v.Group == 'Control')[0]]
cv = cv.drop(labels=8)
pv = v.loc[np.where(v.Group == 'PTSD')[0]]

cvlm = lm(cv['Context Reinstatement'],cv['% Signal Change'])
pvlm = lm(pv['Context Reinstatement'],pv['% Signal Change'])
print('control vmpfc = %s'%(cvlm.pvalue))
print('ptsd vmpfc = %s'%(pvlm.pvalue))
ttest_ind(cv['% Signal Change'],pv['% Signal Change'])
# # sns.set_style('whitegrid')
# # sns.set_style('ticks')
# # sns.set_style(rc={'axes.linewidth':'2'})
# # plt.rcParams['xtick.labelsize'] = 18 
# # plt.rcParams['ytick.labelsize'] = 16
# # plt.rcParams['axes.labelsize'] = 22
# # plt.rcParams['axes.titlesize'] = 24
# # plt.rcParams['legend.labelspacing'] = .25

# #implement factor plot to get the colors the same as the violin plot

# fig,ax3 = plt.subplots()
# ax3 = sns.lmplot(x='ev',y='early_CSp_pe',data=cv, height=6.5,aspect=8/6.5, palette=["#3B9AB2" "#78B7C5"])
# plt.text(.72,-1.75,'r = %.3f;  p = %.3f'%(cvlm.rvalue,cvlm.pvalue), fontsize=18)
# ax3.set(title='Controls; vmPFC',xlabel='Initial Scene Evidence',ylabel='% Signal Change')
# ax3.savefig(os.path.join(data_dir,'graphing','cnirc','c_vmPFC_corr.png'), dpi=300)

# fig,ax4 = plt.subplots()
# ax4 = sns.lmplot(x='ev',y='early_CSp_pe',data=pv, height=6.5,aspect=8/6.5, palette="#E1AF00")
# plt.text(.2,-1,'r = %.3f;  p = %.3f'%(pvlm.rvalue,pvlm.pvalue), fontsize=18)
# ax4.set(title='PTSD; vmPFC',xlabel='Initial Scene Evidence',ylabel='% Signal Change')
# ax4.savefig(os.path.join(data_dir,'graphing','cnirc','p_vmPFC_corr.png'), dpi=300)
######################################################################################################
#for amygdala
ev = pd.read_csv(os.path.join(data_dir,'graphing','signal_change','mvpa_ev.csv'))
ev = ev.rename(columns={'ev':'Context Reinstatement'})
psc = pd.read_csv(os.path.join(data_dir,'graphing','signal_change','beta_values.csv'))

psc = psc.set_index(['roi'])
amyg = psc.loc['amygdala_beta']
amyg.reset_index(inplace=True)


a = pd.concat([amyg,ev],axis=1)
a = a.rename(columns={'early_CSp_cope':'% Signal Change'})

ca = a.loc[np.where(a.Group == 'Control')[0]]
pa = a.loc[np.where(a.Group == 'PTSD')[0]]
ga = sns.lmplot(x='Context Reinstatement', y='% Signal Change', data=a, col='Group', hue='Group',
				palette=[wes_palettes['Chevalier'][0],wes_palettes['FantasticFox'][3]])
calm = lm(ca['Context Reinstatement'],ca['% Signal Change'])
palm = lm(pa['Context Reinstatement'],pa['% Signal Change'])
print('control amyg = %s'%(calm.pvalue))
print('ptsd amyg = %s'%(palm.pvalue))
ttest_ind(ca['% Signal Change'],pa['% Signal Change'])

# ax5 = sns.lmplot(x='ev',y='early_CSp_pe',data=ca, height=6.5,aspect=8/6.5)
# plt.text(.72,-.6,'r = %.3f;  p = %.3f'%(calm.rvalue,calm.pvalue), fontsize=18)
# ax5.set(title='Controls; Amygdala',xlabel='Initial Scene Evidence',ylabel='% Signal Change')
# ax5.savefig(os.path.join(data_dir,'graphing','cnirc','c_amyg_corr.png'), dpi=300)

# ax6 = sns.lmplot(x='ev',y='early_CSp_pe',data=pa, height=6.5,aspect=8/6.5)
# plt.text(.2,-.4,'r = %.3f;  p = %.3f'%(palm.rvalue,palm.pvalue), fontsize=18)
# ax6.set(title='PTSD; Amygdala',xlabel='Initial Scene Evidence',ylabel='% Signal Change')
# ax6.savefig(os.path.join(data_dir,'graphing','cnirc','p_amyg_corr.png'), dpi=300)


'''









res.exp_stats(trial_=2,vis=False)
print(res.pair_df)
pres.exp_stats(trial_=2,vis=False)
print(pres.pair_df)


# for tr in range(1,20):
# 	print(tr)
# 	print(ttest_ind(res.group_stats['extinction_recall']['csplus']['ev'][:tr],
# 			pres.group_stats['extinction_recall']['csplus']['ev'][:tr]))

con = np.stack(res.group_df['scene'].loc['extinction_recall'].values)
_ptsd = np.stack(pres.group_df['scene'].loc['extinction_recall'].values)

c_rest = np.stack(res.group_df['rest'].loc['extinction_recall'].values)
p_rest = np.stack(pres.group_df['rest'].loc['extinction_recall'].values)

for tr in range(1,135):
	print(tr)

	c = np.concatenate(con[:,:tr])
	p = np.concatenate(_ptsd[:,:tr])

	c_r = np.concatenate(c_rest[:,:tr])
	p_r = np.concatenate(p_rest[:,:tr])

	_c = c - c_r
	_p = p - p_r
	
	print(ttest_ind(_c,_p))







# df_con = df.loc[np.where(df.group=='control')[0]]
# poisson_mod = sm.Poisson(df_con.outcome, df_con.outcome)
# poisson_res = poisson_mod.fit(method="newton")
# print(poisson_res.summary())

# df_p = df.loc[np.where(df.group=='ptsd')[0]]
# poisson_mod = sm.Poisson(df_p.outcome, df_p.outcome)
# poisson_res = poisson_mod.fit(method="newton")
# print(poisson_res.summary())



con = {1:{}, 2:{}}
ptsd = {1:{}, 2:{}}

for trial in [1,2]:
	res.exp_stats(trial_=trial)
	pres.exp_stats(trial_=trial)
	
	# con[trial] = np.concatenate((res.ev_baseline['no'][0], res.ev_baseline['expect'][0],
	# 					res.ev_baseline['no'][1], res.ev_baseline['expect'][1]))
	# ptsd[trial] = np.concatenate((pres.ev_baseline['no'][0], pres.ev_baseline['expect'][0],
	# 					pres.ev_baseline['no'][1], pres.ev_baseline['expect'][1]))

	con[trial] = np.concatenate((res.stat_map['no'][1]['scene'], res.stat_map['expect'][1]['scene']))
	ptsd[trial] = np.concatenate((pres.stat_map['no'][1]['scene'], pres.stat_map['expect'][1]['scene']))
	
	print('control no to control expect', ttest_ind(res.ev_baseline['no'][0][0],res.ev_baseline['expect'][0][0]))
	print('control no to ptsd no', ttest_ind(res.ev_baseline['no'][0][0],pres.ev_baseline['no'][0][0]))
	print('control no to ptsd expect', ttest_ind(res.ev_baseline['no'][0][0],pres.ev_baseline['expect'][0][0]))

	print('/n')

	print('control no to control expect', ttest_ind(res.stat_map['no'][0]['scene'],res.stat_map['expect'][0]['scene']))
	print('control no to ptsd no', ttest_ind(res.stat_map['no'][0]['scene'],pres.stat_map['no'][0]['scene']))
	print('control no to ptsd expect', ttest_ind(res.stat_map['no'][0]['scene'],pres.stat_map['expect'][0]['scene']))
	
	print('/n')


	# print(ttest_ind(res.stat_map['no'][1]['scene'],pres.stat_map['no'][1]['scene']))
res.exp_stats(trial_=1)
pres.exp_stats(trial_=1)




res.stat_df.reset_index(inplace=True)
res.stat_df.set_index(['response','condition','trial'],inplace=True)
pres.stat_df.reset_index(inplace=True)
pres.stat_df.set_index(['response','condition','trial'],inplace=True)

for trial in [1,2]:
	con_no = res.stat_df['evidence'].loc['no'].loc['scene'].loc[trial] - res.stat_df['evidence'].loc['no'].loc['rest'].loc[trial]
	con_yes = res.stat_df['evidence'].loc['expect'].loc['scene'].loc[trial] - res.stat_df['evidence'].loc['expect'].loc['rest'].loc[trial]
	p_no = pres.stat_df['evidence'].loc['no'].loc['scene'].loc[trial] - pres.stat_df['evidence'].loc['no'].loc['rest'].loc[trial]
	p_yes = pres.stat_df['evidence'].loc['expect'].loc['scene'].loc[trial] - pres.stat_df['evidence'].loc['expect'].loc['rest'].loc[trial]

	df = pd.DataFrame([], columns=['group','response','baseline'])
	
	df['group']	= np.concatenate((np.repeat(['control'],19), np.repeat(['ptsd'],11)))
	




	# print('control no to control expect', ttest_ind(con_no, con_yes))
	# print('control no to ptsd no', ttest_ind(con_no, p_no))
	# print('control no to ptsd expect', ttest_ind(con_no, p_yes))
	# print('/n')


# res.exp_stats(trial_=1)
# pres.exp_stats(trial_=1)
# df = pd.DataFrame([], columns=['group','response','evidence','baseline'])
# df['group']	= np.concatenate((np.repeat(['control'],38), np.repeat(['ptsd'],24)))
# df['response']	= np.concatenate((np.repeat(['no'],23), np.repeat(['expect'],15),
# 								np.repeat(['no'],11), np.repeat(['expect'],13)))

# df['baseline'] = np.concatenate((res._no_,res._expect_,pres._no_,pres._expect_))
# df['baseline'] = np.concatenate((res.ev_baseline['no'][1],res.ev_baseline['expect'][1],
#  								pres.ev_baseline['no'][1],pres.ev_baseline['expect'][1]))
# df['evidence'] = np.concatenate((res.stat_map['no'][1]['scene'],res.stat_map['expect'][1]['scene'],
# 								pres.stat_map['no'][1]['scene'],pres.stat_map['expect'][1]['scene']))



# formula = 'baseline ~ C(group) + C(response) + C(group):C(response)'
# model = ols(formula, df).fit()
# aov_table = anova_lm(model, typ=2)
# print(aov_table)


fig, ax1 = plt.subplots()
ind2 = np.arange(2)    # the x locations for the groups
width = 0.4         # the width of the bars
sns.set_style('whitegrid')
sns.set_style('ticks')
p1 = ax1.bar(ind2[0], res.ev_base_graph['avg'].loc['no'].loc['1'], width, yerr=res.ev_base_graph['err'].loc['no'].loc['1'],
	color=plt.cm.Set1.colors[3], alpha=.8)
p2 = ax1.bar(ind2[0]+width, res.ev_base_graph['avg'].loc['expect'].loc['1'], width, yerr=res.ev_base_graph['err'].loc['expect'].loc['1'],
	edgecolor=plt.cm.Set1.colors[3], color='w', linewidth=4)
p3 = ax1.bar(ind2[1], pres.ev_base_graph['avg'].loc['no'].loc['1'], width, yerr=pres.ev_base_graph['err'].loc['no'].loc['1'],
	color=plt.cm.Set1.colors[3], alpha=.8)
p4 = ax1.bar(ind2[1]+width, pres.ev_base_graph['avg'].loc['expect'].loc['1'], width, yerr=pres.ev_base_graph['err'].loc['expect'].loc['1'],
	edgecolor=plt.cm.Set1.colors[3], color='w', linewidth=4)

legend = ax1.legend((p1[0], p2[0]), ('No', 'Yes'),fontsize='xx-large',title='Expect a shock?',loc='upper right')
legend.get_title().set_fontsize('18') #legend 'Title' fontsize
ax1.set_ylim([-.4,1])
ax1.set_xticks(ind2 + width / 2)
ax1.set_xticklabels(('Control (N=19)','PTSD (N=11)'))
ax1.set_title('Differences in Relative Scene Evidence')
ax1.set_ylabel('Relative Scene Evidence (Scene-Rest)')
ax1.set_xlabel('Group')
# ax1.plot(.2,.75,marker='$*$',markersize=28, color='black')
# ax1.text(.14,.72,s=' ',fontsize=32, color='black')
# ax1.text(.2,.72,s=' ',fontsize=32, color='black')
# ax1.text(.14,.72,s='- -',fontsize=35, color='black')

fig.set_size_inches(8, 6)
plt.tight_layout()
# fig.savefig(os.path.join(data_dir,'graphing','quals','group_baseline_ev.png'), dpi=300)
# fig.savefig(os.path.join(data_dir,'graphing','quals','2trial_exp_tr.png'), dpi=500)



'''
'''
sns.set_style('whitegrid')
sns.set_style('ticks')
sns.set_style(rc={'axes.linewidth':'2'})
plt.rcParams['xtick.labelsize'] = 10 
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.labelspacing'] = .25

trials = [1]
nrows = 2

xaxis_tr = [-2,-1,0,1,2,3]

fig, ax = plt.subplots(nrows,2, sharex='col',sharey='row')
for trial in [0]:
	for cond, color in zip(['csplus','scene','rest'], [plt.cm.Set1.colors[0],plt.cm.Set1.colors[3],plt.cm.Set1.colors[-1],'green']):
		
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

		ax[0][0].set_title('CS+ Trial %s; Did not expect a shock (N=%s)'%(trials[trial], res.nexp['no'][trials[trial]]))
		ax[0][1].set_title('CS+ Trial %s; Expected a shock (N=%s)'%(trials[trial], res.nexp['expect'][trials[trial]]))


		ax[1][0].plot(xaxis_tr, pres.err_['ev'].loc['no'].loc[trials[trial]].loc[cond], color=color, marker='o', markersize=5, label=labelcond)
		
		ax[1][0].fill_between(xaxis_tr, pres.err_['ev'].loc['no'].loc[trials[trial]].loc[cond] - pres.err_['err'].loc['no'].loc[trials[trial]].loc[cond],
		 pres.err_['ev'].loc['no'].loc[trials[trial]].loc[cond] + pres.err_['err'].loc['no'].loc[trials[trial]].loc[cond],
		 alpha=.5, color=color)

		ax[1][1].plot(xaxis_tr, pres.err_['ev'].loc['expect'].loc[trials[trial]].loc[cond], label='%s'%(labelcond), color=color, marker='o', markersize=5)
		
		ax[1][1].fill_between(xaxis_tr, pres.err_['ev'].loc['expect'].loc[trials[trial]].loc[cond] - pres.err_['err'].loc['expect'].loc[trials[trial]].loc[cond],
		 pres.err_['ev'].loc['expect'].loc[trials[trial]].loc[cond] + pres.err_['err'].loc['expect'].loc[trials[trial]].loc[cond],
		 alpha=.5, color=color)

		ax[1][0].set_title('CS+ Trial %s; Did not expect a shock (N=%s)'%(trials[trial], pres.nexp['no'][trials[trial]]))
		ax[1][1].set_title('CS+ Trial %s; Expected a shock (N=%s)'%(trials[trial], pres.nexp['expect'][trials[trial]]))

ax[1][0].set_xlabel('TR (away from stimulus onset)')
ax[1][1].set_xlabel('TR (away from stimulus onset)')

ax[0][0].set_ylabel('Classifier Evidence')
ax[1][0].set_ylabel('Classifier Evidence')

ax[0][0].plot(1,.2,marker='$*$',markersize=10, color='black')
# ax[1][0].plot(0,.5,marker='$*$',markersize=10, color='black')
fig.set_size_inches(10, 6)

# ax[0][1].legend(loc='upper center', bbox_to_anchor=(0.5, 1), frameon=True, ncol=3,borderaxespad=1)#,prop={'size': 8})
# ax[0][1].legend(bbox_to_anchor=(0, .02, 1., 1), frameon=True, loc=3, ncol=3,borderaxespad=0,prop={'size': 8})
# ax[0][0].legend(loc='upper center', bbox_to_anchor=(0.5, 1), frameon=True, ncol=3,borderaxespad=1)#,prop={'size': 8})














_con = np.concatenate((con[1], con[2]))
_ptsd = np.concatenate((ptsd[1], ptsd[2]))
print(ttest_ind(con[1],ptsd[1]))
print(ttest_ind(con[2],ptsd[2]))
print(ttest_ind(_con, _ptsd))



#2       no   0  2.076388  0.064588
#3       no   1  3.663724  0.004362
#0   1.390299  0.182374
#1   3.653655  0.001966
'''
# pres = group_decode(imgs='tr', k=nvox, SC=True, S_DS=True, rmv_scram=True, p=True)
# pres.exp_event()
# pres.vis_exp_event()



#phase_bar_plot(res, title='lololol')


# {beta_res.vis_cond_phase(phase=phase, title='beta_sc_ds') for phase in decode.decode_runs}
# {tr.vis_cond_phase(phase=phase, title='tr_sc_ds') for phase in decode.decode_runs}

# #res_stats = get_bar_stats(VTC_LOC)
# #res_stats.to_csv('%s/graphing/mvpa_analysis/stats_vtc_loc.csv'%(data_dir))




# {beta_res.vis_event_res(phase, title='Train on Betas, scene_ds, k=500') for phase in decode.decode_runs}
# {tr.vis_event_res(phase, title='TR, scene_ds, k=1000') for phase in decode.decode_runs}

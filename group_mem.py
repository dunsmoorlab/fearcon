import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys
from scipy import stats
from scipy.stats import ttest_rel, ttest_ind
from fc_config import *
from preprocess_library import meta
from fc_behavioral import *
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import statsmodels.api as sm

# sns.set_style('whitegrid')
# sns.set_style('ticks')
# sns.set_style(rc={'axes.linewidth':'2'})
# plt.rcParams['xtick.labelsize'] = 18 
# plt.rcParams['ytick.labelsize'] = 16
# plt.rcParams['axes.labelsize'] = 22
# plt.rcParams['axes.titlesize'] = 28
# plt.rcParams['legend.labelspacing'] = .25




res = recognition_memory(hch=True)
pres = recognition_memory(p=True,hch=True)

res.phase_cr['group'] = 'control'
pres.phase_cr['group'] = 'ptsd'

group_mem = pd.concat([res.phase_cr,pres.phase_cr])

group_mem.to_csv(os.path.join(data_dir,'graphing','memory','group_memory.csv'))

sw = sns.factorplot(data=group_mem, x='condition', y='cr',
					col='phase',hue='group',
					kind='point', palette='hls')


c_csp = res.phase_err.loc[np.where(res.phase_err['condition'] == 'CS+')[0]]
c_csm = res.phase_err.loc[np.where(res.phase_err['condition'] == 'CS-')[0]]

p_csp = pres.phase_err.loc[np.where(pres.phase_err['condition'] == 'CS+')[0]]
p_csm = pres.phase_err.loc[np.where(pres.phase_err['condition'] == 'CS-')[0]]

fig, ax = plt.subplots()
width = 0.2
ind1 = np.arange(4)
ind2 = [x + width for x in ind1]
ind3 = [x + width for x in ind2]
ind4 = [x + width for x in ind3]

c_CSP = ax.bar(ind1, c_csp['cr'], width, yerr=(c_csp['err']), color=plt.cm.Set1.colors[0], alpha=.8, edgecolor='black')
c_CSM = ax.bar(ind2, c_csm['cr'], width, yerr=(c_csm['err']), color=plt.cm.Set1.colors[1], alpha=.8, edgecolor='black')

p_CSP = ax.bar(ind3, p_csp['cr'], width, yerr=(p_csp['err']), color='w', edgecolor=plt.cm.Set1.colors[0])
p_CSM = ax.bar(ind4, p_csm['cr'], width, yerr=(p_csm['err']), color='w', edgecolor=plt.cm.Set1.colors[1])


ax.set_xticks([r + (width/2) for r in ind2])
ax.set_xticklabels(['baseline','fear_conditioning','extinction','false_alarms'])

ax.set_ylim([0,.7])
pretty_graph(ax=ax, xlab='Phase', ylab='Corrected Recognition', main='CR by Phase with SEM', legend=True)
ax.legend((c_CSP[0],c_CSM[0], p_CSP[0], p_CSM[0]), ('CS+, control', 'CS-, control', 'CS+, PTSD', 'CS-, PTSD'), fontsize='larger')
fig.set_size_inches(9, 5.5)
plt.tight_layout()
# plt.savefig(os.path.join(data_dir,'graphing', 'cns', 'hch_CR_mem.png'))

# formula = 'cr ~ C(phase)'
# # formula2 = 'evidence ~ C(group) + C(response) + C(group):C(response)'
# model = ols(formula, group_mem).fit()
# aov_table = anova_lm(model, typ=1)
# print(aov_table)

memory_phase = ['baseline','fear_conditioning','extinction','false_alarm']

group_stats = group_mem.set_index(['group','phase','condition'])
phase_t_p = pd.DataFrame(index=memory_phase, columns=['tstat','pval'])
phase_t_m = pd.DataFrame(index=memory_phase, columns=['tstat','pval'])

for phase in memory_phase:
			
	phase_t_p['tstat'][phase], phase_t_p['pval'][phase] = ttest_ind(group_stats['cr']['control'][phase]['CS+'], group_stats['cr']['ptsd'][phase]['CS+'])
	phase_t_m['tstat'][phase], phase_t_m['pval'][phase] = ttest_ind(group_stats['cr']['control'][phase]['CS-'], group_stats['cr']['ptsd'][phase]['CS-'])

print('CS+, Control vs. PTSD')
print(phase_t_p)
print('CS-, Control vs. PTSD')
print(phase_t_m)

for phase in memory_phase:
	print(phase)
	control_dif = group_stats['cr']['control'][phase]['CS+'].values - group_stats['cr']['control'][phase]['CS-'].values
	ptsd_dif = group_stats['cr']['ptsd'][phase]['CS+'].values - group_stats['cr']['ptsd'][phase]['CS-'].values
	print(ttest_ind(control_dif,ptsd_dif))
# fig.set_size_inches(8, 6.5)
# fig.savefig(os.path.join(data_dir,'graphing','memory','group_cr.png'), dpi=300)






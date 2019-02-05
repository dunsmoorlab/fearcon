from beta_rsa import *
from fc_config import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind

roi = vmPFC_beta
fs = 500


c = group_rsa(subs=sub_args,save_dict=roi,fs=fs)
p = group_rsa(subs=p_sub_args,save_dict=roi,fs=fs)



for f in [False,True]:
	c.sub_stats(f=f)
	p.sub_stats(f=f)

	if not f:print('baseline within', ttest_ind(c.stats['baseline'],p.stats['baseline']))
	if not f:print('conditioning within', ttest_ind(c.stats['fear_conditioning'],p.stats['fear_conditioning']))
	if not f:print('extinction within', ttest_ind(c.stats['extinction'],p.stats['extinction']))
	if not f:print('renewal within', ttest_ind(c.stats['renewal'],p.stats['renewal']))
	if not f:print('\n')
	if not f:print('baseline to conditioning', ttest_ind(c.stats['b_f'],p.stats['b_f']))
	if not f:print('baseline to extinction', ttest_ind(c.stats['b_e'],p.stats['b_e']))
	if not f:print('conditioning to extinction', ttest_ind(c.stats['f_e'],p.stats['f_e']))
	if not f:print('\n')
	print('renewal to baseline', ttest_ind(c.stats['r_b'],p.stats['r_b']))
	print('renewal to conditioning', ttest_ind(c.stats['r_f'],p.stats['r_f']))
	print('renewal to extinction', ttest_ind(c.stats['r_e'],p.stats['r_e']))
	print('\n')

# cdif = c.stats['extinction'] - c.stats['fear_conditioning']
# pdif = p.stats['extinction'] - p.stats['fear_conditioning']
# ttest_ind(cdif,pdif)

ex = pd.DataFrame(np.concatenate((c.stats['r_e'],p.stats['r_e'])),columns=['Mean Pattern Similarity'])
ex['Group'] = ''
ex['Group'].loc[0:20] = 'Control'
ex['Group'].loc[20:40] = 'PTSD'
sns.set_context('poster')
sns.set_style('whitegrid',{'axes.facecolor':'.9','figure.facecolor':'.9'})
plt.figure(figsize=(8,8))
sw1 = sns.swarmplot(x='Group',y='Mean Pattern Similarity',hue='Group',data=ex, size=10, alpha=.9,
 palette=[wes_palettes['Chevalier'][0],wes_palettes['Royal1'][1]])
sns.pointplot(x='Group',y='Mean Pattern Similarity',data=ex, palette=['black','black'],capsize=.1)
# sw1.set_ylim(-.015,.055)
plt.title('vmPFC')
sw1.legend_.remove()
plt.tight_layout()
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
from wesanderson import wes_palettes

from fc_behavioral import *

res = shock_expectancy()
pres = shock_expectancy(p=True)


# sns.set_style('whitegrid')
# sns.set_style('ticks')
# sns.set_style(rc={'axes.linewidth':'2'})
# plt.rcParams['xtick.labelsize'] = 18 
# plt.rcParams['ytick.labelsize'] = 20
# plt.rcParams['axes.labelsize'] = 22
# plt.rcParams['axes.titlesize'] = 24
# plt.rcParams['legend.labelspacing'] = .25
# sns.set_context('poster')	
# sns.set_style('whitegrid',{'axes.facecolor':'.9','figure.facecolor':'.9'})	


# fig, ax = plt.subplots(sharey=True)

# ax1 = plt.subplot2grid((2,2),(0,0))
# ax2 = plt.subplot2grid((2,2),(0,1))
# ax3 = plt.subplot2grid((2,2),(1,0), colspan=2)

# cond = 'CS'
conds = ['CS+','CS-']
# sns.set_style('ticks',{'axes.grid':True,'grid.color':'.9','axes.edgecolor':'black'})
sns.set_style('ticks')
for cond in conds:
	fig, (ax1, ax2) = plt.subplots(1,2,sharex=True)
	fig3, ax3 = plt.subplots()
	ax1.plot(range(1,25), res.prop_df['avg'].loc['fear_conditioning'].loc[cond], marker='o', color=wes_palettes['Zissou'][0], markersize=6, linewidth=4)

	ax1.plot(range(1,25), pres.prop_df['avg'].loc['fear_conditioning'].loc[cond],  marker='o', color=wes_palettes['Royal1'][1], markersize=6, linewidth=4)
	ax1.set_xticks(np.arange(4,25,4))


	ax2.plot(range(1,25), res.prop_df['avg'].loc['extinction'].loc[cond], marker='o', color=wes_palettes['Zissou'][0], markersize=6, linewidth=4)

	ax2.plot(range(1,25), pres.prop_df['avg'].loc['extinction'].loc[cond], marker='o', color=wes_palettes['Royal1'][1], markersize=6, linewidth=4)
	ax2.set_xticks(np.arange(4,25,4))


	ax3.plot(range(1,13), res.prop_df['avg'].loc['extinction_recall'].loc[cond], marker='o', color=wes_palettes['Zissou'][0], markersize=6, linewidth=4)

	ax3.plot(range(1,13), pres.prop_df['avg'].loc['extinction_recall'].loc[cond], marker='o', color=wes_palettes['Royal1'][1], markersize=6, linewidth=4)
	ax3.set_xticks(np.arange(2,13,2))
	ax3.set_yticks(np.arange(.2,1.2,.2)) 

	ax1.set_ylim([0,1])
	ax2.set_ylim([0,1])
	ax3.set_ylim([0,1])
	ax1.set_xlim([.5,24.5])
	ax2.set_xlim([.5,24.5])
	ax3.set_xlim([.5,12.5])

	ax1.set_title('Fear Conditioning %s'%(cond))
	ax2.set_title('Extinction %s'%(cond))
	ax3.set_title('Extinction Recall %s'%(cond))

	# ax1.legend(loc='upper right', fontsize='larger')
	# ax2.legend(loc='upper right', fontsize='larger')
	# ax3.legend(loc='upper right', fontsize='larger')
	fig.suptitle(x=0.03, y=0.5, t='Proportion Expecting a Shock', va='center', rotation='vertical', size=24)
	ax3.set_ylabel('Proportion Expecting a Shock')

	# ax1.set_xlabel('Trial',size = 'x-large')
	ax2.set_xlabel('Trial')
	ax3.set_xlabel('Trial')

	fig.set_size_inches(6, 12)
	fig3.set_size_inches(5, 5)
	
	fig.tight_layout(w_pad=.5)
	fig3.tight_layout()
	
	sns.despine(ax=ax1)
	sns.despine(ax=ax2)
	sns.despine(ax=ax3)
	# fig.savefig(os.path.join(data_dir,'graphing','quals','day1_shock_expectancy_%s.png'%(cond)))
	# fig3.savefig(os.path.join(data_dir,'graphing','quals','day2_shock_expectancy_%s.png'%(cond)))

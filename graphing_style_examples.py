import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wesanderson import wes_palettes
from fc_config import data_dir, sub_args, p_sub_args
sns.set_context('talk')
sns.set_style('ticks',{'axes.grid':True,'grid.color':'.9','axes.edgecolor':'black'})
gpal = list((wes_palettes['Zissou'][0],wes_palettes['Royal1'][1]))
cpal = ['darkorange','grey']





df = pd.read_csv(os.path.join(data_dir,'graphing','beta_RSA','mOFC_collected_rsa.csv'))

df = pd.DataFrame({
    'z': np.concatenate((df.c_e_p,df.c_e_m,df.p_e_p,df.p_e_m)),
    'group': np.repeat(['control','ptsd'],48),
    'condition': np.tile(np.repeat(['CS+','CS-'],24),2),
    'subject': np.tile([sub_args,p_sub_args],2).flatten()
    })
#boxplot
fig, ax = plt.subplots()
sns.boxplot(x='group',y='z',hue='condition',data=df,palette=cpal,fliersize=8,ax=ax)
sns.despine(ax=ax);ax.legend_.remove()

#box with swarm
fig, ax = plt.subplots()
sns.boxplot(x='group',y='z',hue='condition',data=df,palette=cpal,fliersize=0,ax=ax)
sns.swarmplot(x='group',y='z',hue='condition',data=df,ax=ax,linewidth=2,edgecolor='black',size=8,
			palette=cpal,dodge=True)
sns.despine(ax=ax);ax.legend_.remove()

#bar with err & swarm
_sems = df.groupby(['group','condition']).sem()
_avgs = df.groupby(['group','condition']).mean()

fig, ax = plt.subplots()

sns.barplot(x='group',y='z',hue='condition',data=df,palette=cpal,ax=ax,ci=None)
sns.stripplot(x='group',y='z',hue='condition',data=df,ax=ax,linewidth=2,edgecolor='black',size=8,
			palette=cpal,dodge=True)
for i, group in enumerate(df.group.unique()):
    ax.errorbar(i-.3,_avgs['z'][group]['CS+'],yerr=_sems['z'][group]['CS+'],color='black',linewidth=3,capsize=5,capthick=3)
    ax.errorbar(i+.1,_avgs['z'][group]['CS-'],yerr=_sems['z'][group]['CS-'],color='black',linewidth=3,capsize=5,capthick=3)
sns.despine(ax=ax);ax.legend_.remove()
	    # t, p = pg.ttest(df[val][df.group=='control'],df[val][df.group=='ptsd'])[['T','p-val']].loc['T-test']
	    # ax.set_title('t = %s; p = %s'%(np.round(t,4),np.round(p,4)))
	    # plt.tight_layout()

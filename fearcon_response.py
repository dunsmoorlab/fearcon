import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, ScalarFormatter
import seaborn as sns
from fc_config import *
####SCR####
SCR = pd.read_csv(os.path.join(data_dir,'graphing','SCR','fc_scr.csv'))

#graph control and ptsd across the both conditioning and extinction in halves
fig, ax = plt.subplots()
ax = sns.boxplot(x='quarter',y='scr',hue='gc',data=SCR,
                palette=['darkorange','grey','orange','lightgrey'],)
# sns.stripplot(x='quarter',y='scr',hue='gc',data=SCR,
#                 palette=['grey','grey','grey','grey'],dodge=True,linewidth=1)
ax.set_xticklabels(['early fear','late fear','early ext','late ext','early rnw','late rnw'])
ax.set_ylabel('Sqrt SCR');ax.set_xlabel('');ax.set_title(label='SCR by phase half')
ax.legend_.remove()
###################
#try for a discriminatory graphing comparing groups
# sns.set_style('whitegrid')
comp = pd.read_csv(os.path.join(data_dir,'graphing','SCR','fc_scr_comp.csv'))
comp_avg = comp.groupby(['group','quarter']).mean()['scr']
comp_err = comp.groupby(['group','quarter']).sem()['scr']
fig, ax = plt.subplots()
# sns.boxplot(x='quarter',y='scr',hue='group',data=comp,palette=gpal,ax=ax,fliersize=8)
sns.barplot(x='quarter',y='scr',hue='group',data=comp,palette=gpal,ax=ax,ci=None)
sns.stripplot(x='quarter',y='scr',hue='group',data=comp,dodge=True,
            palette=gpal,linewidth=2,edgecolor='black',size=8,ax=ax)
for i, quarter in enumerate(comp.quarter.unique()):
    x = ax.get_xticks()[i]
    ax.errorbar(x-.35,comp_avg.loc[('control',quarter)], yerr=comp_err.loc[('control',quarter)], color='black', linewidth=3,capsize=5,capthick=3)
    ax.errorbar(x+.05,comp_avg.loc[('ptsd',quarter)], yerr=comp_err.loc[('ptsd',quarter)], color='black', linewidth=3,capsize=5,capthick=3)
ax.hlines(0,ax.get_xlim()[0],ax.get_xlim()[-1],color='black',linewidth=2)
ax.set_xticklabels(['early fear','late fear','early ext','late ext','early rnw','late rnw'])
ax.set_title('SCR (CS+ - CS-) by phase');
ax.set_ylabel('Sqrt SCR');ax.set_xlabel('');ax.legend_.remove()
sns.despine(ax=ax)

scr_stats = SCR.groupby(['group','condition','quarter']).mean()
scr_stats = scr_stats.drop(columns=['half','subject']).rename(columns={'scr':'avg'})
scr_stats['err'] = SCR.groupby(['group','condition','quarter']).sem()['scr']

#By group CS+ and CS- in single panel
for group in ['control','ptsd']:
    fig, ax = plt.subplots()
    sns.barplot(x='quarter',y='scr',hue='condition',data=SCR.query('group == @group'),palette=cpal,ax=ax,ci=None)
    sns.stripplot(x='quarter',y='scr',hue='condition',data=SCR.query('group == @group'),dodge=True,
            palette=cpal,linewidth=2,edgecolor='black',size=8,ax=ax)
    ax.legend_.remove()
    # for i, quarter in enumerate(scr_stats.quarter.unique()):
    x = ax.get_xticks()
    ax.errorbar(x-.35,scr_stats.loc[(group,'CS+'),'avg'], yerr=scr_stats.loc[(group,'CS+'),'err'], color='black', linewidth=3,capsize=5,capthick=3,fmt='o')
    ax.errorbar(x+.05,scr_stats.loc[(group,'CS-'),'avg'], yerr=scr_stats.loc[(group,'CS-'),'err'], color='black', linewidth=3,capsize=5,capthick=3,fmt='o')
    ax.set_title(group)
    ax.set_ylim(0,2.2)
#t-test against 0 for split
sstats = SCR.set_index(['group','condition','quarter'])
for group in ['control','ptsd']:
    for condition in ['CS+','CS-']:
        for quarter in range(1,7):
            t = pg.ttest(sstats.loc[(group,condition,quarter),'scr'],0)
            print(group,condition,quarter,'\n',t,'\n\n')



#Late extinction to early renewal stats
rnw = comp.set_index(['quarter','subject']).loc[([4,5])]
c = rnw[rnw.group == 'control']
p = rnw[rnw.group == 'ptsd']
pg.ttest(c.loc[5,'scr'],c.loc[4,'scr'],paired=True)
pg.ttest(p.loc[5,'scr'],p.loc[4,'scr'],paired=True)

rcomp = comp.set_index(['quarter','subject'])
rcomp = (rcomp.loc[5,'scr'] - rcomp.loc[4,'scr']).reset_index()
rcomp['group'] = rcomp.subject.apply(lgroup)
pg.ttest(rcomp.scr[rcomp.group == 'ptsd'],rcomp.scr[rcomp.group == 'control'])


csp = (SCR[SCR.condition=='CS+']).set_index('quarter','subject').loc[([4,5])]
cp = csp[csp.group=='control']
pg.ttest(cp.loc[5,'scr'],cp.loc[4,'scr'],paired=True)

pp = csp[csp.group=='ptsd']
pg.ttest(pp.loc[5,'scr'],pp.loc[4,'scr'],paired=True)

#All phases diff than 0
res = {}
comp = comp.set_index(['group','quarter'])
for group in ['control','ptsd']:
    res[group] = {}
    for quarter in range(1,7):
        tres = pg.ttest(comp.loc[(group,quarter),'scr'],0)
        tres['group'] = group
        tres['quarter'] = quarter
        res[group][quarter] = tres
c = pd.concat(res['control'].values())
p = pd.concat(res['ptsd'].values())

g = pd.concat([c,p])
g['corrected'] = pg.multicomp(g['p-val'].values,method='bonf')[1]



########################
#better shock expectancy graphs
from fc_behavioral import *
ce = shock_expectancy()
pe = shock_expectancy(p=True)

ce.exp_df['group'] = 'control'
pe.exp_df['group'] = 'ptsd'

exp = pd.concat([ce.exp_df,pe.exp_df])
exp = exp.set_index(['condition','phase','trial'])
exp = exp.sort_index()
exp['quarter'] = 0

for con in ['CS+','CS-']:
    exp.loc[(con,'fear_conditioning',slice(1,12)),'quarter'] = 1
    exp.loc[(con,'fear_conditioning',slice(13,24)),'quarter'] = 2
    exp.loc[(con,'extinction',slice(1,12)),'quarter'] = 3
    exp.loc[(con,'extinction',slice(13,24)),'quarter'] = 4    
    exp.loc[(con,'extinction_recall',slice(1,4)),'quarter'] = 5
    exp.loc[(con,'extinction_recall',slice(5,12)),'quarter'] = 6
exp = exp.reset_index().groupby(['subject','condition','quarter']).mean().reset_index().drop(columns=['trial','cavg'])
exp['group'] = exp.subject.apply(lgroup)

exp_stats = exp.groupby(['group','condition','quarter']).mean()
exp_stats = exp_stats.drop(columns=['subject']).rename(columns={'exp':'avg'})
exp_stats['err'] = exp.groupby(['group','condition','quarter']).sem()['exp']


for group in ['control','ptsd']:
    fig,ax = plt.subplots()    
    sns.barplot(x='quarter',y='exp',hue='condition',data=exp.query('group == @group'),palette=cpal,ax=ax,ci=None)
    sns.stripplot(x='quarter',y='exp',hue='condition',data=exp.query('group == @group'),dodge=True,
            palette=cpal,linewidth=2,edgecolor='black',size=8,ax=ax)
    ax.legend_.remove()
    # for i, quarter in enumerate(scr_stats.quarter.unique()):
    x = ax.get_xticks()
    ax.errorbar(x-.35,exp_stats.loc[(group,'CS+'),'avg'], yerr=exp_stats.loc[(group,'CS+'),'err'], color='black', linewidth=3,capsize=5,capthick=3,fmt='o')
    ax.errorbar(x+.05,exp_stats.loc[(group,'CS-'),'avg'], yerr=exp_stats.loc[(group,'CS-'),'err'], color='black', linewidth=3,capsize=5,capthick=3,fmt='o')
    ax.set_title(group)
    ax.set_ylim(0,1.1)

# eres = {}
eres = exp.set_index(['group','condition','quarter'])
for group in ['control','ptsd']:
    for condition in ['CS+','CS-']:
    # eres[group] = {}
        for quarter in range(1,7):
            tres = pg.ttest(eres.loc[(group,condition,quarter),'exp'],0)
        # tres['group'] = group
        # tres['quarter'] = quarter
        # eres[group][quarter] = tres
            print(group,condition,quarter,'\n',tres,'\n')
ec = pd.concat(eres['control'].values())
ep = pd.concat(eres['ptsd'].values())

eg = pd.concat([ec,ep])
eg['corrected'] = pg.multicomp(eg['p-val'].values,method='bonf')[1]

#########
#late extinction rsa
roi = 'mOFC'
label = 'e_e__e_r'
print(roi)
p = pd.read_csv(os.path.join(data_dir,'graphing','beta_rsa','split_%s_csp.csv'%(roi)))
m = pd.read_csv(os.path.join(data_dir,'graphing','beta_rsa','split_%s_csm.csv'%(roi)))

ep = p[p.label == label]
em = m[m.label == label]

ep = ep.reset_index(drop=True)
em = em.reset_index(drop=True)
ep['condition'] = 'CS+'
em['condition'] = 'CS-'

rsa = pd.concat([ep,em]).reset_index(drop=True)
pg.mixed_anova(data=rsa,dv='z',between='group',within='condition',subject='sub')
pg.pairwise_ttests(data=rsa,dv='z',between='group',within='condition',subject='sub')
barstrip(rsa,'z',cond=True,points='strip')

c = rsa.query('group == "control"')
p = rsa.query('group == "ptsd"')
pg.ttest(c.z[c.condition=='CS+'],c.z[c.condition=='CS-'],paired=True)
pg.ttest(p.z[p.condition=='CS+'],p.z[p.condition=='CS-'],paired=True)

#########
#partial correlation with ev-vmPFC-exp
crmod['exp'] = eres.loc[('control','CS+',5),'exp'].values
prmod['exp'] = eres.loc[('ptsd','CS+',5),'exp'].values

#########
#new construct validity figure
###SCR
comp = pd.read_csv(os.path.join(data_dir,'graphing','SCR','fc_scr_comp.csv'))
comp = comp[comp.quarter != 6]
comp['phase'] = ''
for i in [1,2]: comp.phase[comp.quarter == i] = 'fear'
comp.phase[comp.quarter == 3] = 'early_ext'
comp.phase[comp.quarter == 4] = 'late_ext'
comp.phase[comp.quarter == 5] = 'early_rnw'
comp.phase = pd.Categorical(comp.phase,['fear','early_ext','late_ext','early_rnw'],ordered=True)

comp_stats = comp.groupby(['group','phase']).mean()
comp_stats = comp_stats.drop(columns=['subject','quarter']).rename(columns={'scr':'avg'})
comp_stats['err'] = comp.groupby(['group','phase']).sem()['scr']

fig, ax = plt.subplots()
sns.barplot(x='phase',y='scr',hue='group',data=comp,palette=gpal,ax=ax,ci=95,errcolor='black',errwidth=3,capsize=.1)
# for i, phase in enumerate(comp.phase.unique()):
#     x = ax.get_xticks()[i]
#     ax.errorbar(x-.2,comp_stats.loc[('control',phase),'avg'], yerr=comp_stats.loc[('control',phase),'err'], color='black', linewidth=3,capsize=5,capthick=3)
#     ax.errorbar(x+.2,comp_stats.loc[('ptsd',phase),'avg'], yerr=comp_stats.loc[('ptsd',phase),'err'], color='black', linewidth=3,capsize=5,capthick=3)
# # ax.hlines(0,ax.get_xlim()[0],ax.get_xlim()[-1],color='black',linewidth=2)
ax.set_xticklabels(['fear','early ext','late ext','early rnw'])
ax.set_title('SCR (CS+ - CS-) by phase');
ax.set_ylabel('Sqrt SCR');ax.set_xlabel('');ax.legend_.remove()
sns.despine(ax=ax)
ax.set_ylim(-.05,.4)
ax.yaxis.set_major_locator(MultipleLocator(.1))
ax.yaxis.set_minor_locator(MultipleLocator(.05))
plt.savefig(os.path.join(data_dir,'paper_writeup','figures','response','scr_validity.eps'),format='eps')

###EXP
from fc_behavioral import *
ce = shock_expectancy()
pe = shock_expectancy(p=True)

ce.exp_df['group'] = 'control'
pe.exp_df['group'] = 'ptsd'

exp = pd.concat([ce.exp_df,pe.exp_df])
exp = exp.set_index(['condition','phase','trial'])
exp = exp.sort_index()
exp['quarter'] = 'fear'

for con in ['CS+','CS-']:
    exp.loc[(con,'fear_conditioning',slice(1,12)),'quarter'] = 'fear'
    exp.loc[(con,'fear_conditioning',slice(13,24)),'quarter'] = 'fear'
    exp.loc[(con,'extinction',slice(1,12)),'quarter'] = 'early_ext'
    exp.loc[(con,'extinction',slice(13,24)),'quarter'] = 'late_ext'
    exp.loc[(con,'extinction_recall',slice(1,4)),'quarter'] = 'early_rnw'
    exp.loc[(con,'extinction_recall',slice(5,12)),'quarter'] = 'drop'
exp = exp[exp.quarter != 'drop']
exp.quarter = pd.Categorical(exp.quarter,['fear','early_ext','late_ext','early_rnw'],ordered=True)
exp = exp.reset_index().groupby(['condition','subject','quarter']).mean().drop(columns=['trial','cavg'])

ecomp = (exp.loc['CS+'] - exp.loc['CS-']).reset_index()
ecomp['group'] = ecomp.subject.apply(lgroup)

exp_stats = ecomp.groupby(['group','quarter']).mean()
exp_stats = exp_stats.drop(columns=['subject']).rename(columns={'exp':'avg'})
exp_stats['err'] = ecomp.groupby(['group','quarter']).sem()['exp']

fig, ax = plt.subplots()
sns.barplot(x='quarter',y='exp',hue='group',data=ecomp,palette=gpal,ax=ax,ci=95,errcolor='black',errwidth=3,capsize=.1)
# for i, quarter in enumerate(ecomp.quarter.unique()):
#     x = ax.get_xticks()[i]
#     ax.errorbar(x-.2,exp_stats.loc[('control',quarter),'avg'], yerr=exp_stats.loc[('control',quarter),'err'], color='black', linewidth=3,capsize=5,capthick=3)
#     ax.errorbar(x+.2,exp_stats.loc[('ptsd',quarter),'avg'], yerr=exp_stats.loc[('ptsd',quarter),'err'], color='black', linewidth=3,capsize=5,capthick=3)
ax.set_xticklabels(['fear','early ext','late ext','early rnw'])
ax.set_title('Exp (CS+ - CS-) by phase');
ax.set_ylabel('Expectancy');ax.set_xlabel('');ax.legend_.remove()
ax.set_ylim(0,.6)
ax.yaxis.set_major_locator(MultipleLocator(.2))
ax.yaxis.set_minor_locator(MultipleLocator(.1))
plt.savefig(os.path.join(data_dir,'paper_writeup','figures','response','exp_validity.eps'),format='eps')

#group comparisons for both measures
ecomp = ecomp.set_index(['group','quarter'])
for phase in ['fear','early_ext','late_ext','early_rnw']:
    print(phase,'\n\n',pg.ttest(ecomp.loc[('control',phase),'exp'], ecomp.loc[('ptsd',phase),'exp']))

comp = comp.set_index(['group','phase'])
for phase in ['fear','early_ext','late_ext','early_rnw']:
    print(phase,'\n\n',pg.ttest(comp.loc[('control',phase),'scr'], comp.loc[('ptsd',phase),'scr']))

#hard code the comparisons to show extinction and renewal
#first we have to get rid of the subjects that are missing SCR data
comp = pg.remove_rm_na(data=comp.reset_index(),dv='scr',subject='subject',within='phase')
comp['group'] = comp.subject.apply(lgroup)
comp = comp.set_index(['group','phase'])

for group in ['control','ptsd']:
    print(group,'\nextinction\nscr\n',pg.ttest(comp.loc[(group,'late_ext'),'scr'], comp.loc[(group,'fear'),'scr'],paired=True),'\n\n')
    print(group,'\nextinction\nexp\n',pg.ttest(ecomp.loc[(group,'late_ext'),'exp'], ecomp.loc[(group,'fear'),'exp'],paired=True),'\n\n')

    print(group,'\nrenewal\nscr\n',pg.ttest(comp.loc[(group,'early_rnw'),'scr'], comp.loc[(group,'late_ext'),'scr'],paired=True),'\n\n')
    print(group,'\nrenewal\nexp\n',pg.ttest(ecomp.loc[(group,'early_rnw'),'exp'], ecomp.loc[(group,'late_ext'),'exp'],paired=True),'\n\n')

#fear to early extinction
for group in ['control','ptsd']:
    print(group,'\n\n',pg.ttest(ecomp.loc[(group,'early_ext'),'exp'], ecomp.loc[(group,'fear'),'exp'], paired=True))
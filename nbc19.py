import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib.gridspec as gridspec
import pingouin as pg
from fc_config import *
from wesanderson import wes_palettes
from nilearn.input_data import NiftiMasker
from glm_timing import glm_timing
from mvpa_analysis import group_decode
from signal_change import collect_ev
from scipy.stats import linregress as lm
from scipy.interpolate import interp1d
from numpy.polynomial.polynomial import polyfit


sns.set_context('talk')
# sns.set_style('ticks',{'axes.grid':True,'grid.color':'.9','axes.edgecolor':'black'})
sns.set_style('ticks')
gpal = list((wes_palettes['Zissou'][0],wes_palettes['Royal1'][1]))
cpal = ['darkorange','grey']
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
sns.set_style('whitegrid')
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
################Exp behavior####################
rbh = pd.read_csv(os.path.join(data_dir,'graphing','nbc19','exp_bhv.csv'))
fig, exp = plt.subplots()
exp = sns.boxplot(x='Group',y='Relative Context Reinstatement',hue='Response',data=df,
                palette=[wes_palettes['Royal1'][0],wes_palettes['Darjeeling2'][1]],
                hue_order=['Expect','Do not expect'])
sns.swarmplot(x='Group',y='Relative Context Reinstatement',hue='Response',data=df,size=10,hue_order=['Expect','Do not expect'],
                        dodge=True,palette=['black','black'])
# exp.set_ylim(-.1,1.1)
exp.legend_.remove()

sns.pointplot(x='Group',y='Relative Context Reinstatement',hue='Response',data=df,hue_order=['Expect','Do not expect'],
    join=False,palette=['black','black'],capsize=.1,dodge=.45)

##try TR...
imgs='beta';SC=False;S_DS=False;rmv_scram=False;rmv_ind=False;rmv_rest=False;verbose=False;split=False;con = 'CS+';binarize=True;nvox = 'all';conds = ['scene','scrambled']
save_dict=beta_ppa_prepped
res = group_decode(conds=conds, imgs=imgs, k=nvox, save_dict=save_dict, rmv_rest=rmv_rest, binarize=binarize, SC=SC, S_DS=S_DS, rmv_scram=rmv_scram, rmv_ind=rmv_ind, verbose=verbose)
pres = group_decode(conds=conds, imgs=imgs, k=nvox, save_dict=save_dict, rmv_rest=rmv_rest, binarize=binarize, SC=SC, S_DS=S_DS, rmv_scram=rmv_scram, rmv_ind=rmv_ind, p=True, verbose=verbose)
collect_ev(res=res,pres=pres,save_dict=save_dict,split=split)

ctr = pd.read_csv(os.path.join(data_dir,'graphing','nbc19','control_exp_ev_df.csv'))
ptr = pd.read_csv(os.path.join(data_dir,'graphing','nbc19','ptsd_exp_ev_df.csv'))
# ctr = res.exp_ev_df.copy()
# ptr = pres.exp_ev_df.copy()
ctr = ctr[ctr.trial == 1][ctr.condition == 'scene']
ctr = ctr[ctr.tr != 3]
ctr = ctr[ctr.tr != 2]
ctr.reset_index(inplace=True,drop=True)
ctr['group'] = 'control'
ptr = ptr[ptr.trial == 1][ptr.condition == 'scene']
ptr = ptr[ptr.tr != 3]
ptr = ptr[ptr.tr != 2]
ptr.reset_index(inplace=True,drop=True)
ptr['group'] = 'ptsd'
TR = pd.concat([ctr,ptr])

gTR = pd.DataFrame([],index=pd.MultiIndex.from_product(
                            [['control','ptsd'],['expect','no'],[-2,-1,0,1]],
                            names=['group','response','tr']),
                        columns=['avg','err'])
gTR.avg = TR.groupby(['group','response','tr']).mean()['evidence']
gTR.err = TR.groupby(['group','response','tr']).sem()['evidence']
gTR.reset_index(inplace=True)

#[expect,no]
colors = ['grey','purple']
hatches = ['x','.']
edgecolors = ['purple','none']
lines = ['--','-']
W = [1,1]
A = [.5,.5]
fig, ax = plt.subplots(1,2,sharey=True)
for g, group in enumerate(gTR.group.unique()):
    data = gTR.query('group == @group')
    # sns.lineplot(x='tr',y='avg',hue='response',data=data,style='response',style_order=['no','expect'],
    #                   palette=['grey','purple'],ax=ax[g])
    # ax[g].set_xticks(data.tr.unique())
    # ax[g].legend_.remove();ax[g].set_xlim(-2,1)
    for i, resp in enumerate(gTR.response.unique()):
        dat = data[data.response == resp]
        # ax[g].fill_between(ax[g].get_xticks(),dat.avg-dat.err,dat.avg+dat.err,alpha=A[i],
        # 	color=colors[i])
        
        # get mean/sem per TR for current class
        mean_vals = dat.avg
        uppersem_vals = mean_vals + dat.err
        lowersem_vals = mean_vals - dat.err
        # interpolate
        x_interp = pd.np.linspace(xvals.min(),xvals.max(),1000,endpoint=True)
        up_spliner = interp1d(xvals,uppersem_vals,kind='cubic')
        lo_spliner = interp1d(xvals,lowersem_vals,kind='cubic')
        mean_spliner = interp1d(xvals,mean_vals,kind='cubic')

        uppersem_interp = up_spliner(x_interp)
        lowersem_interp = lo_spliner(x_interp)
        mean_interp = mean_spliner(x_interp)
        # draw
        ax[g].plot(x_interp,mean_interp,color=colors[i],linestyle=lines[i])
        ax[g].fill_between(x_interp,uppersem_interp,lowersem_interp,
                        color=colors[i],alpha=A[i])
        sns.despine(ax=ax[g]);ax[g].set_xlim(-2,1)


CN = sTR.evidence[sTR.group == 'control'][sTR.response == 'no'].copy()
CE = sTR.evidence[sTR.group == 'control'][sTR.response == 'expect'].copy()
PN = sTR.evidence[sTR.group == 'ptsd'][sTR.response == 'no'].copy()
PE = sTR.evidence[sTR.group == 'ptsd'][sTR.response == 'expect'].copy()
#stats with all data
#control
pg.ttest(CN,CE)
#ptsd
pg.ttest(PN,CE)

CE_o = rm_out(CE)
aTR = pd.DataFrame({'group': ['control']*22 + ['ptsd']*24,
                    'evidence': np.concatenate([CN,CE_o,PN,PE]),
                    'response':['no']*CN.shape[0] + ['yes']*CE_o.shape[0] + ['no']*PN.shape[0] + ['yes']*PE.shape[0]})
###############single trial vmPFC beta###############

bdf = pd.read_csv(os.path.join(data_dir,'graphing','signal_change','extinction_recall_trial_betas.csv'))
bdf = bdf.set_index(['subject','trial'])
bdf['condition'] = ''
bdf['con_trial'] = ''
for sub in all_sub_args:
    timing = glm_timing(sub,'extinction_recall')
    cons = timing.phase_events(con=True)
    conbytrial = timing.phase_meta.cstypebytrial
    
    bdf.loc[sub,'condition'] = np.repeat(cons.values,7)
    bdf.loc[sub,'con_trial'] = np.repeat(conbytrial.values,7)
bdf.reset_index(inplace=True)

tdf = bdf[bdf.con_trial=='CS+_001']
tdf.reset_index(inplace=True,drop=True)
tdf['group'] = np.repeat(('control','ptsd'),int(tdf.shape[0]/2))

ev = pd.read_csv(os.path.join(data_dir,'graphing','signal_change','mvpa_ev.csv'))
tdf['response'] = np.repeat(ev.Response.values,7)
tdf['res_group'] = tdf.group + '_' + tdf.response

sns.pointplot(x='roi',y='beta',hue='res_group',data=tdf,dodge=.5,join=False,
            palette=['grey','blue','lightgrey','lightblue'],
            hue_order=['control_exp','control_no','ptsd_exp','ptsd_no'])
plt.title('1st CS+ trial beta estimates')

tev = res.exp_ev_df.copy()
tev = tev[tev.trial == 1]
tev = tev[tev.tr == 0]
tev = tev[tev.condition == 'scene']

cmtdf = tdf.query('group == "control"').copy()
mtdf = tdf.query('roi == "hippocampus"').copy()
cmtdf = mtdf.query('group == "control"').copy()


cmtdf['ev'] = ev.ev[ev.Group == 'Control'].values
cmtdf.reset_index(inplace=True,drop=True)
cmtdf['bresp'] = 0
cmtdf.loc[np.where(cmtdf.response == 'exp')[0],'bresp'] = 1
pg.mediation_analysis(x='ev',y='bresp',m='beta',data=cmtdf)
##################################################

ev = pd.read_csv(os.path.join(data_dir,'graphing','signal_change','mvpa_ev.csv'))
rb = pd.read_csv(os.path.join(data_dir,'graphing','signal_change','run004_beta_values.csv'))
eb = pd.read_csv(os.path.join(data_dir,'graphing','signal_change','run003_beta_values.csv'))
fb = pd.read_csv(os.path.join(data_dir,'graphing','signal_change','run002_beta_values.csv'))
scr = pd.read_csv(os.path.join(data_dir,'graphing','SCR','fc_scr_comp.csv'))
rsa = pd.read_csv(os.path.join(data_dir,'graphing','beta_RSA','mOFC_collected_rsa.csv'))
v = pd.DataFrame({'ev':ev.ev,
                  'renewal':rb.early_CSp_CSm[rb.roi == 'mOFC_beta'].values,
                  'extinction':eb.CSp_CSm[eb.roi == 'mOFC_beta'].values,
                  'group':np.repeat(('control','ptsd'),24),
                  'rnw_scr':scr.scr[scr.quarter == 5].values,
                  'ext_rsa':np.concatenate((rsa.c_e_p.values - rsa.c_e_m.values,rsa.p_e_p.values - rsa.p_e_m.values))
                  })

cv = v[v.group == 'control']
pv = v[v.group == 'ptsd']
pg.mediation_analysis(x='ev',m='extinction',y='renewal',data=cv,n_boot=10000)
pg.pairwise_corr(cv)
##########

rmod = pd.DataFrame({'ev': ev.ev,
                     'vmPFC': rb.early_CSp_CSm[rb.roi == 'mOFC_beta'].values,
                     'HC': rb.early_CSp_CSm[rb.roi == 'hippocampus_beta'].values,
                     'amyg': rb.early_CSp_CSm[rb.roi == 'amygdala_beta'].values,
                     'group': np.repeat(('control','ptsd'),24),
                     'bgroup': np.repeat((0,1),24),
                     'rnw_scr': scr.scr[scr.quarter == 5].values,
                     'ext_vmPFC': eb.CSp_CSm[eb.roi == 'mOFC_beta'].values,
                     'ext_scr': scr.scr[scr.quarter == 4].values,
                     'ext_rsa':np.concatenate((rsa.c_e_p.values - rsa.c_e_m.values,rsa.p_e_p.values - rsa.p_e_m.values)),
                     })
crmod = rmod[rmod.group == 'control']
prmod = rmod[rmod.group == 'ptsd']
prmod.reset_index(inplace=True,drop=True)
#try excluding non-responders for SCR
# crmod.drop(index=np.where(crmod.rnw_scr == 0)[0],inplace=True)
# prmod.drop(index=np.where(prmod.rnw_scr == 0)[0],inplace=True)

cmed, cdist = pg.mediation_analysis(x='ev',m=['vmPFC','HC'],y='amyg',data=crmod,
						n_boot=1000,seed=42,return_dist=True)
pmed, pdist = pg.mediation_analysis(x='ev',m=['vmPFC','HC'],y='amyg',data=prmod,
						n_boot=1000,seed=42,return_dist=True)
# cdist = pd.DataFrame({'vmPFC':cdist[:,0],'HC':cdist[:,1]})
# pdist = pd.DataFrame({'vmPFC':pdist[:,0],'HC':pdist[:,1]})
Dist = pd.DataFrame({'group':np.repeat(['control','ptsd'],1000),
                     'vmPFC':np.concatenate([cdist[:,0],pdist[:,0]]),
                     'HC':np.concatenate([cdist[:,1],pdist[:,1]])})
#make color maps
c_cmap = sns.light_palette(gpal[0],as_cmap=True)
p_cmap = sns.light_palette(gpal[1],as_cmap=True)
for roi in ['vmPFC','HC']:
    fig, ax = plt.subplots()
    sns.violinplot(x='group',y=roi,palette=gpal,data=Dist,scale='count',ax=ax,inner=None)
    ax.hlines(y=cmed.coef[cmed.path == 'Indirect %s'%(roi)],xmin=-.125,xmax=.125,lw=2)
    ax.hlines(y=cmed['CI[2.5%]'][cmed.path == 'Indirect %s'%(roi)],xmin=-.125,xmax=.125,lw=2,linestyle='--')
    ax.hlines(y=cmed['CI[97.5%]'][cmed.path == 'Indirect %s'%(roi)],xmin=-.125,xmax=.125,lw=2,linestyle='--')

    ax.hlines(y=pmed.coef[pmed.path == 'Indirect %s'%(roi)],xmin=.875,xmax=1.125,lw=2)
    ax.hlines(y=pmed['CI[2.5%]'][pmed.path == 'Indirect %s'%(roi)],xmin=.875,xmax=1.125,lw=2,linestyle='--')
    ax.hlines(y=pmed['CI[97.5%]'][pmed.path == 'Indirect %s'%(roi)],xmin=.875,xmax=1.125,lw=2,linestyle='--')
    
    sns.despine(ax=ax)
    # sns.distplot(cdist[roi],color=gpal[0],ax=ax,kde_kws={'lw':3},vertical=True)
	# sns.distplot(pdist[roi],color=gpal[1],ax=ax,kde_kws={'lw':3},vertical=True)
	# ax.hlines(np.mean(cdist.vmPFC),xmin=0,xmax=.1,color='black',lw=2)
	# ax.hlines(np.mean(pdist.vmPFC),xmin=0,xmax=.3,color='black',lw=2)



#control
# fig,ax=plt.subplots()
# sns.lmplot(x='ev',y='vmPFC',data=crmod,hue='group',palette=[wes_palettes['Zissou'][0]])
# clm = lm(crmod.ev,crmod.vmPFC)
# plt.title('Control r=%s p=%s'%(clm.rvalue.round(4),clm.pvalue.round(4)))
# #PTSD
# fig,ax=plt.subplots()
# sns.lmplot(x='ev',y='vmPFC',data=prmod,hue='group',palette=[wes_palettes['Royal1'][1]])
# plm = lm(prmod.ev,prmod.vmPFC)
# plt.title('PTSD r=%s p=%s'%(plm.rvalue.round(4),plm.pvalue.round(4)))

#
Vcorr = pd.DataFrame({
    'subject':np.tile(all_sub_args,2),
    'group':np.tile(rmod.group.values,2),
    'phase':np.repeat(['ext','rnw'],48),
    'ev':np.tile(rmod.ev.values,2),
    'beta':np.concatenate([rmod.ext_vmPFC.values,rmod.vmPFC.values])
    })
fig, ax = plt.subplots()
ax = sns.lmplot(x='ev',y='beta',hue='group',col='phase',data=Vcorr,
                palette=[wes_palettes['Zissou'][0],wes_palettes['Royal1'][1]],
                sharey=False)
ax.set(xlim=(0,.8))
#JK i don't like this so lets do it differently by phase
rlm = sns.lmplot(x='ev',y='vmPFC',hue='group',data=rmod,legend=False,
    ci=None,truncate=True,scatter_kws={'edgecolors':'black','alpha':1},
    line_kws={'path_effects':[PathEffects.withStroke(linewidth=5,foreground='black')]},
    palette=[wes_palettes['Zissou'][0],wes_palettes['Royal1'][1]])
rlm.set(xlim=(0,.8),ylim=(-60,40),xticks=[0,.2,.6,.4,.8])
#day1
elm = sns.lmplot(x='ext_vmPFC',y='ev',hue='group',data=rmod,legend=False,
    ci=None,truncate=True,scatter_kws={'edgecolors':'black','alpha':1},
    line_kws={'path_effects':[PathEffects.withStroke(linewidth=5,foreground='black')]},
    palette=[wes_palettes['Zissou'][0],wes_palettes['Royal1'][1]])
elm.set(xlim=(-20,20),ylim=(0,.8),yticks=[0,.2,.6,.4,.8])
######relationship between groups####

betamod = pd.DataFrame({
                        'group': np.repeat(('control','ptsd'),24),
                        'fear_vmPFC': fb.CSp_CSm[fb.roi == 'mOFC_beta'].values,
                        'fear_HC': fb.CSp_CSm[fb.roi == 'hippocampus_beta'].values,
                        'fear_amyg': fb.CSp_CSm[fb.roi == 'amygdala_beta'].values,
                        'ext_vmPFC': eb.CSp_CSm[eb.roi == 'mOFC_beta'].values,
                        'ext_HC': eb.CSp_CSm[eb.roi == 'hippocampus_beta'].values,
                        'ext_amyg': eb.CSp_CSm[eb.roi == 'amygdala_beta'].values,
                        'rnw_vmPFC': rb.early_CSp_CSm[rb.roi == 'mOFC_beta'].values,
                        'rnw_HC': rb.early_CSp_CSm[rb.roi == 'hippocampus_beta'].values,
                        'rnw_amyg': rb.early_CSp_CSm[rb.roi == 'amygdala_beta'].values,
})
fig, bax = plt.subplots(3,3,sharey=True)
for r, roi in enumerate(['vmPFC','HC','amyg']):
    for i, phase in enumerate(['fear','ext','rnw']):
        barstrip(betamod,'%s_%s'%(phase,roi),points=None,ax=bax[r,i])
        bax[r,i].hlines(y=0,xmin=bax[r,i].get_xlim()[0],xmax=bax[r,i].get_xlim()[1],lw=2)
        sns.despine(ax=bax[r,i],bottom=True)
for val in betamod.columns:
    if val is not 'group':
	    fig, ax = plt.subplots()
	    sns.boxplot(x='group',y=val,data=betamod,palette=gpal,ax=ax)
	    # sns.barplot(x='group',y=val,data=rmod,palette=gpal,ax=ax,ci=None)
	    # sns.stripplot(x='group',y=val,data=rmod,palette=gpal,ax=ax,linewidth=2,edgecolor='black',size=8)
	    # for i, group in enumerate(rmod.group.unique()):
	    #     ax.errorbar(i,rmod_avgs[val][group],yerr=rmod_sems[val][group],color='black',linewidth=3)
	    sns.despine(ax=ax)
	    t, p = pg.ttest(betamod[val][betamod.group=='control'],betamod[val][betamod.group=='ptsd'])[['T','p-val']].loc['T-test']
	    ax.set_title('t = %s; p = %s'%(np.round(t,4),np.round(p,4)))
	    plt.tight_layout()
####corr graphing
def corr_plot(dfs):
    gs1 = gridspec.GridSpec(2,2,height_ratios=[.9,.05],hspace=.5)
    ax1 = plt.subplot(gs1[0, 0]);ax2 = plt.subplot(gs1[0, 1]);ax3 = plt.subplot(gs1[1, 0:2]);ax = [ax1,ax2,ax3]
    for i, df in enumerate(dfs):
        group = df.group.unique()[0]
        data = pd.concat([df.ev,df.vmPFC,df.HC,df.amyg],axis=1)
        corr = np.corrcoef(data,rowvar=False)
        p_vals = np.zeros((4,4));p_vals[np.tril_indices_from(p_vals,k=-1)] = pg.pairwise_corr(data)['p-unc'].values
        p_vals[3,0], p_vals[2,1] = p_vals[2,1], p_vals[3,0] #numpy fills in the rows first and not the columns so we have to switch 2 values
        p_text = p_vals.astype(str)
        p_text[np.where(p_vals > .05)] = '';p_text[np.where(p_vals < .05)] = '*';p_text[np.where(p_vals < .01)] = '**';p_text[np.where(p_vals < .001)] = '***'
        mask = np.zeros((4,4));mask[np.triu_indices_from(mask)] = True;mask[np.diag_indices_from(mask)] = True
        xlabels = ['Context','vmPFC','HC',''];ylabels = ['','vmPFC','HC','Amygdala']
        # fig, (ax, cbar_ax) = plt.subplots(2, gridspec_kw={'height_ratios':(.9,.05),'hspace':.5})
        cbar_ax = ax[2]
        Cmap = pal.cmocean.diverging.Curl_4.mpl_colormap
        # Cmap = sns.blend_palette(wes_palettes['Zissou'],n_colors=5,as_cmap=True)
        with sns.axes_style('white'):
            sns.heatmap(corr,mask=mask,fmt='',square=True,cmap='PRGn',center=0,vmin=-.5,vmax=1,
                        xticklabels=xlabels,yticklabels=ylabels,linewidths=.7,annot=p_text,
                        ax=ax[i],cbar_ax=cbar_ax,cbar_kws={'orientation':'horizontal'})
        ax[i].set_xticklabels(ax[i].get_xticklabels(),rotation=0);ax[i].set_yticklabels(ax[i].get_yticklabels(),rotation=0)
        ax[i].set_title('%s Subjects'%(group))
    cbar_ax.set_title("Pearson's r")
corr_plot([crmod,prmod])
###########################################
#RSA subject
# for roi in ['mOFC','dACC','amygdala','insula','hippocampus']:
roi = 'PPA'
label = 'e__e_r'
print(roi)
p = pd.read_csv(os.path.join(data_dir,'graphing','beta_rsa','%s_csp.csv'%(roi)))
m = pd.read_csv(os.path.join(data_dir,'graphing','beta_rsa','%s_csm.csv'%(roi)))


ep = p[p.label == 'e__e_r']
em = m[m.label == 'e__e_r']

fp = p[p.label == 'f__e_r']
fm = m[m.label == 'f__e_r']

ep.reset_index(inplace=True,drop=True)
em.reset_index(inplace=True,drop=True)
fp.reset_index(inplace=True,drop=True)
fm.reset_index(inplace=True,drop=True)

rsa = pd.DataFrame({
    'c_e_p': ep.z[ep.group == 'control'].values,
    'c_e_m': em.z[ep.group == 'control'].values,
    'c_f_p': fp.z[ep.group == 'control'].values,
    'c_f_m': fm.z[ep.group == 'control'].values,
    'p_e_p': ep.z[ep.group == 'ptsd'].values,
    'p_e_m': em.z[ep.group == 'ptsd'].values,
    'p_f_p': fp.z[ep.group == 'ptsd'].values,
    'p_f_m': fm.z[ep.group == 'ptsd'].values
    })
    
print('FEAR',ttest_ind(rsa.c_f_p, rsa.p_f_p ))
print('EXT',ttest_ind(rsa.c_e_p ,rsa.p_e_p))
rsa.to_csv(os.path.join(data_dir,'graphing','beta_RSA','%s_collected_rsa.csv'%(roi)),index=False)
rsa_comp = pd.DataFrame({'comp':ep.z - em.z,
                'group':ep.group,
                'sub':ep['sub']})
mOFC_out = [5,23,103,107,116]
rsa_comp = rsa_comp.set_index(['sub'])
for sub in mOFC_out:rsa_comp.drop(index=sub,axis=0,inplace=True)
rsa_comp.reset_index(inplace=True)
sns.swarmplot(x='group',y='comp',data=rsa_comp,dodge=True)
####################################################
#better rsa graphing
# for roi in ['mOFC','dACC','amygdala','insula','hippocampus','PPA']:
for roi in ['hippocampus','PPA']:
# for roi in ['mOFC']:
    mR = pd.read_csv(os.path.join(data_dir,'graphing','beta_RSA','%s_collected_rsa.csv'%(roi)))
    mR = pd.DataFrame({
        'z': np.concatenate((mR.c_e_p,mR.c_e_m,mR.p_e_p,mR.p_e_m)),
        'group': np.repeat(['control','ptsd'],48),
        'condition': np.tile(np.repeat(['CS+','CS-'],24),2),
        'subject': np.tile([sub_args,p_sub_args],2).flatten()
        })
    barstrip(mR,'z',cond=True,points='strip')
    anova = pg.mixed_anova(dv='z',subject='subject',within='condition',between='group',data=mR)
    posthoc = pg.pairwise_ttests(dv='z',subject='subject',within='condition',between='group',data=mR)

    print(roi,'\n',anova,'\n',posthoc)
#####################################################
#Survey data
survey = pd.read_csv(data_dir+'Demographics_Survey/summary_stats.csv')
for val in ['sex','ethnicity']:
	fig, ax = plt.subplots()
	sns.countplot(x=val,hue='group',data=survey,palette=gpal,ax=ax)
	sns.despine(ax=ax);ax.legend_.remove();plt.tight_layout()
for val in ['IUS','SAI_pre','BDI','BAI_pre']:
    barstrip()
    T, p = np.round(pg.ttest(survey[val][survey.group == 'control'], survey[val][survey.group == 'ptsd'])[['T','p-val']].values[0],4)
	print(val, T, p)

barstrip(survey,['SAI_pre','BDI','BAI_pre','IUS','age'])
barstrip(survey,['age'],)

d1p = pd.read_csv(data_dir+'Demographics_Survey/day1_post_response.csv')
barstrip(d1p,['n_shock','intensity'])

d1p.sort_values(['shock_word'],inplace=True)
fig, ax= plt.subplots()
sns.countplot(x='shock_word',hue='group',data=d1p,ax=ax,palette=gpal);sns.despine(ax=ax);ax.legend_.remove()


words = pd.read_csv(data_dir+'Demographics_Survey/day1_post_words.csv')
# barstrip(words,words.columns)

w_avg = words.groupby(['word','group']).mean()
w_err = words.groupby(['word','group']).sem()
fig, ax = plt.subplots()
sns.barplot(x='word',y='rating',hue='group',data=words,palette=gpal,ax=ax,ci=None)
for i, word in enumerate(words.word.unique()):
	ax.errorbar(i-.2,w_avg.loc[(word,'control'),'rating'],yerr=w_err.loc[(word,'control'),'rating'],
				color='black',linewidth=3,capsize=5,capthick=3)
	ax.errorbar(i+.2,w_avg.loc[(word,'ptsd'),'rating'],yerr=w_err.loc[(word,'ptsd'),'rating'],
				color='black',linewidth=3,capsize=5,capthick=3)
ax.set_ylim(1,4);sns.despine(ax=ax);ax.set_yticks([1,2,3,4]);ax.legend_.remove()


pcl = pd.read_csv(data_dir+'Demographics_Survey/pcl_part3.csv')
pcl = pcl.drop(np.where(pcl.subject == 124)[0])

total = pcl.groupby(['subject']).sum();total['group'] = 'ptsd';total=total.drop(columns='question').reset_index()
pcl_avg = total.mean()['score']
pcl_sem = total.sem()['score']

#105 and 120 did not self report PTSD
tswarm = total.set_index(['subject']).copy()
tswarm['diagnosis'] = 'PTSD'
for sub in [105,120]:tswarm.loc[(sub,'diagnosis')] = 'Anxiety_OCD'
tswarm.reset_index(inplace=True)


fig, ax = plt.subplots()
sns.barplot(x='group',y='score',data=total,palette=[gpal[1]],ax=ax,ci=None)
sns.swarmplot(x='group',y='score',data=tswarm,hue='diagnosis',palette=[gpal[1],'darkblue'],ax=ax,linewidth=2,edgecolor='black',size=8)
ax.errorbar(0-.25,pcl_avg,yerr=pcl_sem,color='black',linewidth=3,capsize=5,capthick=3)
ax.hlines(33,ax.get_xlim()[0],ax.get_xlim()[1],linestyle='--');sns.despine(ax=ax);ax.legend_.remove()

#average block score
bscore = pcl.groupby(['subject','DSM_block']).mean().reset_index().drop(columns='question')
b_avg = bscore.groupby(['DSM_block']).mean().drop(columns='subject')
b_sem = bscore.groupby(['DSM_block']).sem().drop(columns='subject')

fig, ax = plt.subplots()
sns.barplot(x='DSM_block',y='score',data=bscore,ci=None,ax=ax,
			palette=sns.light_palette(gpal[1],6)[2:])
sns.swarmplot(x='DSM_block',y='score',data=bscore,ax=ax,linewidth=2,edgecolor='black',size=8,
			palette=sns.light_palette(gpal[1],6)[2:])
for i, B in enumerate(bscore.DSM_block.unique()):
	ax.errorbar(i-.25,b_avg.loc[B,'score'],yerr=b_sem.loc[B,'score'],color='black',linewidth=3,capsize=5,capthick=3)
sns.despine(ax=ax)

#criteria met
subs = pcl.subject.unique()
blocks = ['B','C','D','E','PTSD']
count = pcl.set_index(['subject','DSM_block']).sort_index().copy()
crit = pd.DataFrame([],columns=['crit'],index=pd.MultiIndex.from_product(
					[subs,blocks],names=['subject','DSM_block']))
for sub in subs:
	for b in blocks:
		if b in ['B','C']: c = np.int(np.where(count.loc[(sub,b),'score'] >= 2)[0].shape[0] > 1)
		if b in ['D','E']: c = np.int(np.where(count.loc[(sub,b),'score'] >= 2)[0].shape[0] > 2)
		if b is 'PTSD':    c = np.int(crit.loc[sub].sum() == 4)
		crit.loc[(sub,b)] = c
crit.reset_index(inplace=True)
fig, ax = plt.subplots()
sns.barplot(x='DSM_block',y='crit',data=crit,estimator=sum,
			palette=sns.light_palette(gpal[1],6)[1:],ax=ax,ci=None)
sns.despine(ax=ax)




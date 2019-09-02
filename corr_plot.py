import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pingouin as pg
from fc_config import *
from wesanderson import wes_palettes
from nilearn.input_data import NiftiMasker
from glm_timing import glm_timing
from mvpa_analysis import group_decode
from signal_change import collect_ev

def corr_plot(df,group):
    cgc = pd.concat([df.ev,df.vmPFC,df.HC,df.amyg],axis=1)
    c_corr = np.corrcoef(cgc,rowvar=False)
    cp_vals = np.zeros((4,4))
    cp_vals[np.tril_indices_from(cp_vals,k=-1)] = pg.pairwise_corr(cgc)['p-unc'].values
    cp_vals[3,0], cp_vals[2,1] = cp_vals[2,1], cp_vals[3,0] #numpy fills in the rows first and not the columns so we have to switch 2 values
    cp_text = cp_vals.astype(str)
    cp_text[np.where(cp_vals > .05)] = ''
    cp_text[np.where(cp_vals < .05)] = '*'
    cp_text[np.where(cp_vals < .01)] = '**'
    cp_text[np.where(cp_vals < .001)] = '***'
    mask = np.zeros((4,4))
    mask[np.triu_indices_from(mask)] = True
    mask[np.diag_indices_from(mask)] = True
    xlabels = ['Context','vmPFC','HC','']
    ylabels = ['','vmPFC','HC','Amygdala']
    fig, (cax, cbar_cax) = plt.subplots(2, gridspec_kw={'height_ratios':(.9,.05),'hspace':.5})
    cax = sns.heatmap(c_corr,mask=mask,annot=cp_text,fmt='',square=True,cmap='PRGn',center=0,xticklabels=xlabels,yticklabels=ylabels
                    ,ax=cax,cbar_ax=cbar_cax,cbar_kws={'orientation':'horizontal'})
    cax.set_xticklabels(cax.get_xticklabels(),rotation=0)
    cax.set_yticklabels(cax.get_yticklabels(),rotation=0)

    cax.set_title('%s Subjects'%(group))
    cbar_cax.set_title("Pearson's r")
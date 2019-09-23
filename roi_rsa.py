from fc_config import *
from preprocess_library import meta

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg


class roi_rsa():

    def __init__(self,group='control'):
        
        self.group = group
        
        if   group == 'control': self.subs = sub_args;
        elif group == 'ptsd':    self.subs = p_sub_args
        elif group == 'between': self.subs = all_sub_args

        self.rois = ['hippocampus','mOFC','dACC','insula','amygdala']
        self.load_rois()
    
    def load_rois(self):

        df = {}
        df_dir = os.path.join(data_dir,'group_ER','roi_dfs')
        
        for roi in self.rois: df[roi] = pd.read_csv(os.path.join(df_dir,'%s_group_stats.csv'%(roi)))

        self.df = pd.concat(df.values())
        self.df.encode = pd.Categorical(self.df.encode,['baseline','fear_conditioning','extinction'],ordered=True)
        self.df.roi = pd.Categorical(self.df.roi,self.rois,ordered=True)
        # self.df = pd.DataFrame.from_dict({(roi)}:df[roi])
        # df = pd.DataFrame.from_dict({(group,label,i): group_stats[group][label][i]

    def graph(self):
        if self.group == 'between': df = self.df #this doesn't really make sense with how i plan to graph the results
        else: df = self.df.query('group == @self.group')
        df = df.groupby(['trial_type','encode','roi','subject']).mean()
        paired = df.loc['CS+'] - df.loc['CS-']
        paired.reset_index(inplace=True);df.reset_index(inplace=True)
        
        emo_paired = self.df.query('group == @self.group')
        emo_paired = emo_paired.groupby(['encode','trial_type','roi','subject']).mean()
        emo_paired = emo_paired.loc['fear_conditioning'] - emo_paired.loc['extinction']
        emo_paired.reset_index(inplace=True)

        sns.set_context('talk');sns.set_style('ticks')
        # fig, ax = plt.subplots(3,1,sharey=True)
        # for i, phase in enumerate(df.encode.unique().categories):
            
            #CS+ and CS- pointplot
            # sns.pointplot(data=df.query('encode == @phase'),x='roi',y='rsa',hue='trial_type',
            #               palette=cpal,capsize=.08,join=False,dodge=True,ax=ax[i])

            #CS+ and CS- swarmplot
            # sns.swarmplot(data=df.query('encode == @phase'),x='roi',y='rsa',hue='trial_type',
            #               palette=cpoint,linewidth=2,edgecolor='black',size=8,ax=ax[i])
            
            #CS+ and CS- boxplot
            # sns.boxplot(data=df.query('encode == @phase'),x='roi',y='rsa',hue='trial_type',
            #             palette=cpal,dodge=True,ax=ax[i])
            
            # ax[i].set_title('Phase = %s'%(phase))
            # ax[i].legend_.remove()

        #Paired, CS+ - CS- pointplot
        phase_pal = sns.color_palette(['black','darkmagenta','seagreen'],desat=.75)
        fig, ax = plt.subplots()
        sns.pointplot(data=paired,x='roi',y='rsa',hue='encode',
                      palette=phase_pal,capsize=.08,join=False,dodge=True,ax=ax)
        ax.hlines(y=0,xmin=ax.get_xlim()[0],xmax=ax.get_xlim()[1])
        plt.title('Relative RSA\n\n[CS+ - CS-]')
        sns.despine(ax=ax)


        fig, ax = plt.subplots()
        sns.pointplot(data=emo_paired,x='roi',y='rsa',hue='trial_type',
                      palette=cpal,dodge=True,capsize=.08,join=False,ax=ax)
        ax.hlines(y=0,xmin=ax.get_xlim()[0],xmax=ax.get_xlim()[1])
        plt.title('Relative RSA\n\n[Conditioning - Extinction]')
        sns.despine(ax=ax)
    # pp2 = sns.pointplot(x='label',y='z',data=df2,hue='group',
    #         capsize=.08,join=False,dodge=True)
# pplot(x='group',y='z',hue='condition',data=df,ax=ax,palette=cpoint,dodge=True,
#                                        linewidth=2,edgecolor='black',size=8)
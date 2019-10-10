import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
import nibabel as nib

from collections import OrderedDict
from nilearn import image
from nilearn.input_data import NiftiMasker
from scipy.stats import pearsonr

from fc_config import *
from preprocess_library import meta
from glm_timing import glm_timing

class roi_rsa():
    def __init__(self,sub=None):

        self.subj = meta(sub)
        self.verbose=False

        #set the phases
        self.encoding_phases = ['baseline','fear_conditioning','extinction']
        self.mem_phases = ['memory_run_1','memory_run_2','memory_run_3']
        self.phases = self.encoding_phases + self.mem_phases
        
        #hardcode copes as weights (as opposed to zmaps)
        weights = 'copes'
        self.weight_dir = os.path.join(data_dir,'rsa_%s'%(weights),self.subj.fsub)
        #need these to load the weights becauase I named them lazily
        self.conditions = {'CS+': 'csp',
                           'CS-': 'csm'}
        self.w_phases = {'baseline'         : 'baseline',
                         'fear_conditioning': 'fear',
                         'extinction'       : 'ext'}
        
        #hardcode rois for now
        self.rois = ['hippocampus','mOFC','dACC','insula','amygdala']
        
        #data needs to be loaded WITHOUT mask to facilitate more intricate analyses
        self.load_data() 
        self.compute_item_rsa()

    def load_data(self):

        # mask_img = nib.load(os.path.join(self.subj.roi,'%s_mask.nii.gz'%(self.mask)))
    
        # self.masker = NiftiMasker(mask_img=mask_img, standardize=False, memory='nilearn_cache',memory_level=2)
        # self.masker.fit(ref)
        # self.W_masker = NiftiMasker(mask_img=mask_img)
    
        #load reference vol
        ref = nib.load(self.subj.refvol_be)        
        
        #encoding data & labels
        self.encoding_data = OrderedDict()
        self.encoding_labels = OrderedDict()

        #ls-u style beta weights
        self.W = OrderedDict()
        
        #retrieva data & labels
        self.mem_data = OrderedDict()
        self.mem_labels = OrderedDict()

        for phase in self.phases:

            beta_img = nib.load(os.path.join(self.subj.bold_dir, fsl_betas[phase]))

            if phase in self.encoding_phases:
                events = glm_timing(self.subj.num,phase).phase_events(stims=True)
                for cs in self.conditions:
                    _con = self.conditions[cs]+'_'+'trial'
                    events[_con] = ''
                    events.loc[np.where(events.trial_type==cs,)[0],_con] = [i for i in range(1,25)]
                    # [i[0] + '_' + str(i[1]) for i in zip(np.repeat('CS+',24),range(1,25))]
                    

                events['phase'] = phase #this isn't in the dataframe yet
                #save labels & data
                self.encoding_labels[phase] = events
                self.encoding_data[phase] = beta_img
                
                #load in the weights here
                self.W[phase] = OrderedDict()
                for con in self.conditions:
                    self.W[phase][con] = nib.load(
                            os.path.join(
                            self.weight_dir,'%s_%s.nii.gz'%
                            (self.w_phases[phase],self.conditions[con])))
        
            elif phase in self.mem_phases:
                events = glm_timing(self.subj.num,phase).mem_events(stims=True)
                events['phase'] = phase #add phase to df
                #save labels & data
                self.mem_labels[phase] = events
                self.mem_data[phase] = beta_img
        
        #get 1 dataframe & image for encoding
        self.encoding_labels = pd.concat(self.encoding_labels.values())
        self.encoding_labels.reset_index(inplace=True,drop=True) #need this bc phase events are all numbered the same
        self.encoding_data = nib.concat_images(self.encoding_data.values(),axis=3)
        #same for retrieval, except we have to remove foils
        self.mem_labels = pd.concat(self.mem_labels.values())
        foil_mask = self.mem_labels.memcond.isin(['Old'])
        self.mem_labels = self.mem_labels[foil_mask].reset_index(drop=True)
        self.mem_data = image.index_img(nib.concat_images(self.mem_data.values(),axis=3), foil_mask)

        #we need this for the roi bootstrapping, probably best to do it here and broadcast
        self.dACC_nvox = np.where( nib.load(os.path.join(self.subj.roi,'dACC_mask.nii.gz')).get_data() == 1 )[0].shape[0]

    def apply_mask(self,roi=None,target=None):
        #pass in an roi and a nifti image to return the data located in the roi mask
        mask = nib.load(os.path.join(self.subj.roi,'%s_mask.nii.gz'%(roi)))
        coor = np.where(mask.get_data() == 1)
        values = target.get_data()[coor]
        if values.ndim > 1:
            values = np.transpose(values) #swap axes to get feature X sample
        return values

    def boot_rsa(self,encoding_trial,mem_trial):
        assert encoding_trial.shape == mem_trial.shape
        n_boot = 1000
        boot_res = np.zeros(n_boot)
        for i in range(n_boot):
            _samp       = np.random.randint(low=0, high=encoding_trial.shape[0], size=self.dACC_nvox)
            boot_res[i] = np.arctanh( pearsonr( encoding_trial[_samp], mem_trial[_samp] )[0] )
        return boot_res.mean()

    def compute_item_rsa(self):
    
        self.rsa = self.mem_labels.copy().drop(columns=['onset','duration'])
        for cs in self.conditions: self.rsa[self.conditions[cs]+'_'+'trial'] = ''
        for roi in self.rois:
            self.rsa[roi] = 0 #going by columns and then melting at the end
            
            #apply the mask to everything for this roi
            encoding_data = self.apply_mask(roi,self.encoding_data)
            mem_data      = self.apply_mask(roi,self.mem_data)
            W = {}
            for phase in self.encoding_phases:
                W[phase] = {}
                for con in self.conditions:
                    W[phase][con] = self.apply_mask(roi,self.W[phase][con])
            
            #great, you've got everything in the mask shape, now run the item rsa
            for i, stim in enumerate(self.rsa.stim):
                mem_loc = i
                encoding_loc = np.where(self.encoding_labels.stim == stim)[0][0]
                
                _phase      = self.rsa.loc[mem_loc,'encode']
                _trial_type = self.rsa.loc[mem_loc,'trial_type']
                _con        = self.conditions[_trial_type]+'_'+'trial'
                self.rsa.loc[i,_con] = self.encoding_labels.loc[encoding_loc,_con] 
                
                encoding_trial = encoding_data[encoding_loc] * W[_phase][_trial_type]
                mem_trial      = mem_data[mem_loc] * W[_phase][_trial_type]

                # p = pearsonr(encoding_trial,mem_trial)[0]
                # z = np.arctanh(p)
                self.rsa.loc[i,roi] = self.boot_rsa(encoding_trial,mem_trial)

        self.rsa['subject'] = self.subj.num
        self.rsa = self.rsa.melt(id_vars=['subject','trial_type','stim','memcond','encode','response',
                               'acc','hc_acc','phase','csp_trial','csm_trial'],
                      value_vars=self.rois
                  ).rename(columns={'variable':'roi', 'value':'rsa'})

        #lets label the first 8 trials of extinction as "early_extinction"
        self.rsa.to_csv(os.path.join(self.subj.rsa,'roi_ER.csv'),index=False)



class group_roi_rsa():

    def __init__(self,group='control',ext_split=True):
        
        self.group = group
        self.ext_split = ext_split

        if   group == 'control': self.subs = sub_args;
        elif group == 'ptsd':    self.subs = p_sub_args
        elif group == 'between': self.subs = all_sub_args

        self.rois = ['hippocampus','mOFC','dACC','insula','amygdala']
        self.conditions = {'CS+': 'csp',
                           'CS-': 'csm'}
        
        if self.ext_split: self.encoding_phases = ['baseline','fear_conditioning','early_extinction','extinction']
        else:              self.encoding_phases = ['baseline','fear_conditioning','extinction']

        self.load_rois()
    
    def load_rois(self):

        df = {}
        # df_dir = os.path.join(data_dir,'group_ER','roi_dfs')        
        # for roi in self.rois: df[roi] = pd.read_csv(os.path.join(df_dir,'%s_group_stats.csv'%(roi)))

        for sub in self.subs:
            subj = meta(sub)
            df[sub] = pd.read_csv(os.path.join(subj.rsa,'roi_ER.csv'))

        self.df = pd.concat(df.values()).reset_index(drop=True)
        #lets label the first 8 trials of extinction as "early_extinction"
        if self.ext_split:
            for cs in self.conditions:
                con = '%s_trial'%(self.conditions[cs])
                for i in range(1,9): 
                    self.df.loc[ self.df[ self.df.encode == 'extinction' ][ self.df[con] == i ].index,'encode' ] = 'early_extinction'
        
        self.df.encode = pd.Categorical(self.df.encode,self.encoding_phases,ordered=True)
        self.df.roi = pd.Categorical(self.df.roi,self.rois,ordered=True)

    def graph(self):
        # if self.group == 'between': df = self.df #this doesn't really make sense with how i plan to graph the results
        # else: df = self.df.query('group == @self.group')
        df = self.df.groupby(['trial_type','encode','roi','subject']).mean()
        paired = df.loc['CS+'] - df.loc['CS-']
        paired.reset_index(inplace=True);df.reset_index(inplace=True)
        
        # emo_paired = self.df.query('group == @self.group')
        emo_paired = self.df.groupby(['encode','trial_type','roi','subject']).mean()
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
        if self.ext_split: phase_pal = sns.color_palette(['black','darkmagenta','lightgreen','seagreen'],desat=.75)
        else:              phase_pal = sns.color_palette(['black','darkmagenta','lightgreen','seagreen'],desat=.75)
        
        fig, ax = plt.subplots()
        sns.pointplot(data=paired,x='roi',y='rsa',hue='encode',palette=phase_pal
                      ,capsize=.08,join=False,dodge=True,ax=ax)
        ax.hlines(y=0,xmin=ax.get_xlim()[0],xmax=ax.get_xlim()[1])
        plt.title('Relative RSA\n\n[CS+ - CS-]')
        sns.despine(ax=ax)

        fig, ax = plt.subplots()
        sns.pointplot(data=emo_paired,x='roi',y='rsa',hue='trial_type',
                      palette=cpal,dodge=True,capsize=.08,join=False,ax=ax)
        ax.hlines(y=0,xmin=ax.get_xlim()[0],xmax=ax.get_xlim()[1])
        plt.title('Relative RSA\n\n[Conditioning - Extinction]')
        sns.despine(ax=ax)

def vox_count():
    vox = {}
    for sub in all_sub_args:
        subj = meta(sub)
        vox[sub] = {}
        for roi in ['hippocampus','mOFC','dACC','insula','amygdala']:
            mask = nib.load(os.path.join(subj.roi,'%s_mask.nii.gz'%(roi)))
            vox[sub][roi] = np.where(mask.get_data() == 1)[0].shape[0]
    df = pd.DataFrame.from_dict(vox,orient='index'
                    ).reset_index(
                    ).melt(id_vars='index'
                    ).rename(columns={'index':'subject', 'variable':'roi', 'value':'nvox'})
    fig, ax = plt.subplots()
    sns.boxplot(data=df,x='roi',y='nvox',ax=ax)
    sns.despine(ax=ax,trim=True)
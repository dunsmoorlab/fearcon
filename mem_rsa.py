import os
import numpy as np
import pandas as pd
import seaborn as sns
import pingouin as pg
from collections import OrderedDict

from fc_config import *
from wesanderson import wes_palettes
from preprocess_library import meta
from glm_timing import *
from mvpa_analysis import *
from nilearn.input_data import NiftiMasker
from nilearn import image

from nilearn.decoding import SearchLight
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold
from sklearn.linear_model import LogisticRegression
from nilearn.image import index_img, new_img_like
from predictive_searchlight import Predictive_SearchLight
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
import inspect
from sklearn.metrics import confusion_matrix


class mem_rsa():
    def __init__(self,sub=None,mask=None):

        self.subj = meta(sub)
        self.verbose=False

        self.encoding_phases = ['baseline','fear_conditioning','extinction']
        self.mem_phases = ['memory_run_1','memory_run_2','memory_run_3']
        self.phases = self.encoding_phases + self.mem_phases
        weights = 'copes'
        self.weight_dir = os.path.join(data_dir,'rsa_%s'%(weights),self.subj.fsub)

        self.mask = mask
        self.load_data()
        # self.compute_item_rsa()

    def load_data(self):

        mask_img = nib.load(os.path.join(self.subj.roi,'%s_mask.nii.gz'%(self.mask)))
        ref = nib.load(self.subj.refvol_be)
        self.masker = NiftiMasker(mask_img=mask_img, standardize=False, memory='nilearn_cache',memory_level=2)
        self.masker.fit(ref)
        # self.W_masker = NiftiMasker(mask_img=mask_img)
        
        self.encoding_data = OrderedDict()
        self.encoding_labels = OrderedDict()
        
        self.mem_data = OrderedDict()
        self.mem_labels = OrderedDict()
        # self.split_data = {}
        # self.sub_dir = os.path.join(self.weight_dir,self.subj.fsub)
        for phase in self.phases:

            beta_img = nib.load(os.path.join(self.subj.bold_dir, fsl_betas[phase]))

            if phase in self.encoding_phases:

                events = glm_timing(self.subj.num,phase).phase_events(stims=True)
                events['phase'] = phase
                self.encoding_labels[phase] = events
                
                self.encoding_data[phase] = beta_img
            
            elif phase in self.mem_phases:

                events = glm_timing(self.subj.num,phase).mem_events(stims=True)

                foil_mask = events.memcond.isin(['Old'])

                events['phase'] = phase
                self.mem_labels[phase] = events
            
                self.mem_data[phase] = beta_img


        self.encoding_labels = pd.concat(self.encoding_labels.values())
        self.encoding_labels.reset_index(inplace=True,drop=True)
        self.encoding_data = self.masker.fit_transform(nib.concat_images(self.encoding_data.values(),axis=3))
        
        self.mem_labels = pd.concat(self.mem_labels.values())
        foil_mask = self.mem_labels.memcond.isin(['Old'])
        self.mem_labels = self.mem_labels[foil_mask].reset_index(drop=True)
        self.mem_data = self.masker.fit_transform(image.index_img(nib.concat_images(self.mem_data.values(),axis=3), foil_mask))


    def compute_item_rsa(self):
        self.rsa_ = self.mem_labels.copy()

        self.rsa_.drop(columns=['onset','duration'],inplace=True)
        self.rsa_['rsa'] = 0
        self.rsa_['roi'] = self.mask

        for i, stim in enumerate(self.mem_labels.stim):
            mem_loc = i
            encoding_loc = np.where(self.encoding_labels.stim == stim)[0][0]

            r = np.corrcoef(self.encoding_data[encoding_loc],self.mem_data[mem_loc])[0,1]
            z = np.arctanh(r)

            self.rsa_.loc[i,'rsa'] = z

        self.rsa_.to_csv(os.path.join(self.subj.rsa,'%s_ER.csv'%(self.mask)),index=False)

    def cross_rsa(self):

        #do some dataframe work to sort everything correctly while preserving the encode_location
        self.encoding_labels.phase = pd.Categorical(self.encoding_labels.phase,['baseline','fear_conditioning','extinction'])
        self.encoding_labels.trial_type = pd.Categorical(self.encoding_labels.trial_type,['CS+','CS-'])
        self.encoding_labels.sort_values(by=['phase','trial_type'],inplace=True)
        self.encoding_labels.reset_index(inplace=True)
        self.encoding_labels.rename(columns={'index':'encode_loc'},inplace=True)

        #do the same thing for the memory items using the encoding labels as reference
        eloc = lambda x: np.where(self.encoding_labels.stim == x)[0][0]
        self.mem_labels['rsa_order'] = self.mem_labels.stim.apply(eloc)
        self.mem_labels.sort_values(by='rsa_order',inplace=True)
        self.mem_labels.reset_index(inplace=True)
        self.mem_labels.rename(columns={'index':'mem_loc'},inplace=True)

        self.cross_mat = np.zeros((144,144)) #remember [row,column]
        for r in range(144):
            Eitem = self.encoding_data[self.encoding_labels.loc[r,'encode_loc']]
            for c in range(144):
                Mitem = self.mem_data[self.mem_labels.loc[c,'mem_loc']]
                self.cross_mat[r,c] = np.arctanh(np.corrcoef(Eitem,Mitem)[0,1])



    def phase_clf(self):
        cope_type = 'csp_csm'
        copes = ['baseline_%s'%(cope_type),'fear_%s'%(cope_type),'ext_%s'%(cope_type)]
        self.cope_data = {}
        for i, phase in enumerate(self.encoding_phases):
            self.cope_data[phase] = self.masker.fit_transform(nib.load(os.path.join(self.weight_dir,copes[i] + '.nii.gz')))

        #as a first pass lets just focus on CS+ images
        csp = np.where(self.mem_labels.trial_type == 'CS+')[0]

        self.clf_out = pd.DataFrame(columns=['encode','guess','baseline','fear_conditioning','extinction'],index=range(csp.shape[0]))
        for i, trial in enumerate(csp):
            self.clf_out.loc[i,'encode'] = self.mem_labels.encode[trial]
            
            for phase in self.encoding_phases:
                self.clf_out.loc[i,phase] = np.corrcoef(self.mem_data[trial],self.cope_data[phase])[0,1]

            self.clf_out.loc[i,'guess'] = self.clf_out.columns[self.clf_out.loc[i] == self.clf_out.loc[i,self.encoding_phases].max()][0]

        self.clf_out.to_csv(os.path.join(self.subj.rsa,'%s_%s_clf.csv'%(self.mask,cope_type)),index=False)

class RSA_estimator(BaseEstimator, ClassifierMixin): #ClassifierMixin most likely interchangeable
    def __init__(self,subject=None):
        
        self.subject = subject #sklearn requires some arg

    def fit(self,X,y=None,*args,**kwargs):

        self.X_train = X #grab half the data for "training"
        self.y_train = y #print this for validation check
        return self

    def score(self,X,y=None):
                
        r = np.corrcoef(self.X_train,X)[0,1] #Compute RSA here
        z = np.arctanh(r) #Convert to fischer-z
        return z        

class ER_searchlight():
    def __init__(self,sub,process_mask=None):
        self.verbose=False

        self.subj = meta(sub)
        self.wb_mask = os.path.join(self.subj.roi,'func_brain_mask.nii.gz')
        self.refvol = nib.load(self.subj.refvol_be)

        

        self.encoding_phases = ['baseline','fear_conditioning','extinction']
        self.mem_phases = ['memory_run_1','memory_run_2','memory_run_3']
        self.phases = self.encoding_phases + self.mem_phases
        
        self.load_data()

    def load_data(self):
        
        self.encoding_data = OrderedDict()
        self.encoding_labels = OrderedDict()
        
        self.mem_data = OrderedDict()
        self.mem_labels = OrderedDict()

        for phase in self.phases:
            beta_img = nib.load(os.path.join(self.subj.bold_dir, fsl_betas[phase]))
            
            if phase in self.encoding_phases:
                events = glm_timing(self.subj.num,phase).phase_events(stims=True)
                events['phase'] = phase
                self.encoding_labels[phase] = events
                self.encoding_data[phase] = beta_img
            
            elif phase in self.mem_phases:
                events = glm_timing(self.subj.num,phase).mem_events(stims=True)
                # foil_mask = events.memcond.isin(['Old'])
                events['phase'] = phase
                self.mem_labels[phase] = events          
                self.mem_data[phase] = beta_img

        self.encoding_labels = pd.concat(self.encoding_labels.values())
        self.encoding_labels.reset_index(inplace=True,drop=True)
        self.encoding_data = nib.concat_images(self.encoding_data.values(),axis=3)
        
        self.mem_labels = pd.concat(self.mem_labels.values())
        foil_mask = self.mem_labels.memcond.isin(['Old'])
        self.mem_labels = self.mem_labels[foil_mask].reset_index(drop=True)
        self.mem_data = image.index_img(nib.concat_images(self.mem_data.values(),axis=3), foil_mask)
    
    def init_SL(self,process_mask=None):
        

        if process_mask is not None:
            process_mask = nib.load(os.path.join(self.subj.roi,process_mask+'_mask.nii.gz'))
        
        self.sl = SearchLight(
                    mask_img=self.wb_mask, #whole brain mask
                    process_mask_img=process_mask, #where to run the searchlight (None = all)
                    radius=9, #radius in mm
                    estimator=RSA_estimator(self.subj.num), #just pass something for sklearn
                    scoring=None, #None defaults to our custom score() function in RSA_estimator, or 'auc_roc',or 'score'
                    cv=GroupKFold(n_splits=2), #not used for RSA
                    verbose=1,
                    n_jobs=-1, #-1: all processors
                    )
    
    def run_SL(self,ER_item,labels,groups):

        self.sl.fit(ER_item,labels,groups) #data, labels, groups

        s_img = new_img_like(self.refvol,self.sl.scores_)

        return s_img


    def wrap_run_SL(self):
        labels = ['encode','mem']
        groups = [1,2]

        self.ER_res = OrderedDict()
        
        for i, stim in enumerate(self.mem_labels.stim):
            mem_loc = i
            encoding_loc = np.where(self.encoding_labels.stim == stim)[0][0]

            ER_item = nib.concat_images([self.encoding_data.slicer[:,:,:,int(encoding_loc)],self.mem_data.slicer[:,:,:,int(mem_loc)]])

            self.ER_res[i] = self.run_SL(ER_item,labels,groups)
        
        self.ER_res = nib.concat_images(self.ER_res.values())

        self.save_str = os.path.join(self.subj.rsa,'item_ER.nii.gz')

        nib.save(self.ER_res,self.save_str)
    
    def reg_res(self):     

        ref2std = os.path.join(subj.subj_dir,'ref2std.mat')

        std = os.path.join(WORK,'standard','MNI152_T1_1mm_brain.nii.gz')

        os.system('flirt -in %s -ref %s -out %s -init %s -applyxfm'%(self.save_str,std,self.save_str,ref2std))
'''
tacc imports
import nibabel as nib
from fc_config import *
from nilearn import image
from collections import OrderedDict
from glm_timing import glm_timing
'''

def cell_means(sub):
    print(sub)
    subj = meta(sub)

    ER = nib.load(os.path.join(subj.rsa,'item_ER.nii.gz'))

    encoding_phases = ['baseline','fear_conditioning','extinction']
    mem_phases = ['memory_run_1','memory_run_2','memory_run_3']
    cons = ['CS+','CS-']
    mem_cons = ['H','M']

    mem_labels = OrderedDict()
    for phase in mem_phases:
        events = glm_timing(subj.num,phase).mem_events(stims=True)
        events['phase'] = phase
        mem_labels[phase] = events

    mem_labels = pd.concat(mem_labels.values())
    foil_mask = mem_labels.memcond.isin(['Old'])
    mem_labels = mem_labels[foil_mask].reset_index(drop=True)
    mem_labels['er_index'] = range(mem_labels.shape[0])

    for phase in encoding_phases:
        for con in cons:
            for Mphase in mem_phases:
                print(phase,con,Mphase[-1])
                idx = mem_labels.er_index[mem_labels.encode == phase][mem_labels.trial_type == con][mem_labels.phase == Mphase]
                cell = image.index_img(ER,idx)
                cell_mean = image.mean_img(cell)
                nib.save(cell_mean,os.path.join(subj.rsa,'%s_%s%s.nii.gz'%(phase,con,Mphase[-1])))

                _hold = os.path.join(subj.rsa,'placeholder.nii.gz')
                nib.save(cell,_hold)                  
                _std = os.path.join(subj.rsa,'std.nii.gz')                             
                os.system('fslmaths %s -Tstd %s'%(_hold,_std))
                
                var = os.path.join(subj.rsa,'%s_%s%s_var.nii.gz'%(phase,con,Mphase[-1]))
                os.system('fslmaths %s -sqr %s'%(_std,var))
                os.system('rm %s'%(_hold));os.system('rm %s'%(_std))

def feat_hack():
    print(sub)
    # subj = meta(sub)

    encoding_phases = ['baseline','fear_conditioning','extinction']
    mem_phases = ['memory_run_1','memory_run_2','memory_run_3']
    cons = ['CS+','CS-']
    mem_cons = ['H','M']
    
    DIR = '/Users/ach3377/Desktop/rsa'

    cope = 1
    for phase in encoding_phases:
        for con in cons:
            for Mphase in mem_phases:
                img = phase + '_' + con + Mphase[-1]
                mean = os.path.join(DIR,img + '.nii.gz')
                var = os.path.join(DIR,img + '_var.nii.gz')
                dest = os.path.join(DIR,'dummyLvl1.feat','stats')
                os.system('cp %s %s/cope%s.nii.gz'%(mean,dest,cope))
                os.system('cp %s %s/varcope%s.nii.gz'%(var,dest,cope))
                cope += 1



class group_mem_rsa():

    def __init__(self,mask=None):

        self.mask = mask
        self.statdir = os.path.join(data_dir,'group_ER',self.mask)
        if not os.path.exists(self.statdir): os.mkdir(self.statdir)
        
        # self.collect_sub_dat()

    def collect_sub_dat(self):
        self.rsa = {}

        for sub in all_sub_args:

            subj = meta(sub)

            subdat = pd.read_csv(os.path.join(subj.rsa,'%s_ER.csv'%(self.mask)))

            subdat['subject'] = subj.num

            if sub in sub_args:
                subdat['group'] = 'control'
            elif sub in p_sub_args:
                subdat['group'] = 'ptsd'

            self.rsa[sub] = subdat

        self.rsa = pd.concat(self.rsa.values())
        self.rsa.encode = pd.Categorical(self.rsa.encode,categories=['baseline','fear_conditioning','extinction'],ordered=True)
        self.rsa.to_csv(os.path.join(self.statdir,self.mask + '_group_stats.csv'),index=False)

    def collect_sub_clf(self,cope_type='csp'):
        self.c_cfmat = np.zeros((len(sub_args),3,3))
        self.p_cfmat = np.zeros((len(p_sub_args),3,3))
        
        c_res = pd.DataFrame([])
        for i,sub in enumerate(sub_args):
            subj = meta(sub)
            subdat = pd.read_csv(os.path.join(subj.rsa,'%s_%s_clf.csv'%(self.mask,cope_type)))
            subdat['subject'] = sub
            subdat['group'] = 'control'
            c_res = pd.concat([c_res,subdat])

            self.c_cfmat[i,:,:] = confusion_matrix(subdat.encode,subdat.guess,labels=['baseline','fear_conditioning','extinction'])
            self.c_cfmat[i,:,:] = self.c_cfmat[i,:,:].astype('float') / self.c_cfmat[i,:,:].sum(axis=1)[:, np.newaxis]

        p_res = pd.DataFrame([])
        for i,sub in enumerate(p_sub_args):
            subj = meta(sub)
            subdat = pd.read_csv(os.path.join(subj.rsa,'%s_%s_clf.csv'%(self.mask,cope_type)))
            subdat['subject'] = sub
            subdat['group'] = 'ptsd'
            p_res = pd.concat([p_res,subdat])
         
            self.p_cfmat[i,:,:] = confusion_matrix(subdat.encode,subdat.guess,labels=['baseline','fear_conditioning','extinction'])
            self.p_cfmat[i,:,:] = self.p_cfmat[i,:,:].astype('float') / self.p_cfmat[i,:,:].sum(axis=1)[:, np.newaxis]
        
        self.c_cfmat = self.c_cfmat.mean(axis=0)
        self.p_cfmat = self.p_cfmat.mean(axis=0)
        
        self.clf_res = pd.concat([c_res,p_res])
        self.clf_res.baseline = np.arctanh(self.clf_res.baseline)
        self.clf_res.fear_conditioning = np.arctanh(self.clf_res.fear_conditioning)
        self.clf_res.extinction = np.arctanh(self.clf_res.extinction)

        self.clf_res = self.clf_res.groupby(['subject','encode']).mean().reset_index().melt(id_vars=['subject','encode'],value_vars=['baseline','fear_conditioning','extinction'],var_name='comp_phase',value_name='rsa')   
        self.clf_res.comp_phase = pd.Categorical(self.clf_res.comp_phase,categories=['baseline','fear_conditioning','extinction'],ordered=True)
        self.clf_res.sort_values(by='subject',inplace=True)
        self.clf_res.encode = pd.Categorical(self.clf_res.encode,categories=['baseline','fear_conditioning','extinction'],ordered=True)
        self.clf_res['group'] = np.repeat(['control','ptsd'],int(24*3*3))
    
    def collect_cross(self):
        self.c_xmat = np.zeros((len(sub_args),144,144))
        self.p_xmat = np.zeros((len(p_sub_args),144,144))
        c_count = 0
        p_count = 0
    
        for sub in all_sub_args:
            subdat = mem_rsa(sub,self.mask)
            subdat.cross_rsa()

            if sub in sub_args:
                self.c_xmat[c_count,:,:] = subdat.cross_mat
                c_count += 1

            elif sub in p_sub_args:
                self.p_xmat[p_count,:,:] = subdat.cross_mat
                p_count += 1

    def vis_cross(self):

        fig, ax = plt.subplots(1,2)
        sns.heatmap(R.c_xmat.mean(axis=0),ax=ax[0],cmap='Greens',vmin=-.1,vmax=.2)
        sns.heatmap(R.p_xmat.mean(axis=0),ax=ax[1],cmap='Greens',vmin=-.1,vmax=.2)
        plt.suptitle('ROI = %s'%(R.mask))

    def vis_clf_res(self,cf=False):
        labels = ['baseline','fear_conditioning','extinction']
        if cf:
            fig, ax = plt.subplots(1,2)
            sns.heatmap(self.c_cfmat,ax=ax[0],cmap='Greens',
                xticklabels=labels,yticklabels=labels,annot=True,vmin=.1,vmax=.75)
            sns.heatmap(self.p_cfmat,ax=ax[1],cmap='Greens',
                xticklabels=labels,yticklabels=labels,annot=True,vmin=.1,vmax=.75)
            plt.suptitle('ROI = %s'%(self.mask))
    
        fig, ax = plt.subplots(1,3,sharey=True)
        for i, phase in enumerate(labels):
            sns.pointplot(x='comp_phase',y='rsa',hue='group',data=self.clf_res.query('encode == @phase'),
                            join=False,dodge=True,ax=ax[i],palette=gpal)
            ax[i].set_title(phase)
            sns.despine(ax=ax[i])
            if i != 0: ax[i].legend_.remove()
            plt.suptitle('encoding phase (ROI = %s)'%(self.mask))

    def vis_mem_rsa(self):
        sns.set_style('ticks');sns.set_context('talk')
        gpal = list((wes_palettes['Zissou'][0],wes_palettes['Royal1'][1]))

        # hitormiss = self.rsa.groupby(['group','subject','hc_acc']).mean().reset_index()
        # fig, ax = plt.subplots()
        # sns.pointplot(x='hc_acc',y='rsa',data=hitormiss,hue='group',dodge=True,ax=ax)

        # pg.mixed_anova(data=hitormiss,dv='rsa',within='hc_acc',subject='subject',between='group',
        #             export_filename=os.path.join(self.statdir,'accXgroup.csv'))


        PHM = self.rsa.groupby(['subject','encode','hc_acc']).mean().reset_index()
        PHM['group'] = np.repeat(['control','ptsd'],int(24*3*2))
        fig, ax = plt.subplots(1,3,sharey=True)
        for i, phase in enumerate(PHM.encode.unique()):
            DATA = PHM.query('encode == @phase')
            sns.pointplot(x='hc_acc',y='rsa',hue='group',dodge=True,palette=gpal,
                        data=DATA,ax=ax[i])
            ax[i].set_title(phase)
            sns.despine(ax=ax[i])
            if i > 0: ax[i].legend_.remove()
            pg.mixed_anova(data=DATA,dv='rsa',within='hc_acc',subject='subject',between='group',
                            export_filename=os.path.join(self.statdir,'%s_accXgroup.tsv'%(phase)))
            # print(pg.mixed_anova(data=DATA,dv='rsa',within='hc_acc',subject='subject',between='group'))
        fig.suptitle(self.mask)

        # for group in ['control','ptsd']:
        #     gdat = self.rsa.query('group == @group')

        #     pdat = gdat.groupby(['subject','encode','trial_type','hc_acc']).mean().reset_index()
        #     fig, ax = plt.subplots(1,3,sharey=True)
        #     for i, phase in enumerate(pdat.encode.unique()):
        #         sns.pointplot(x='trial_type',y='rsa',hue='hc_acc',dodge=True,data=pdat.query('encode == @phase'),ax=ax[i])
        #         ax[i].set_title(phase)    
        #         sns.despine(ax=ax[i])
        #     fig.suptitle(group)



        # CON = self.rsa.groupby(['subject','encode','trial_type']).mean().reset_index()
        # CON['group'] = np.repeat(['control','ptsd'],int(24*3*2))
        # fig, ax = plt.subplots(1,3,sharey=True)
        # for i, phase in enumerate(CON.encode.unique()):
        #     DATA = CON.query('encode == @phase')
        #     sns.pointplot(x='trial_type',y='rsa',hue='group',dodge=True,palette=gpal,
        #                 data=DATA,ax=ax[i])
        #     ax[i].set_title(phase)
        #     sns.despine(ax=ax[i])
        #     if i > 0: ax[i].legend_.remove()
        #     # pg.mixed_anova(data=DATA,dv='rsa',within='hc_acc',subject='subject',between='group',
        #     #                 export_filename=os.path.join(self.statdir,'%s_accXgroup.tsv'%(phase)))
        #     # print(pg.mixed_anova(data=DATA,dv='rsa',within='hc_acc',subject='subject',between='group'))
        # fig.suptitle(self.mask)

        # INTRX = self.rsa.groupby(['subject','encode','trial_type','hc_acc']).mean().reset_index()   
        # INTRX['group'] = np.repeat(['control','ptsd'],int(24*3*2*2))
        # fig, ax = plt.subplots(2,3,sharey=True)
        # for i, con in enumerate(INTRX.trial_type.unique()):
        #     for j, phase in enumerate(INTRX.encode.unique()):
        #         DATA = INTRX[INTRX.encode == phase][INTRX.trial_type == con]
        #         sns.pointplot(x='hc_acc',y='rsa',hue='group',dodge=True,palette=gpal,
        #                         data=DATA,ax=ax[i,j])
        #         ax[i,j].set_title(phase+'_'+con)
        #         sns.despine(ax=ax[i,j])
        #         if not i + j == 0: ax[i,j].legend_.remove()
        # fig.suptitle(self.mask)

def create_mem_dfs():
    mem_phases = ['memory_run_1','memory_run_2','memory_run_3']
    df = {}
    for sub in all_sub_args:
        sub_df = {}

        for phase in mem_phases: sub_df[phase] = glm_timing(sub,phase).mem_events()

        sub_df = pd.concat(sub_df.values())

        sub_df = sub_df[sub_df.memcond.isin(['Old'])].reset_index(drop=True)
        sub_df = sub_df.drop(columns=['onset','duration','memcond'])
        
        sub_df['subject'] = sub
        
        if sub in sub_args: sub_df['group'] = 'control'
        if sub in p_sub_args: sub_df['group'] = 'ptsd'

        sub_df.acc = (sub_df.acc == 'H').astype(int)
        sub_df.hc_acc = (sub_df.hc_acc == 'H').astype(int)
        
        df[sub] = sub_df

    df = pd.concat(df.values()).reset_index(drop=True)
    df['rsa'] = 0

    #loading data
    data = np.load(os.path.join(data_dir,'group_ER','all_subs_std_item_ER.npy'))
    for i in range(data.shape[2]):
        vdf = df.copy()
        vdf.rsa = data[:,:,i].flatten()
        
        outstr = os.path.join(data_dir,'group_ER','voxel_dfs','voxel_%s.csv'%(i))
        vdf.to_csv(outstr,index=False)

    #knowing which subs to exlude from memory anovas
    lc_remove = np.ndarray(0)
    hc_remove = np.ndarray(0)
    for sub in all_sub_args:
        for phase in df.encode.unique():
            for con in df.trial_type.unique():
                for mem_cond in [0,1]:
                    if np.where(df.acc[df.subject == sub][df.encode == phase][df.trial_type == con] == mem_cond)[0].shape[0] < 5:
                        lc_remove = np.append(lc_remove,sub)
                    if np.where(df.hc_acc[df.subject == sub][df.encode == phase][df.trial_type == con] == mem_cond)[0].shape[0] < 5:    
                        hc_remove = np.append(hc_remove,sub)
    lc_remove = np.unique(lc_remove)
    hc_remove = np.unique(hc_remove)



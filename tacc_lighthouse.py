import os
import numpy as np
import pandas as pd
import nibabel as nib

from fc_config import *
from glm_timing import glm_timing

from nilearn.decoding import SearchLight
from nilearn.input_data import NiftiMasker
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold
from sklearn.linear_model import LogisticRegression
from nilearn.image import index_img, new_img_like
from predictive_searchlight import Predictive_SearchLight

def wrap_predict_proba(estimator,X_test,y_test,**kwargs):
    for i, v in enumerate(estimator.classes_):
        if v == 'outdoor': scene_loc = i
    # if verbose: print(X_test.shape)
    score = np.mean(estimator.predict_proba(X_test)[:,scene_loc])
    return score

class lighthouse():

    def __init__(self,sub,func,train='localizer',test='extinction_recall',
                  cv=2,process_mask=None,standardize=True):

        self.func = func
        self.test_phase = test
        self.train_phase = train
        #these are stable
        self.subj = meta(sub)
        self.refvol = nib.load(self.subj.refvol_be)
        self.wb_mask = os.path.join(self.subj.roi,'func_brain_mask.nii.gz')
        self.cv = GroupKFold(n_splits=2)

        self.init_SL(process_mask)
        self.load_data(standardize=standardize)
        self.run_SL()

    def init_SL(self,process_mask):
        if self.func == 'predictive': 
            SL = Predictive_SearchLight
            scoring = wrap_predict_proba
        elif self.func == 'xval':
            SL = SearchLight
            scoring = 'roc_auc'
        if process_mask is not None:
            process_mask = nib.load(os.path.join(self.subj.roi,process_mask+'_mask.nii.gz'))
        self.sl = SL(
                    mask_img=self.wb_mask,
                    process_mask_img=process_mask,
                    radius=3,
                    estimator=LogisticRegression(solver='liblinear'),
                    scoring=scoring,
                    cv=self.cv,
                    verbose=1,
                    n_jobs=-1
                    )

    def load_data(self,standardize=True):
        if standardize: masker = NiftiMasker(mask_img=self.wb_mask,standardize=standardize)
        
        if self.train_phase == 'localizer': load_phases = ['localizer_1','localizer_2']
        elif self.train_phase == 'day1':    load_phases = ['fear_conditioning','extinction']
        
        #handle the exception case first
        if self.train_phase == 'localizer' and self.subj.num == 107:    
            train_dat = nib.load(self.subj.bold_dir + fsl_betas['localizer_1'])
        
            if standardize:
                train_dat = masker.fit_transform(train_dat)
                train_dat = masker.inverse_transform(train_dat)
        
            train_lab = glm_timing(self.subj.num,'localizer_1').loc_blocks(con=True).reset_index(drop=True)
        
        else:
            train_dat = {}
        
            for phase in load_phases:
                train_dat[phase] = nib.load(self.subj.bold_dir + fsl_betas[phase])
            
                if standardize:
                    train_dat[phase] = masker.fit_transform(train_dat[phase])
                    train_dat[phase] = masker.inverse_transform(train_dat[phase])
        
            train_dat = nib.concat_images([train_dat[ load_phases[0] ],train_dat[ load_phases[1] ] ], axis=-1)
            if self.train_phase == 'localizer': train_lab = pd.concat([glm_timing(self.subj.num,load_phases[0]).loc_blocks(con=True),glm_timing(self.subj.num,load_phases[1]).loc_blocks(con=True)]).reset_index(drop=True)
            elif self.train_phase == 'day1':    train_lab = pd.concat([glm_timing(self.subj.num,load_phases[0]).phase_events(con=True),glm_timing(self.subj.num,load_phases[1]).phase_events(con=True)]).reset_index(drop=True)

        #find the correct conditions
        if self.train_phase == 'localizer': 
            condition_mask = train_lab.isin(['outdoor','scrambled'])
            train_lab = train_lab[condition_mask]
        elif self.train_phase == 'day1':    
            condition_mask = train_lab.isin(['CS+'])
            train_lab = np.repeat(load_phases,int(condition_mask.shape[0]/4))
        
        train_dat = index_img(train_dat,condition_mask)
        
        #if predictive load test data
        if self.func == 'predictive':
            test_dat = nib.load(self.subj.bold_dir + fsl_betas[self.test_phase])
            
            if standardize:
                test_dat = masker.fit_transform(test_dat)
                test_dat = masker.inverse_transform(test_dat)
                
            cond_mask = glm_timing(self.subj.num,self.test_phase).phase_events(con=True)
            csp_mask = cond_mask.isin(['CS+'])
            test_dat = index_img(test_dat,csp_mask)
            
            if self.test_phase == 'extinction_recall':
                csp4 = np.concatenate([np.repeat(True,4),np.repeat(False,8)])
                test_dat = index_img(test_dat,csp4)

            #this is necessarily junk
            test_lab = np.repeat(['outdoor','scrambled'],int(test_dat.shape[-1]/2))

            #join train and test data
            self.data = nib.concat_images([train_dat,test_dat],axis=-1)
            self.labels = np.concatenate([train_lab.values,test_lab])
            self.groups = np.concatenate([np.repeat(1,train_lab.shape[0]),np.repeat(2,test_lab.shape[0])])

        #if xval just need the localizer data
        elif self.func == 'xval':
            self.data = train_dat
            self.labels = train_lab
            if self.train_phase == 'localizer': self.groups = np.repeat([1,2],int(train_lab.shape[0]/2))
            elif self.train_phase == 'day1':    self.groups = np.tile([1,2],int(train_lab.shape[0]/2))

    def run_SL(self):
        
        print('FITTING %s'%(self.subj.num))
        self.sl.fit(self.data,self.labels,self.groups)

        if self.func == 'predictive':
            self.sl_res = self.sl.scores_[0]
            save_str = self.func + '_' + self.test_phase
        
        elif self.func == 'xval':
            self.sl_res = self.sl.scores_
            save_str = self.func + '_' + self.train_phase
        
        s_img = new_img_like(self.refvol,self.sl_res)
        
        nib.save(s_img,os.path.join(self.subj.sl_dir,save_str + '.nii.gz'))





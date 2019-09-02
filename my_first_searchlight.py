import os
import numpy as np
import pandas as pd
from collections import OrderedDict

from nilearn.input_data import NiftiMasker
from nilearn import image
from nilearn.decoding import SearchLight
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold
from sklearn.linear_model import LogisticRegression
from nilearn.image import index_img, new_img_like



class ER_searchlight():

    def __init__(self,sub,process_mask=None):
        '''
        res = ER_searchlight(sub,process_mask)
        '''
        self.verbose=False

        self.subj = meta(sub) #this is for file paths
        
        #you need these
        self.wb_mask = os.path.join(self.subj.roi,'func_brain_mask.nii.gz')
        self.refvol = nib.load(self.subj.refvol_be)

        
        #these are gus variables
        self.encoding_phases = ['baseline','fear_conditioning','extinction']
        self.mem_phases = ['memory_run_1','memory_run_2','memory_run_3']
        self.phases = self.encoding_phases + self.mem_phases
        

        '''
        comment these out to not run them when you call the class
        '''
        #load the data
        self.load_data()
        self.init_SL()
        self.run_SL()


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


    def wrap_run_SL(self): #this is just for gus ER
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

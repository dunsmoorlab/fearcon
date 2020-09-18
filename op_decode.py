from fc_config import *
from beta_rsa import *
from preprocess_library import *
from glm_timing import *
import seaborn as sns
import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score
from cf_mat_plot import plot_confusion_matrix
from wesanderson import wes_palettes
from sklearn.preprocessing import label_binarize
from nilearn.input_data import NiftiMasker
from nilearn.masking import apply_mask
from nilearn.image import resample_to_img
from collections import OrderedDict
from nilearn import image
from roi_rsa import roi_rsa
import time
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score, cross_val_predict
from sklearn.multiclass import OneVsRestClassifier

class op_decode():
    def __init__(self,sub=None,roi='circuit_mask',ext_split=True,feature=False):

        # self.verbose=verbose
        self.subj = meta(sub)
        self.ext_split = ext_split
        self.roi = roi
        self.encoding_phases = ['baseline','fear_conditioning','extinction']
        self.mem_phases = ['memory_run_1','memory_run_2','memory_run_3']
        self.phases = self.encoding_phases + self.mem_phases
        weights = 'copes'
        self.weight_dir = os.path.join(data_dir,'rsa_%s'%(weights),self.subj.fsub)
        #need these to load the weights becauase I named them lazily
        self.conditions = {'CS+': 'csp',
                           'CS-': 'csm'}
        self.w_phases = {'baseline'         : 'baseline',
                         'fear_conditioning': 'fear',
                         'extinction'       : 'ext'}
        self.load_data()
        self.encoding_data = roi_rsa.apply_mask(self,roi=self.roi,target=self.encoding_data)

        
        if self.ext_split:
            labels = self.encoding_labels.encode.values
            self.classes = ['baseline','fear_conditioning','early_extinction','extinction']
        else: 
            labels = self.encoding_labels.phase.values
            self.classes = ['baseline','fear_conditioning','extinction']
        
        self.xval(data=self.encoding_data,labels=labels)
        

    def load_data(self):
    
        #load reference vol
        ref = nib.load(self.subj.refvol_be)        
        
        #encoding data & labels
        self.encoding_data = OrderedDict()
        self.encoding_labels = OrderedDict()

        #ls-u style beta weights
        # self.W = OrderedDict()
        
        #retrieva data & labels
        # self.mem_data = OrderedDict()
        # self.mem_labels = OrderedDict()

        for phase in self.phases:

            beta_img = nib.load(os.path.join(self.subj.bold_dir, fsl_betas[phase]))

            if phase in self.encoding_phases:
                events = glm_timing(self.subj.num,phase).phase_events(stims=True)
                for cs in self.conditions:
                    _con = self.conditions[cs]+'_'+'trial'
                    events[_con] = ''
                    events.loc[np.where(events.trial_type==cs,)[0],_con] = [i for i in range(1,25)]
                    

                events['phase'] = phase #this isn't in the dataframe yet
                #save labels & data
                self.encoding_labels[phase] = events
                self.encoding_data[phase] = beta_img
                
                #load in the weights here
                # self.W[phase] = OrderedDict()
                # for con in self.conditions:
                #     self.W[phase][con] = nib.load(
                #             os.path.join(
                #             self.weight_dir,'%s_%s.nii.gz'%
                #             (self.w_phases[phase],self.conditions[con])))
        
            # elif phase in self.mem_phases:
                # events = glm_timing(self.subj.num,phase).mem_events(stims=True)
                # events['phase'] = phase #add phase to df
                ##save labels & data
                # self.mem_labels[phase] = events
                # self.mem_data[phase] = beta_img
        
        #get 1 dataframe & image for encoding
        self.encoding_labels = pd.concat(self.encoding_labels.values())
        self.encoding_labels.reset_index(inplace=True,drop=True) #need this bc phase events are all numbered the same
        self.encoding_data = nib.concat_images(self.encoding_data.values(),axis=3)
        #same for retrieval, except we have to remove foils
        # self.mem_labels = pd.concat(self.mem_labels.values())
        # foil_mask = self.mem_labels.memcond.isin(['Old'])
        # self.mem_labels = self.mem_labels[foil_mask].reset_index(drop=True)
        # self.mem_data = image.index_img(nib.concat_images(self.mem_data.values(),axis=3), foil_mask)

        encode_csp_mask = self.encoding_labels.trial_type.isin(['CS+'])
        self.encoding_labels = self.encoding_labels[encode_csp_mask].reset_index(drop=True)
        self.encoding_data = image.index_img(self.encoding_data,encode_csp_mask)

        # mem_csp_mask = self.mem_labels.trial_type.isin(['CS+'])
        # self.mem_labels = self.mem_labels[mem_csp_mask].reset_index(drop=True)
        # self.mem_data = image.index_img(self.mem_data,mem_csp_mask)

        self.encoding_labels['encode'] = self.encoding_labels.phase
        for i in range(1,9): 
            self.encoding_labels.loc[ self.encoding_labels[ self.encoding_labels.phase == 'extinction' ][ self.encoding_labels['csp_trial'] == i ].index,'encode' ] = 'early_extinction'


    # def init_fit_clf(self,data=None,labels=None,feature=None):

    #     # self.clf = Pipeline([ ('anova', SelectKBest(f_classif, k=None)), ('clf', LogisticRegression() ) ])
    #     self.clf = LinearSVC()
    #     self.clf.fit(data, labels)

    def xval(self,data,labels):
        xval_groups = np.tile([1,2],int(data.shape[0]/2))
        logo = LeaveOneGroupOut()
        self.clf = LogisticRegression(solver='lbfgs',multi_class='ovr')
        self.xval_score = cross_val_score(self.clf,
                                            data,y=labels,
                                            groups=xval_groups,cv=logo,
                                            scoring='accuracy',).mean()
        self.cf_mat = confusion_matrix(labels,
                        cross_val_predict(self.clf,data,y=labels,
                                        groups=xval_groups,cv=logo),
                        labels=self.classes)
        #normalize the confusion matrix
        self.cf_mat = np.array([row/row.sum() for row in self.cf_mat])
    def predict(self):
        #load extinction recall and clean out CS-
        
        self.test_data = nib.load(os.path.join(self.subj.bold_dir, fsl_betas['extinction_recall']))
        self.test_labels = glm_timing(self.subj.num,'extinction_recall').phase_events(con=True)
        test_csp_mask = self.test_labels.isin(['CS+'])

        self.test_labels = self.test_labels[test_csp_mask].reset_index(drop=True)
        self.test_data = image.index_img(self.test_data,test_csp_mask)
        self.test_data = roi_rsa.apply_mask(self,roi=self.roi,target=self.test_data)
        # test_data = np.delete(test_data, np.where(glm_timing(self.subj.num,'extinction_recall').phase_events(con=True) == 'CS-'), axis=0)

        # # proba_res = self.clf.decision_function(test_data)
        # proba_res = 1. / (1. + np.exp(-self.clf.decision_function(test_data)))

        # _res = {label: proba_res[:,i] for i, label in enumerate(self.clf.classes_)}

        # return _res



class cross_decode():
    def __init__(self,subjects=None,load_data=False,read_data=False,read_res=False,mask='wb',test_fit=False,solver='lbfgs',multi='ovr'):

        print(subjects)
        
        self.subjects = subjects
        if subjects == p_sub_args: self.group = 'ptsd'
        else: self.group = 'control'
        
        self.beta_dir = os.path.join(data_dir,'std_betas')
        self.mask = mask

        if self.mask is 'wb':
            mask_img = os.path.join(self.beta_dir,'MNI152_T1_3mm_brain_mask.nii.gz')
        elif self.mask is 'vmPFC':
            mask_img = os.path.join(data_dir,'group_roi_masks','vmPFC_roi','group_func_vmPFC_mask.nii.gz')
        self.masker = NiftiMasker(mask_img=mask_img, standardize=True)
        self.masker.fit()

        self.multi = multi

        if read_data: self.read_data()
        elif load_data: self.load_data()

        if read_res: 
            self.read_res()
        else: 
            self.init_fit_clf(data=self.data,labels=self.labels,test_fit=test_fit,solver=solver,multi=multi)

        self.cf_labels = ['baseline', 'early_fear', 'late_fear', 'early_ext','late_ext']#,'early_rnw','late_rnw']
        
        if not test_fit:    
            self.xval()
            self.group_stats()

        
    def read_res(self):
        with open(os.path.join(data_dir,'graphing','op_decode','%s_%s_res.p'%(self.group,self.mask)),'rb') as file:
            inDat = pickle.load(file)
        self.cf_mats = inDat['cf_mats']
        self.aucs = inDat['aucs']
        self.rnw_prob = inDat['rnw_prob']
        del inDat

    def read_data(self):
        with open(os.path.join(self.beta_dir,'%s_%s.p'%(self.group,self.mask)),'rb') as file:
            saved = pickle.load(file)
        self.data = saved['data']
        self.labels = saved['labels']
        self.rnw_data = saved['rnw_data']
        self.sub_labels = saved['sub_labels']
        self.rnw_sub_labels = saved['rnw_sub_labels']

        del saved

    def load_data(self):


        self.data = {}
        self.rnw_data = {}
        for sub in self.subjects:
            self.data[sub] = {}
            subj = meta(sub)
            print(sub)
            sub_dir = os.path.join(self.beta_dir,subj.fsub,'beta_std')
            for phase, run in zip(['baseline','fear_conditioning','extinction','extinction_recall'],range(1,5)):                
                print(phase)
                #load in data
                beta_img = os.path.join(sub_dir,'run00%s_beta_std.nii.gz'%(run))
                #fit the mask
                beta_img = self.masker.fit_transform(beta_img)
                #get trial type information
                run_labels = glm_timing(sub,phase).phase_events(con=True)
                #clean out CS- trials
                beta_img = np.delete(beta_img, np.where(run_labels == 'CS-')[0],axis=0)
                if phase is not 'extinction_recall':
                    #if its baseline then were going to average every 2 images to downsample
                    if phase is 'baseline':
                        beta_img = beta_img.reshape(int(beta_img.shape[0]/2),beta_img.shape[1],2).mean(axis=-1)
                    #save it to the train data
                    self.data[sub][phase] = beta_img
                else:
                    #save the renewal data
                    self.rnw_data[sub] = beta_img
            
            self.data[sub] = np.concatenate((self.data[sub]['baseline'],self.data[sub]['fear_conditioning'],self.data[sub]['extinction']))#,self.data[sub]['extinction_recall']))

        sub1_labels = np.concatenate((np.repeat('baseline',12),
                                     np.repeat('early_fear',12),np.repeat('late_fear',12),
                                     np.repeat('early_ext',12),np.repeat('late_ext',12)))
                                     #np.repeat('early_rnw',4),np.repeat('late_rnw',8)))

        self.labels = np.tile(sub1_labels,len(self.subjects))
        self.sub_labels = np.repeat(self.subjects,len(sub1_labels))
        self.rnw_sub_labels = np.repeat(self.subjects,12)

        if self.subjects == sub_args:
            self.data = np.concatenate((self.data[1],self.data[2],self.data[3],self.data[4],self.data[5],self.data[6],self.data[7],self.data[8],self.data[9],self.data[10],self.data[12],self.data[13],self.data[14],self.data[15],self.data[16],self.data[17],self.data[18],self.data[19],self.data[20],self.data[21],self.data[23],self.data[24],self.data[25],self.data[26]))
        
        elif self.subjects == p_sub_args:   
            self.data = np.concatenate((self.data[101],self.data[102],self.data[103],self.data[104],self.data[105],self.data[106],self.data[107],self.data[108],self.data[109],self.data[110],self.data[111],self.data[112],self.data[113],self.data[114],self.data[115],self.data[116],self.data[117],self.data[118],self.data[120],self.data[121],self.data[122],self.data[123],self.data[124],self.data[125]))
    
    def init_fit_clf(self,data=None,labels=None,fs=None,test_fit=False,solver='lbfgs',multi='ovr'):
        print(solver)
        #self.clf = Pipeline([ ('clf', LogisticRegression(solver='lbfgs',n_jobs=4,verbose=1) ) ])
        self.clf = Pipeline([ ('clf', LogisticRegression(solver=solver,multi_class=multi) ) ])

        if test_fit:
            now = time.time()
            self.clf.fit(data, labels)
            print(time.time() - now)

            self.coef_ = self.clf.steps[0][1].coef_
            self.avg_data = np.zeros(self.coef_.shape)
            for i,label in enumerate(self.clf.classes_):
                self.avg_data[i,:] = self.data[self.labels==label,:].mean(axis=0)
            self.importances = self.avg_data*self.coef_*((self.avg_data*self.coef_)>0)*np.sign(self.coef_)              


            template = nib.load(os.path.join(data_dir,self.beta_dir,'MNI152_T1_1mm_brain.nii.gz'))

            self.imp_maps = {}
            for i,label in enumerate(self.clf.classes_):
                self.imp_maps[label] = self.masker.inverse_transform(self.importances[i,:])
                self.imp_maps[label] = resample_to_img(self.imp_maps[label],template,interpolation='nearest')
                nib.save(self.imp_maps[label],os.path.join(data_dir,'graphing','op_decode','importance_maps',self.group,self.mask,'%s.nii.gz'%(label)))

    def xval(self):
    
        self.cf_mats = {}
        self.aucs = {}
        self.rnw_prob = {}
        self.rnw_guess = {}
        
        # auc = dict.fromkeys(list(np.unique(self.groups)))
        for sub in self.subjects:
            print(sub)
            self.clf.fit(self.data[self.sub_labels != sub], self.labels[self.sub_labels != sub])

            #AUC
            des_func = self.clf.decision_function(self.data[self.sub_labels == sub])
            bin_lab = label_binarize(self.labels[self.sub_labels==sub],self.clf.classes_)
            auc = roc_auc_score(bin_lab,des_func,average=None)
            self.aucs[sub] = { label: score for label, score in zip(self.clf.classes_,auc) }

            #confusion matrix
            cfmat_guess = self.clf.predict(self.data[self.sub_labels == sub])
            self.cf_mats[sub] = confusion_matrix(self.labels[self.sub_labels==sub], cfmat_guess, labels=self.cf_labels)

            #predict & guess on renewal
            if self.multi == 'ovr':
                rnw_raw = 1. / (1. + np.exp(-self.clf.decision_function(self.rnw_data[sub])))
            elif self.multi == 'multinomial':
                rnw_raw = self.clf.predict_proba(self.rnw_data[sub])
            #aggregate for all subs     
            self.rnw_prob[sub] = {label: rnw_raw[:,i] for i, label in enumerate(self.clf.classes_)}
            self.rnw_guess[sub] = self.clf.predict(self.rnw_data[sub])

            #save out the data
        outDat = {'cf_mats':self.cf_mats,'aucs':self.aucs,'rnw_prob':self.rnw_prob}
        with open(os.path.join(data_dir,'graphing','op_decode','%s_%s_res.p'%(self.group,self.mask)),'wb') as file:
            pickle.dump(outDat,file)

    def group_stats(self):          

        #cfmat
        if self.subjects == sub_args:
            self.cf_mat_avg = np.mean( np.array( [self.cf_mats[1], self.cf_mats[2], self.cf_mats[3], self.cf_mats[4], self.cf_mats[5], self.cf_mats[6], self.cf_mats[7], self.cf_mats[8], self.cf_mats[9], self.cf_mats[10], self.cf_mats[12], self.cf_mats[13], self.cf_mats[14], self.cf_mats[15], self.cf_mats[16], self.cf_mats[17], self.cf_mats[18], self.cf_mats[19], self.cf_mats[20], self.cf_mats[21], self.cf_mats[23], self.cf_mats[24], self.cf_mats[25], self.cf_mats[26]] ), axis=0 )

        elif self.subjects == p_sub_args:
            self.cf_mat_avg = np.mean(np.array( [self.cf_mats[101],self.cf_mats[102],self.cf_mats[103],self.cf_mats[104],self.cf_mats[105],self.cf_mats[106],self.cf_mats[107],self.cf_mats[108],self.cf_mats[109],self.cf_mats[110],self.cf_mats[111],self.cf_mats[112],self.cf_mats[113],self.cf_mats[114],self.cf_mats[115],self.cf_mats[116],self.cf_mats[117],self.cf_mats[118],self.cf_mats[120],self.cf_mats[121],self.cf_mats[122],self.cf_mats[123],self.cf_mats[124],self.cf_mats[125]] ), axis=0)

        plot_confusion_matrix(self.cf_mat_avg,self.cf_labels,normalize=True,title=self.group)#,cmap=wes_palettes['Zissou'])
        plt.savefig(os.path.join(data_dir,'graphing','op_decode','%s_%s_cfmat.png'%(self.group,self.mask)),dpi=300,bbox_inches='tight',pad_inches=.2)
        plt.close()
        
        #AUC
        self.auc_df = pd.DataFrame.from_dict({(sub,label): self.aucs[sub][label]
                                            for sub in self.aucs.keys()
                                            for label in self.aucs[sub].keys()}, orient='index')
        self.auc_df.reset_index(inplace=True)
        self.auc_df.rename({0:'auc'},axis=1,inplace=True)
        #apply series to get rid of (subject,label) tuple
        self.auc_df = pd.concat([self.auc_df['index'].apply(pd.Series), self.auc_df['auc']],axis=1)#columns=['subject','label','auc'])
        self.auc_df.rename({0:'subject',1:'label'},axis=1,inplace=True)

        sns.set_context('poster')
        fig, bx = plt.subplots(figsize=(12,9))
        bx = sns.pointplot(x='label',y='auc',data=self.auc_df, color='grey',capsize=.05,join=False,order=self.cf_labels)
        sns.swarmplot(x='label',y='auc',hue='subject',data=self.auc_df,alpha=.6,size=8,palette='bright',order=self.cf_labels)
        bx.axhline(y=.5,color='black',linestyle='--')
        plt.title('%s AUC'%(self.group))
        bx.legend_.remove()
        bx.set_ylim(.3,1)
        plt.savefig(os.path.join(data_dir,'graphing','op_decode','%s_%s_AUC.png'%(self.group,self.mask)),dpi=300,bbox_inches='tight',pad_inches=.2)
        plt.close()

        # renewal evidence
        self.rnw_df = pd.DataFrame.from_dict({(sub,label): self.rnw_prob[sub][label]
                                    for sub in self.rnw_prob.keys()
                                    for label in self.rnw_prob[sub].keys()}, orient='index')
        self.rnw_df.reset_index(inplace=True)
        self.rnw_df['subject'] = self.rnw_df['index'].apply(pd.Series)[0]
        self.rnw_df['label'] = self.rnw_df['index'].apply(pd.Series)[1]
        self.rnw_df.drop('index',axis=1,inplace=True)
        self.rnw_df = self.rnw_df.melt(id_vars=['subject','label'])
        self.rnw_df.rename({'variable':'trial','value':'evidence'},axis=1,inplace=True)
        self.rnw_df.trial += 1

        ss = sns.lineplot(x='trial',y='evidence',hue='label',style='label',err_style='bars',data=self.rnw_df)
        plt.savefig(os.path.join(data_dir,'graphing','op_decode','%s_%s_trial_evidence.png'%(self.group,self.mask)),dpi=300,bbox_inches='tight',pad_inches=.2)
        plt.close()
        
        self.rnw_df.set_index(['subject','trial','label'],inplace=True)
        self.rnw_df.sort_index(inplace=True)
        idx=pd.IndexSlice
        self.early_avg = self.rnw_df.loc[idx[:,1:4,:],:].groupby(level=[0,2]).mean()
        self.early_avg.reset_index(inplace=True)

        fig, ss = plt.subplots(figsize=(12,9))
        ss = sns.pointplot(x='label',y='evidence',data=self.early_avg, color='grey',capsize=.05,join=False,order=self.cf_labels)
        sns.swarmplot(x='label',y='evidence',hue='subject',data=self.early_avg,alpha=.6,size=8,palette='bright',order=self.cf_labels)
        plt.title('%s early renewal evidences'%(self.group))
        ss.legend_.remove()
        plt.savefig(os.path.join(data_dir,'graphing','op_decode','%s_%s_early_evidence.png'%(self.group,self.mask)),dpi=300,bbox_inches='tight',pad_inches=.2)
        plt.close()


        #lets try with scene evidence
        self.ctx = pd.read_csv(os.path.join(data_dir,'graphing','signal_change','mvpa_ev.csv'))
        if self.group == 'control': Group = 'Control'
        if self.group == 'ptsd':    Group = 'PTSD'

        self.ctx = self.ctx[self.ctx.Group == Group]
        self.ctx.set_index('subject',inplace=True)
        self.early_avg.set_index('subject',inplace=True)
        self.early_avg['Response'] = self.ctx.Response
        self.early_avg['ctx'] = self.ctx.ev
        self.early_avg.reset_index(inplace=True)

        rr = sns.lmplot(x='evidence', y='ctx', data=self.early_avg, col='label',col_order=self.cf_labels)
            #height=8, aspect=10/8)


#op,cmat,pmat = group_op(save_dict,fs)
def group_op(roi='circuit_mask',ext_split=True):
    if ext_split: classes = ['baseline', 'conditioning', 'early_extinction', 'late_extinction']
    else:         classes = ['baseline', 'conditioning', 'extinction']
    ACC = {}
    CF = {}

    for sub in all_sub_args:
        sub_res = op_decode(sub=sub,roi=roi,ext_split=ext_split)
        ACC[sub] = sub_res.xval_score
        CF[sub] = sub_res.cf_mat

    acc = pd.DataFrame.from_dict(ACC,orient='index'
                    ).reset_index(
                    ).rename(columns={'index':'subject',0:'accuracy'})
    acc['group'] = acc.subject.apply(lgroup)
    fig, ax = plt.subplots()
    sns.boxplot(data=acc,x='group',y='accuracy',palette=gpal,ax=ax)
    ax.hlines( ( 1/len(classes)),ax.get_xlim()[0],ax.get_xlim()[1],
                color='grey',linestyle='--',linewidth=3)

    cf = np.stack(CF.values())
    cmat = cf[:24,:,:].mean(axis=0)
    pmat = cf[24:,:,:].mean(axis=0)


    plot_confusion_matrix(cmat,classes,normalize=True,title='Control')
    plot_confusion_matrix(pmat,classes,normalize=True,title='PTSD')

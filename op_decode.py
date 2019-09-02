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
import time

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









class op_decode():
	def __init__(self,sub=None,save_dict=None,fs=False,verbose=False):


		self.verbose=verbose
		self.subj = meta(sub)
		day1_phases = ['baseline','fear_conditioning','extinction']

		self.data = {phase: np.load(self.subj.bold_dir + save_dict[phase])[dataz] for phase in day1_phases}
		self.labels = {phase: glm_timing(sub,phase).phase_events(con=True) for phase in day1_phases} 

		#clean out the CS- trials
		self.data = {phase: np.delete(self.data[phase], np.where(self.labels[phase] == 'CS-')[0], axis=0) for phase in day1_phases}

		# self.train_labels = np.concatenate((np.repeat('baseline_csp',24),np.repeat('baseline_csm',24),
		# 							np.repeat('early_fear_csp',12),np.repeat('late_fear_csp',12),
		# 							np.repeat('early_fear_csm',12),np.repeat('late_fear_csm',12),
		# 							np.repeat('early_ext_csp',12),np.repeat('late_ext_csp',12),
		# 							np.repeat('early_ext_csm',12),np.repeat('late_ext_csm',12)))

		self.train_labels = np.concatenate((np.repeat('baseline',24),
									np.repeat('early_fear',12),np.repeat('late_fear',12),
									np.repeat('early_ext',12),np.repeat('late_ext',12)))
		self.classes = np.unique(self.train_labels)
		self.groups = np.tile([1,1,2,2],18)
		# self.train_labels = np.concatenate((np.repeat('baseline',48),np.repeat('fear_conditioning',48),np.repeat('extinction',48)))
		
		#concatenate the data
		self.data = np.concatenate((self.data['baseline'],self.data['fear_conditioning'],self.data['extinction']))

		self.init_fit_clf(data=self.data,labels=self.train_labels,fs=fs)
		
		self.op_res = self.predict(save_dict=save_dict)

		self.cfmat = self.cf_mat()

	def init_fit_clf(self,data=None,labels=None,fs=None):

		if self.verbose: print('Initializing Classifier')
		self.clf = Pipeline([ ('anova', SelectKBest(f_classif, k=fs)), ('clf', LogisticRegression() ) ])

		if self.verbose: print('Fitting Classifier to localizer')
		self.clf.fit(data, labels)

	def predict(self,save_dict):
		#load extinction recall and clean out CS-
		test_data = np.load(self.subj.bold_dir + save_dict['extinction_recall'])[dataz]
		test_data = np.delete(test_data, np.where(glm_timing(self.subj.num,'extinction_recall').phase_events(con=True) == 'CS-'), axis=0)

		if self.verbose: print(list(self.clf.classes_))

		# proba_res = self.clf.decision_function(test_data)
		proba_res = 1. / (1. + np.exp(-self.clf.decision_function(test_data)))

		_res = {label: proba_res[:,i] for i, label in enumerate(self.clf.classes_)}

		return _res

	def cf_mat(self):
	
		cf_mats = dict.fromkeys(list(np.unique(self.groups)))
		# auc = dict.fromkeys(list(np.unique(self.groups)))

		for run in np.unique(self.groups):
			_res = self.clf.fit(self.data[self.groups == run], self.train_labels[self.groups == run]).predict(self.data[self.groups != run])
			# print(MLB().fit_transform(_res))
			cf_mats[run] = confusion_matrix(self.train_labels[self.groups!=run], _res, labels=['baseline', 'early_fear', 'late_fear', 'early_ext','late_ext'])
			
			# cf_rep[run] = precision_recall_fscore_support(labels[groups != run], _res, labels=list(np.unique(labels)), average=None)[0]

		cf_mat_avg = np.mean( np.array( [cf_mats[1], cf_mats[2] ] ), axis=0 )


		# auc[run] = roc_auc_score(self.train_labels[self.groups!=run], self.clf.fit(self.data[self.groups == run], self.train_labels[self.groups == run]).decision_function(self.data[self.groups !=run]), average='samples')



		return cf_mat_avg

#op,cmat,pmat = group_op(save_dict,fs)
def group_op(save_dict=None,fs='all',verbose=False):


	op = {}
	c_mat = None
	p_mat = None
	
	for sub in all_sub_args:
		sub_res = op_decode(sub=sub,save_dict=save_dict,fs=fs,verbose=verbose)
		op[sub] = sub_res.op_res

		if sub_res.subj.num < 100:
			if c_mat is not None: c_mat = c_mat + sub_res.cfmat
			else:c_mat = sub_res.cfmat
		elif sub_res.subj.num > 100:
			if p_mat is not None: p_mat = p_mat + sub_res.cfmat
			else:p_mat = sub_res.cfmat

	op = pd.DataFrame.from_dict({(sub,label): op[sub][label]
												for sub in op.keys()
												for label in op[sub].keys()})

	op = op.reset_index().melt(id_vars=['index'])
	op.rename(columns={'index':'tr','variable_0':'subject','variable_1':'condition','value':'evidence'},inplace=True)

	op['Group'] = np.concatenate((np.repeat('Control',len(np.where(op['subject'] < 100)[0])),
									np.repeat('PTSD',len(np.where(op['subject'] > 100)[0])) ))
	c_mat = c_mat/len(sub_args)
	p_mat = p_mat/len(p_sub_args)


	# return op, c_mat, p_mat

	# fig, g = plt.subplots()
	# g = sns.FacetGrid(op,col='condition',hue='Group')
	# g = g.map(sns.lineplot, 'tr','evidence').add_legend()

	classes = ['baseline', 'early_fear', 'late_fear', 'early_ext','late_ext']
	plot_confusion_matrix(c_mat,classes,normalize=True,title='Control')#,cmap=wes_palettes['Zissou'])
	plot_confusion_matrix(p_mat,classes,normalize=True,title='PTSD')#,cmap=wes_palettes['Zissou'])
	# ax = sns.lineplot(x='tr',y='evidence',hue='condition',data=op)

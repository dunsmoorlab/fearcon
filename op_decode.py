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
from sklearn.preprocessing import MultiLabelBinarizer as MLB
from nilearn.input_data import NiftiMasker
from nilearn.masking import apply_mask


class cross_decode():
	def __init__(self,subjects=None,beta_dir='/Volumes/DunsmoorRed/std_betas',mask=None):

		print(subjects)
		
		self.subjects = subjects
		self.beta_dir = beta_dir

		self.masker = NiftiMasker(mask_img=mask, standardize=True)

	def load_data(self):

		self.data = {}

		for sub in self.subjects:
			self.data[sub] = {}
			subj = meta(sub)
			
			for phase, run in zip(['baseline','fear_conditioning','extinction','renewal'],range(1,5)):
				beta = os.path.join(self.beta_dir,subj.fsub,'beta_std','run00%s_beta_std.nii.gz'%(run))
				print(phase)
				self.data[sub][phase] = self.masker.fit_transform(beta)




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

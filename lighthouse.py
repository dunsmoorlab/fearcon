import os
import numpy as np
import nibabel as nib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import make_pipeline, Pipeline 


from fc_config import mvpa_prepped

#set up the RSA object
class rsa(object):
	#initialize some base variables that will be useful
	
	#set the TR shift for the analysis
	hdr_shift = 3
	end_shift = 6 - hdr_shift

	data_dir = '/Users/ach3377/GoogleDrive/FC_FMRI_DATA/'

	def __init__(self, subj):
		
		print('WAWDHUAWDHOADW')
		
		
		self.subj = subj

		self.subj_dir = os.path.join(self.data_dir, 'Sub{0:0=3d}'.format(self.subj))

		self.bold_dir = os.path.join(self.subj_dir,'bold')

		self.label_dir = os.path.join(self.subj_dir,'model/MVPA/labels')

		#load in the meta data here
		self.meta = pd.read_csv(os.path.join(self.subj_dir,'behavior','Sub{0:0=3d}_elog.csv'.format(self.subj)))

		#see what the csplus and csminus are for this sub
		if self.meta['DataFile.Basename'][0][0] == 'A':
			self.csplus = 'animal'
			self.csminus = 'tool'
		elif self.meta['DataFile.Basename'][0][0] == 'T':
			self.csplus = 'tool'
			self.csminus = 'animal'

		#load these during initialization
		self.load_localizer()
		self.load_test_data()
		self.feature_reduction()

	
	def load_localizer(self, loc_runs=['localizer_1','localizer_2']):

		#load in the localizer data
		self.loc_data = { phase: np.load(os.path.join(self.bold_dir,mvpa_prepped[phase])) for phase in loc_runs }

		#shift over for HDR
		self.loc_data = { phase: self.loc_data[phase][self.hdr_shift:-self.end_shift] for phase in loc_runs }

		#load in the labels
		self.loc_labels = { phase: np.load(os.path.join(self.label_dir,'%s_labels.npy'%(phase))) for phase in loc_runs }

		#find TRs of rest
		self.loc_rest_index = { phase: np.where(self.loc_labels[phase]=='rest') for phase in loc_runs }

		#delete rest from the localizer runs and labels
		for phase in loc_runs:
			self.loc_data[phase] = np.delete(self.loc_data[phase], self.loc_rest_index[phase], axis=0)

			self.loc_labels[phase] = np.delete(self.loc_labels[phase], self.loc_rest_index[phase], axis=0)

			#collapse across scenes
			for i, label in enumerate(self.loc_labels[phase]):
				if label == 'indoor' or label == 'outdoor':
					self.loc_labels[phase][i] = 'scene'
				if label == self.csplus:
					self.loc_labels[phase][i] = 'CS+'
				if label == self.csminus:
					self.loc_labels[phase][i] = 'CS-'


		#combine runs and labels for model fitting
		self.loc_data = np.concatenate([self.loc_data['localizer_1'], self.loc_data['localizer_2']])

		self.loc_labels = np.concatenate([self.loc_labels['localizer_1'], self.loc_labels['localizer_2']])


	def load_test_data(self,test_runs=['baseline','fear_conditioning','extinction','memory_run_1','memory_run_2','memory_run_3']):

		self.test_data = { phase: np.load(os.path.join(self.bold_dir,mvpa_prepped[phase])) for phase in test_runs }
		
		#shift over for HDR
		self.test_data = { phase: self.test_data[phase][self.hdr_shift:-self.end_shift] for phase in test_runs }

		#load in labels
		self.test_labels = { phase: np.load(os.path.join(self.label_dir,'%s_labels.npy'%(phase))) for phase in test_runs }

		#find TRs of rest
		self.test_rest_index = { phase: np.where(self.test_labels[phase]=='rest') for phase in test_runs }
		self.test_rest_index['extinction'] = np.where(self.test_labels['extinction'] == 'scene')

		#delete rest from runs and labels
		for phase in test_runs:
			self.test_data[phase] = np.delete(self.test_data[phase], self.test_rest_index[phase], axis=0)

			self.test_labels[phase] = np.delete(self.test_labels[phase], self.test_rest_index[phase], axis=0)


	def feature_reduction(self,k=1000):

		self.classif_alg = LogisticRegression()

		self.feature_selection = SelectKBest(f_classif,k)

		self.clf = Pipeline([('anova', self.feature_selection), ('alg', self.classif_alg)])

		self.clf.fit(self.loc_data, self.loc_labels)

		#reshape the test_data to 1000 voxels using the localizer data
		for phase in self.test_data:
			self.test_data[phase] = self.feature_selection.fit(self.loc_data,self.loc_labels).transform(self.test_data[phase])


	def mean_patterns(self):
		#get index of csplus
		self.csplus_index = { phase: np.where(self.test_labels[phase] == 'CS+') for phase in self.test_labels }
		#get index of csminus
		self.csmin_index = { phase: np.where(self.test_labels[phase] == 'CS-') for phase in self.test_labels }

		#collect the mean patterns for csplus and csmin in each phase
		self.mean_csplus = { phase: self.test_data[phase][self.csplus_index[phase]].mean(axis=0) for phase in self.test_data }

		self.mean_csmin = { phase: self.test_data[phase][self.csmin_index[phase]].mean(axis=0) for phase in self.test_data }

	def compare_pats(self)

		self.comp = np.corrcoef(self.mean_csplus[phase],self.mean_csmin[phase])
		#need to just replicate the feature selection and then we can move on to loading in the other runs and then
		#collecting the patterns






























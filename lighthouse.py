import os
import numpy as np
import nibabel as nib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import make_pipeline, Pipeline 


from fc_config import mvpa_prepped, dataz, data_dir, hdr_shift

#set up the RSA object
class rsa(object):
	

	loc_runs=['localizer_1','localizer_2']

	test_runs=['baseline','fear_conditioning','extinction','memory_run_1','memory_run_2','memory_run_3']


	def __init__(self, subj):
		
		print('WAWDHUAWDHOADW')
		
		
		self.subj = subj

		self.subj_dir = os.path.join(data_dir, 'Sub{0:0=3d}'.format(self.subj))

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
		self.collect_stim_index()
		self.delete_rest()
		self.feature_reduction()
		self.unique_stims()
		self.mean_patterns()
		self.compare_pats()

	
	def load_localizer(self):

		#load in the localizer data
		self.loc_data = { phase: np.load(os.path.join(self.bold_dir,mvpa_prepped[phase]))[dataz] for phase in self.loc_runs }

		#shift over for HDR
		self.loc_data = { phase: self.loc_data[phase][hdr_shift['start']:-hdr_shift['end']] for phase in self.loc_runs }

		#load in the labels
		self.loc_labels = { phase: np.load(os.path.join(self.label_dir,'%s_labels.npy'%(phase))) for phase in self.loc_runs }

		#find TRs of rest
		self.loc_rest_index = { phase: np.where(self.loc_labels[phase]=='rest') for phase in self.loc_runs }

		#delete rest from the localizer runs and labels
		for phase in self.loc_runs:
			self.loc_data[phase] = np.delete(self.loc_data[phase], self.loc_rest_index[phase], axis=0)

			self.loc_labels[phase] = np.delete(self.loc_labels[phase], self.loc_rest_index[phase], axis=0)

			#collapse across scenes and rename animal/tool to CS+
			for i, label in enumerate(self.loc_labels[phase]):
				if label == 'indoor' or label == 'outdoor':
					self.loc_labels[phase][i] = 'scene'
				if label == self.csplus:
					self.loc_labels[phase][i] = 'CS+'
				if label == self.csminus:
					self.loc_labels[phase][i] = 'CS-'

		#find TRs of scenes and then remove them from data and labels
		self.loc_scene_index = { phase: np.where(self.loc_labels[phase]=='scene') for phase in self.loc_runs }

		for phase in self.loc_runs:

			self.loc_data[phase] = np.delete(self.loc_data[phase], self.loc_scene_index[phase], axis=0)

			self.loc_labels[phase] = np.delete(self.loc_labels[phase], self.loc_scene_index[phase], axis=0)

		#do the same for scrambled
		self.loc_scrambled_index = { phase: np.where(self.loc_labels[phase]=='scrambled') for phase in self.loc_runs }

		for phase in self.loc_runs:

			self.loc_data[phase] = np.delete(self.loc_data[phase], self.loc_scrambled_index[phase], axis=0)
			
			self.loc_labels[phase] = np.delete(self.loc_labels[phase], self.loc_scrambled_index[phase], axis=0)


		#combine runs and labels for model fitting
		self.loc_data = np.concatenate([self.loc_data['localizer_1'], self.loc_data['localizer_2']])

		self.loc_labels = np.concatenate([self.loc_labels['localizer_1'], self.loc_labels['localizer_2']])


	def load_test_data(self):

		self.test_data = { phase: np.load(os.path.join(self.bold_dir,mvpa_prepped[phase]))[dataz] for phase in self.test_runs }
		
		#shift over for HDR
		self.test_data = { phase: self.test_data[phase][hdr_shift['start']:-hdr_shift['end']] for phase in self.test_runs }

		#load in labels
		self.test_labels = { phase: np.load(os.path.join(self.label_dir,'%s_labels.npy'%(phase))) for phase in self.test_runs }

		self.test_labels = { phase: self.test_labels[phase].astype('|U8') for phase in self.test_runs}

	def delete_rest(self):
		#find TRs of rest
		self.test_rest_index = { phase: np.where(self.test_labels[phase]=='rest') for phase in self.test_runs }
		self.test_rest_index['extinction'] = np.where(self.test_labels['extinction'] == 'scene')
		self.test_rest_index['extinction'] = np.append(self.test_rest_index['extinction'], np.where(self.test_labels['extinction'] == 'rest'))

		#delete rest from runs and labels
		for phase in self.test_runs:
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
		self.csplus_index = { phase: [i for i, label in enumerate(self.test_labels[phase]) if 'CS+' in label] for phase in self.test_labels }
		#get index of csminus
		self.csmin_index = { phase: [i for i, label in enumerate(self.test_labels[phase]) if 'CS-' in label] for phase in self.test_labels }

		#collect the mean patterns for csplus and csmin in each phase
		self.mean_csplus = { phase: self.test_data[phase][self.csplus_index[phase]].mean(axis=0) for phase in self.test_data }

		self.mean_csmin = { phase: self.test_data[phase][self.csmin_index[phase]].mean(axis=0) for phase in self.test_data }

		#np.where(self.test_labels[phase] == 'CS+')

	def phase_pattern_corr(self, _cond, _phase):

		if _cond == 'CS+':
			self.comp_cond = self.mean_csplus
		elif _cond == 'CS-':
			self.comp_cond = self.mean_csmin
		
		self.csplus_res = [ np.corrcoef(self.comp_cond[_phase], self.mean_csplus[phase] )[0][1] for phase in self.mean_csplus ]
		self.csmin_res = [ np.corrcoef(self.comp_cond[_phase], self.mean_csmin[phase] )[0][1] for phase in self.mean_csmin ]

		return(np.vstack((self.csplus_res,self.csmin_res)).reshape((-1),order='F') )


	def stim_pattern_corr(self,_stim):

		self.stim_corr = [ np.corrcoef( self.comp_these_stims[_stim], self.comp_these_stims[stim] )[0][1] for stim in self.comp_these_stims ]

		return self.stim_corr


	def compare_pats(self):
		self.cond_names = ['base_CS+','base_CS-','fear_CS+','fear_CS-','ext_CS+','ext_CS-','mem_1_CS+','mem_1_CS-','mem_2_CS+','mem_2_CS-','mem_3_CS+','mem_3_CS-']
		self.comp_mat = pd.DataFrame([], index = self.cond_names, columns = self.cond_names)

		for cond in self.comp_mat.columns:
			if 'CS+' in cond:
				if 'base' in cond:
					self.comp_mat[cond] = self.phase_pattern_corr('CS+','baseline')
				if 'fear' in cond:
					self.comp_mat[cond] = self.phase_pattern_corr('CS+','fear_conditioning')
				if 'ext' in cond:
					self.comp_mat[cond] = self.phase_pattern_corr('CS+','extinction')
				if 'mem_1' in cond:
					self.comp_mat[cond] = self.phase_pattern_corr('CS+','memory_run_1')
				if 'mem_2' in cond:
					self.comp_mat[cond] = self.phase_pattern_corr('CS+','memory_run_2')
				if 'mem_3' in cond:
					self.comp_mat[cond] = self.phase_pattern_corr('CS+','memory_run_3')
			if 'CS-' in cond:
				if 'base' in cond:
					self.comp_mat[cond] = self.phase_pattern_corr('CS-','baseline')
				if 'fear' in cond:
					self.comp_mat[cond] = self.phase_pattern_corr('CS-','fear_conditioning')
				if 'ext' in cond:
					self.comp_mat[cond] = self.phase_pattern_corr('CS-','extinction')
				if 'mem_1' in cond:
					self.comp_mat[cond] = self.phase_pattern_corr('CS-','memory_run_1')
				if 'mem_2' in cond:
					self.comp_mat[cond] = self.phase_pattern_corr('CS-','memory_run_2')
				if 'mem_3' in cond:
					self.comp_mat[cond] = self.phase_pattern_corr('CS-','memory_run_3')		
		

	def collect_stim_index(self):
		
		self.reset_counter()

		for phase in self.test_labels:

			self.csplus_counter = 1
			self.csmin_counter = 1

			for i, label in enumerate(self.test_labels[phase]):

				if label == 'CS+':

					self.comp_stim = 'CS+'

				elif label == 'CS-':

					self.comp_stim = 'CS-'

				if label == self.comp_stim:

					self.sliding_window.extend([i])

				if label != self.comp_stim:
					
					if self.comp_stim == 'CS+':

						self.test_labels[phase][self.sliding_window] = 'CS+_{0:0=2d}'.format(self.csplus_counter)

						self.csplus_counter += 1

						self.reset_counter()

					elif self.comp_stim == 'CS-':

						self.test_labels[phase][self.sliding_window] = 'CS-_{0:0=2d}'.format(self.csmin_counter)

						self.csmin_counter += 1

						self.reset_counter()

	def reset_counter(self):

		self.comp_stim = ''
		self.sliding_window = []


	def unique_stims(self):

		self.unique = { phase: np.unique(self.test_labels[phase]) for phase in self.test_labels }


	def phase_stim_patterns(self,phase):

		self.mean_stim_pats = {}

		for stim in self.unique[phase]:

			self.stim_pats = self.test_data[phase][np.where(self.test_labels[phase] == stim)]

			self.mean_stim_pats[stim] = self.stim_pats.mean(axis=0)

		return self.mean_stim_pats


	def comp_phase_stim_patterns(self,phase):

		self.comp_these_stims = self.phase_stim_patterns(phase)

		self.phase_comp_mat = pd.DataFrame([], columns = self.unique[phase], index = self.unique[phase])

		for stim in self.phase_comp_mat.columns:

			self.phase_comp_mat[stim] = self.stim_pattern_corr(stim)

		return self.phase_comp_mat





#and then generate the stimuli specific mean patterns
























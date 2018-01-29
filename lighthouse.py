import os
import numpy as np
import nibabel as nib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import make_pipeline, Pipeline 


from fc_config import mvpa_prepped, dataz, data_dir, hdr_shift, init_dirs, fsub

#set up the RSA object
class rsa(object):
	

	loc_runs=['localizer_1','localizer_2']

	test_runs=['baseline','fear_conditioning','extinction','memory_run_1','memory_run_2','memory_run_3']


	def __init__(self, subj):
		
		
		self.subj = subj

		self.subj_dir, self.bold_dir = init_dirs(subj)

		self.fsub = fsub(subj)

		print('creating rsa object for %s'%(self.fsub))
		
		self.label_dir = os.path.join(self.subj_dir,'model/MVPA/labels')

		#load in the meta data here
		self.meta = pd.read_csv(os.path.join(self.subj_dir,'behavior','Sub{0:0=3d}_elog.csv'.format(self.subj)))

		#self.cs_lookup()

		#load these during initialization
		#self.load_localizer()
		#self.load_test_data()
		#self.collect_stim_index()
		#self.delete_rest()
		#self.feature_reduction()
		#self.unique_stims()
		#self.mean_patterns(test_labels=self.test_labels,test_data=self.test_data)
		#self.compare_mean_pats()

	
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


	#update 1/25/17
	#re-written to handle general format data
	#in its original context, it should now be implemented as:
	#feature_reduction(k=1000, loc_dat=self.loc_data, loc_lab=self.loc_labels, test_dat=self.test_data)
	def feature_reduction(self,k=1000,train_dat=None,train_lab=None,reduce_dat=None):

		classif_alg = LogisticRegression()

		feature_selection = SelectKBest(f_classif,k)

		self.clf = Pipeline([('anova', feature_selection), ('alg', classif_alg)])

		self.clf.fit(train_dat, train_lab)

		#reshape the test_data to 1000 voxels using the localizer data
		for phase in reduce_dat:
			reduce_dat[phase] = feature_selection.fit(train_dat, train_lab).transform(reduce_dat[phase])


	def mean_patterns(self,test_labels=None,test_data=None):
		#get index of csplus
		self.csplus_index = { phase: [i for i, label in enumerate(test_labels[phase]) if 'CS+' in label] for phase in test_labels }
		#get index of csminus
		self.csmin_index = { phase: [i for i, label in enumerate(test_labels[phase]) if 'CS-' in label] for phase in test_labels }

		#collect the mean patterns for csplus and csmin in each phase
		self.mean_csplus = { phase: test_data[phase][self.csplus_index[phase]].mean(axis=0) for phase in test_data }

		self.mean_csmin = { phase: test_data[phase][self.csmin_index[phase]].mean(axis=0) for phase in test_data }

		#np.where(self.test_labels[phase] == 'CS+')

	def phase_pattern_corr(self, _cond, _phase):

		if _cond == 'CS+':
			comp_cond = self.mean_csplus
		elif _cond == 'CS-':
			comp_cond = self.mean_csmin
		
		csplus_res = [ np.corrcoef(comp_cond[_phase], self.mean_csplus[phase] )[0][1] for phase in self.mean_csplus ]
		csmin_res = [ np.corrcoef(comp_cond[_phase], self.mean_csmin[phase] )[0][1] for phase in self.mean_csmin ]

		return(np.vstack((csplus_res,csmin_res)).reshape((-1),order='F') )


	def stim_pattern_corr(self,_stim=None,comp_these_stims=None):

		stim_corr = [ np.corrcoef( comp_these_stims[_stim], comp_these_stims[stim] )[0][1] for stim in comp_these_stims ]

		return stim_corr


	def compare_mean_pats(self):
		cond_names = ['base_CS+','base_CS-','fear_CS+','fear_CS-','ext_CS+','ext_CS-','mem_1_CS+','mem_1_CS-','mem_2_CS+','mem_2_CS-','mem_3_CS+','mem_3_CS-']
		self.comp_mat = pd.DataFrame([], index = cond_names, columns = cond_names)

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


	def unique_stims(self,test_labels=None):

		self.unique = { phase: np.unique(test_labels[phase]) for phase in test_labels }

	#in the case of beta_rsa the counted_labels and the labels are the same
	#other wise this counted_labels=self.unique and test_labels=self.test_labels, test_data = self.test_data
	#BUT ACTUALLY DON'T NEED TO DO THIS FOR BETA RSA BECAUSE WE ALREADY ONLY HAVE 1 "VOLUME" PER STIM
	#BUT, we do it anyways because it creates a dictionary in which each trial is associated with its CS_00 handle
	def phase_stim_patterns(self,phase=None, counted_labels=None, test_data=None, test_labels=None):

		mean_stim_pats = {}

		for stim in counted_labels[phase]:

			stim_pats = test_data[phase][np.where(test_labels[phase] == stim)]

			mean_stim_pats[stim] = stim_pats.mean(axis=0)

		return mean_stim_pats

	#most of these arguments are here just to pass to phase_stim_patterns()
	def comp_phase_stim_patterns(self,phase, counted_labels=None, test_data=None, test_labels=None):

		comp_these_stims = self.phase_stim_patterns(phase=phase, counted_labels=counted_labels, test_data=test_data, test_labels=test_labels)

		phase_comp_mat = pd.DataFrame([], columns=counted_labels[phase], index=counted_labels[phase])

		for stim in phase_comp_mat.columns:

			phase_comp_mat[stim] = self.stim_pattern_corr(_stim=stim, comp_these_stims=comp_these_stims)

		return phase_comp_mat


	#see what the csplus and csminus are for this sub
	def cs_lookup(self):	
		if self.meta['DataFile.Basename'][0][0] == 'A':
			self.csplus = 'animal'
			self.csminus = 'tool'
		elif self.meta['DataFile.Basename'][0][0] == 'T':
			self.csplus = 'tool'
			self.csminus = 'animal'




#and then generate the stimuli specific mean patterns
























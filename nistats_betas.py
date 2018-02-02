'''
Augustin Hennings 2018
Dunsmoor/Lewis-Peacock Labs
UT Austin
'''


'''
The following is a script designed to generate ls-s style betas a la Mumford et al., 2012
Steps to using this script are:
	1. read what an ls-s beta estimate is
	2. generating your own config variables
	3. DO SOME QUALITY CONTROL
This scripts implements nistats for general linear model estimation, this is a really cool & great pythonic package,
but it is extremely new (alpha release) and updates to the core could easily break this script

Thanks & good luck
'''


'''
this script should not be run directly, but should be referenced in a seperate wrapper script
for example: group_beta(phase='fear_conditioning', overwrite=False)
where group_beta is the following class:


from nistats_betas import generate_lss_betas
from fc_config import sub_args

def group_beta(phase=None,overwrite=False):

	for sub in sub_args:

		generate_lss_betas(subj=sub, phase=phase, tr=2, hrf_model='glover', display=False, overwrite=overwrite)
'''



import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib


#these are variables that are specific to my project, but have descriptive names
#replace them here with things that make sense for your own data structure
#take a look at fc_config.py if you need more info
from fc_config import data_dir, init_dirs, nifti_paths, py_run_key

#this is my own script, won't work for anything but FearCon
from glm_timing import glm_timing

from nistats.first_level_model import FirstLevelModel

from nistats.reporting import plot_design_matrix



class generate_lss_betas(object):

	#takes in subject, phase (which run), tr, and hemodynamic response function
	#recommend leaving 'display' off (False), other wise you will see each design matrix which could be a lot 
	def __init__(self,subj=0,phase=None,tr=0,hrf_model=None,display=False,overwrite=False):

		self.subj = subj
		#formatted sub name
		self.fsub = 'Sub{0:0=3d}'.format(self.subj)

		print('initializing %s'%(self.fsub))

		#point to the subject, bold, and run directories
		self.subj_dir, self.bold_dir, self.run_dir = init_dirs(subj, phase)

		#set the display variable (for looking at design matrices & such)
		self.display = display


		#initialize output directories
		self.beta_out_dir =  os.path.join(self.run_dir, 'ls-s_betas')

		if not os.path.exists(self.beta_out_dir):

			os.mkdir(self.beta_out_dir)

		#if overwrite is set to False and there are already betas in the out_dir, just skip this subject
		if not overwrite:
			if len(os.listdir(self.beta_out_dir)) > 10:
				print('Skipping %s, overwrite=False'%(self.fsub))
			else:
				overwrite=True

		#otherwise make the betas
		if overwrite:
			#generate the events structure
			#this is going to be different for every experiment, and will require you to write your own code
			#the format should be a pandas DataFrame, with the columns: duration, onset, & trial_type
			self.events = glm_timing(subj,phase).phase_events()

			#something weird was happening with my phase events, this should prevent it from happening again
			if self.events.isnull().values.any():
				sys.exit('null value encountered in glm_timing(%s,%s).phase_events()'%(subj,phase))

			#do work
			self.load_bold(phase=phase)

			self.init_glm(tr=tr,hrf_model=hrf_model)


			#real work happens here
			self.beta_loop()


	def load_bold(self,phase):
		print('loading %s'%(phase))

		#need path to motion corrected functional run
		self.func = os.path.join(self.bold_dir,nifti_paths[phase])


	def create_mean(self):

		#create the mean functional image, which is only needed for graphing,
		#this also should be maybe be moved to a wrapper script if not needed consistently
		self.mean_func = image.mean_img(self.func)


	def init_glm(self,tr=0,hrf_model=None):

		print('initializing nistats GLM; tr=%s, hrf_model=%s'%(tr,hrf_model))

		#init the model
		#this is what needs the most work in terms of reading documentation and tweaking parameters
		#https://nistats.github.io/modules/reference.html#module-nistats.first_level_model
		self.glm = FirstLevelModel(t_r=tr, slice_time_ref=0, hrf_model=hrf_model, n_jobs=4)


	def fit_glm(self, beta_model):
		'''
		#fit the GLM
		Normally the glm would be fit to all events, but ls-s style beta estimation is different
		the glm has to be fit iteratively over all trials -
		each time there is one regressor for the trial of interest and another regressor for all
		other trials
		'''
		self.glm_fit = self.glm.fit(self.func, beta_model)


	def display_design_matrix(self,design_matrix=None):

		#self explanatory, use or lose
		plot_design_matrix(design_matrix)
		plt.show()


	def beta_loop(self):

		#set the number of betas to estimate to the number of trials
		self.ntrials = len(self.events.index)

		print('generating %s ls-s betas'%(self.ntrials))

		self.beta_contrast=None
		#set up a loop through all trials
		for target_trial in self.events.index:

			print('trial %s'%(target_trial+1))

			#identify all the non-target trials
			non_targets = np.concatenate((np.array(range(0,target_trial)), np.array(range(target_trial + 1, self.ntrials))), axis=0)
			
			#copy the events structure
			beta_event = self.events.copy()

			#set the trial type of target trial to 'trial_beta'
			beta_event.loc[target_trial,'trial_type'] = 'trial_beta'

			#set all the other trial types to 'non_target'
			beta_event.loc[non_targets,'trial_type'] = 'non_target'

			#fit the glm using this new event structure
			self.fit_glm(beta_model=beta_event)

			#grab the design matrix
			beta_design_matrix = self.glm_fit.design_matrices_[0]

			if self.display:
				self.display_design_matrix(beta_design_matrix)

			#the contrast will actually be the same for all trials, since its just a vector:
			#[-1,1,0,0] or something similar. So we only need to create it the first time
			if self.beta_contrast is None:
				self.beta_contrast = self.set_contrasts(beta_design_matrix)

			#generate the beta estimation
			beta_z_map = self.glm_stats(self.beta_contrast)

			#save it!
			self.save_stats(beta_z_map,target_trial)


	def set_contrasts(self,design_matrix=None):

		#create the contrast matrix with one 1 in each row corresponding to its intercept
		#(np.eye takes care of this)
		contrast_matrix = np.eye(design_matrix.shape[1])

		#fill out the rest of the contrast matrix with variables from the design matrix
		contrasts = dict([(column, contrast_matrix[i]) for i, column in enumerate(design_matrix.columns)])

		'''
		set up the beta contrast
		this is designed to generate an esitmate for activity specific to this trial relative to
		all other trials. See Mumford et al. (2012) for more info on ls-s beta estimation
 		'''

		beta_contrast = contrasts['trial_beta'] - contrasts['non_target']

		return beta_contrast

	def glm_stats(self,contrast):

		#the effect map can be saved and computed as well, but for now just using z_map
		z_map = self.glm_fit.compute_contrast(contrast, output_type='z_score')

		return z_map


	def save_stats(self,img2save=None,trialNum=None):

		#save the beta with the fromat beta_000.nii.gz, with the 000 corresponding to trial number
		#trialNum + 1 to correct for pythonic indexing
		nib.save(img2save, os.path.join(self.beta_out_dir,'beta_{0:0=3d}.nii.gz'.format(trialNum + 1)))










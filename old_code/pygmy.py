import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib

from nilearn.plotting import plot_stat_map, plot_anat, plot_img
from nilearn.image import mean_img, image, new_img_like
from nilearn.input_data import NiftiMasker

from nistats.first_level_model import FirstLevelModel
from nistats.reporting import plot_design_matrix

from fc_config import data_dir, get_subj_dir, get_bold_dir, py_run_key, nifti_paths, phase2rundir

from glm_timing import glm_timing

from preprocess_library import meta


class first_level(object):

	#these are some FearCon specific/global variables
	tr = 2
	#glover is cannonical double gamma HRF - derivative goes unused and should be removed if I don't figure out what it does
	hrf_model = 'glover + derivative'
	#slice time reference
	slice_time_ref = .5

	def __init__(self,subj=0,phase=None,day1_mem=False,interaction=False,display=False):

		'''
		some weird stuff happens the way i load in the motion confounds so this is necessary
		'''

		#init some paths
		self.subj = meta(subj)

		print('%s %s'%(self.subj.fsub,phase))

		self.phase = phase
		'''
		###this actually goes unused now so need to figure out if it needs to stay
		#which anatomical are we using
		self.anatomical_type = 'struct_brain.nii.gz'

		self.struct = os.path.join(self.subj_dir,'anatomy',self.anatomical_type)
		'''
		
		
		#init and create the output directories if needed
		self.pyGLM_dir = os.path.join(self.subj.subj_dir,'model','pyGLM')

		self.lev1_dir = os.path.join(self.pyGLM_dir,py_run_key[phase])

		if not os.path.exists(self.pyGLM_dir):

			os.mkdir(self.pyGLM_dir)

		if not os.path.exists(self.lev1_dir):
		
			os.mkdir(self.lev1_dir)

		#generate the timing pandas df
		self.events = glm_timing(subj,phase).phase_events(mem=day1_mem, intrx=interaction)

		self.func = self.load_bold(phase)
		print('fitting glm')
		self.glm = self.init_glm()
		self.glm, self.design_matrix = self.fit_glm(self.glm, self.func, self.events)
		if display:
			self.display_design_matrix(self.design_matrix)
		self.set_contrasts(mem=day1_mem,interaction=interaction)

		if interaction:
			self.contrast_titles = ['CS+_hit__CS-_miss']
		# 	# self.contrast_titles = ['CS+_hit','CS+_miss','CS-_hit','CS-_miss']
			print(self.contrast_titles[0])
			self.glm_stats(self.glm, self.Interaction, self.contrast_titles[0])

		# 	# print(self.contrast_titles[1])
		# 	# self.glm_stats(self.glm, self.CSplus_Miss,self.contrast_titles[1])

		# 	# print(self.contrast_titles[2])
		# 	# self.glm_stats(self.glm, self.CSmin_Hit,self.contrast_titles[2])

		# 	# print(self.contrast_titles[3])
		# 	# self.glm_stats(self.glm, self.CSmin_Miss,self.contrast_titles[3])
	
		# #this is the meat of the program and should be moved to the pyGLM_lev1.py if more control is needed
		# elif day1_mem:
		# 	self.contrast_titles = ['Hit___Miss','Miss___Hit']
		# 	print(self.contrast_titles[0])
		# 	self.glm_stats(self.glm, self.Hit_minus_Miss,self.contrast_titles[0])
		# 	print(self.contrast_titles[1])
		# 	self.glm_stats(self.glm, self.Miss_minus_Hit,self.contrast_titles[1])

		# else:
		# 	self.contrast_titles = ['CSplus___CSmin','CSmin___CSplus']
		# 	print(self.contrast_titles[0])
		# 	self.glm_stats(self.glm, self.CSplus_minus_CSmin,self.contrast_titles[0])
		# 	print(self.contrast_titles[1])
		# 	self.glm_stats(self.glm, self.CSmin_minus_CSplus,self.contrast_titles[1])


	def load_bold(self,phase):

		#need path to motion corrected functional run
		func = os.path.join(self.subj.bold_dir,nifti_paths[phase])
		

		return func


	def create_mean(self,func):

		#create the mean functional image, which is only needed for graphing, this also should be maybe be moved to pyGLM_lev1.py
		mean_func = image.mean_img(func)
		return mean_func

	def init_glm(self):
				
		masker= NiftiMasker(mask_img=self.subj.brainmask, smoothing_fwhm=5, detrend=True,
									high_pass=0.0078, t_r=2, standardize=True).fit(imgs=self.func)


		#init the model
		#this is what needs the most work in terms of reading documentation and tweaking parameters
		glm = FirstLevelModel(t_r=first_level.tr, mask=masker, slice_time_ref=first_level.slice_time_ref,
									hrf_model=first_level.hrf_model,  signal_scaling=False)
		
		return glm
	

	def fit_glm(self, glm, func, events):

		motion_confounds = np.loadtxt(os.path.join(self.subj.bold_dir, phase2rundir[self.phase], 'motion_assess', 'confound.txt'))
		
		if len(motion_confounds) > 0:

			if len(motion_confounds.shape) == 2:
				_columns = range(motion_confounds.shape[1])
			elif len(motion_confounds.shape) == 1:
				_columns = range(1)

			motion_confounds_df = pd.DataFrame(motion_confounds, columns=_columns)

		else: 
			motion_confounds_df = None

		#fit the GLM
		glm = glm.fit(func, events, confounds=motion_confounds_df)

		#grab the design matrix
		design_matrix = glm.design_matrices_[0]

		return glm, design_matrix

	def display_design_matrix(self,design_matrix):

		#self explanatory
		plot_design_matrix(design_matrix)
		plt.show()

	def set_contrasts(self,mem=False, interaction=False):

		#create the contrast matrix with one 1 in each row corresponding to its intercept
		#(np.eye takes care of this)
		self.contrast_matrix = np.eye(self.design_matrix.shape[1])

		#fill out the rest of the contrast matrix with variables from the design matrix
		self.contrasts = dict([(column, self.contrast_matrix[i])
			for i, column in enumerate(self.design_matrix.columns)])

		#set up contrasts, these will be pretty stable over the life of FearCon
		if interaction:
			self.Interaction = self.contrasts['CS+_hit'] - self.contrasts['CS-_miss']
			# self.Interaction = self.contrasts['CS+_hit'] + (-1*self.contrasts['CS+_miss']) + (-1*self.contrasts['CS-_hit']) + self.contrasts['CS-_miss']


			# self.CSplus_Hit = self.contrasts['CS+_hit']
			# self.CSplus_Miss = self.contrasts['CS+_miss']
			# self.CSmin_Hit = self.contrasts['CS-_hit']
			#THIS WAS WRONG HAVE TO RERUN
			# self.CSmin_Miss = self.contrasts['CS-_miss']


		elif mem:
			self.Hit_minus_Miss = self.contrasts['hit'] - self.contrasts['miss']
			self.Miss_minus_Hit = self.contrasts['miss'] - self.contrasts['hit']
		
		else:
			self.CSplus_minus_CSmin = self.contrasts['CS+'] - self.contrasts['CS-']
			self.CSmin_minus_CSplus = self.contrasts['CS-'] - self.contrasts['CS+']

	def glm_stats(self,glm,contrast_,save_title=None):

		#generate effect maps
		# effect_size_map = glm.compute_contrast(contrast_, output_type='effect_size')
		
		#generate effect_variace maps
		# effect_var_map = glm.compute_contrast(contrast_, output_type='effect_variance')

		#generate z_score map
		z_map = glm.compute_contrast(contrast_, stat_type='F', output_type='z_score')

		if save_title is not None:
			# self.save_stats(stats=effect_size_map, _type='eff_size', title=save_title)
			# self.save_stats(stats=effect_var_map, _type='eff_var', title=save_title)
			self.save_stats(stats=z_map, _type='z_map', title=save_title)

	def save_stats(self,stats,_type,title):
		
		nib.save(stats, os.path.join(self.lev1_dir,'%s_%s.nii.gz'%(title,_type)))

		# nib.save(stats, os.path.join(self.lev1_dir,'%s_pval_map.nii.gz'%(title)))

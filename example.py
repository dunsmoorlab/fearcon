import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib

from nilearn.plotting import plot_stat_map, plot_anat, plot_img
from nilearn.image import mean_img, image, new_img_like

from nistats.first_level_model import FirstLevelModel
from nistats.reporting import plot_design_matrix

from fc_config import data_dir, get_subj_dir, get_bold_dir, nifti_paths, py_run_key

from glm_timing import glm_timing


class first_level(object):

	#these are some FearCon specific/global variables
	tr = 2
	#not sure how to use this yet
	slice_time_ref = 0
	#glover is cannonical double gamma HRF - derivative goes unused and should be removed if I don't figure out what it does
	hrf_model = 'glover + derivative'

	def __init__(self,subj=0,phase=None,display=False):

		#init some paths
		self.subj = subj

		self.fsub = 'Sub{0:0=3d}'.format(self.subj)

		print('%s first level GLM; %s'%(self.fsub,phase))

		self.subj_dir = get_subj_dir(subj)

		self.bold_dir = get_bold_dir(self.subj_dir)

		#which anatomical are we using
		###this actually goes unused now so need to figure out if it needs to stay
		#self.anatomical_type = 'struct_brain.nii.gz'

		#self.struct = os.path.join(self.subj_dir,'anatomy',self.anatomical_type)

		#init and create the output directories if needed
		self.pyGLM_dir = os.path.join(self.subj_dir,'model','pyGLM')

		self.lev1_dir = os.path.join(self.pyGLM_dir,py_run_key[phase])

		if not os.path.exists(self.pyGLM_dir):

			os.mkdir(self.pyGLM_dir)

		if not os.path.exists(self.lev1_dir):
		
			os.mkdir(self.lev1_dir)

		#generate the timing pandas df
		# self.events = glm_timing(subj,phase).phase_events()


		# self.load_bold(phase)
		# self.create_mean()
		# self.init_glm()
		# self.fit_glm()
		# if display:
		# 	self.display_design_matrix()
		# self.set_contrasts()

		# #this is the meat of the program and should be moved to the pyGLM_lev1.py if more control is needed
		# if phase == 'baseline' or phase == 'fear_conditioning':
			
		# 	self.contrast_titles = ['CSplus___CSmin','CSmin___CSplus']

		# 	self.glm_stats(self.CSplus_minus_CSmin,self.contrast_titles[0])
		# 	self.glm_stats(self.CSmin_minus_CSplus,self.contrast_titles[1])


	def load_bold(self,phase):

		#need path to motion corrected functional run
		self.func = os.path.join(self.bold_dir,nifti_paths[phase])

	def create_mean(self):

		#create the mean functional image, which is only needed for graphing, this also should be maybe be moved to pyGLM_lev1.py
		self.mean_func = image.mean_img(self.func)

	def init_glm(self):

		#init the model
		#this is what needs the most work in terms of reading documentation and tweaking parameters
		self.glm = FirstLevelModel(t_r=first_level.tr, slice_time_ref=first_level.slice_time_ref,
									hrf_model=first_level.hrf_model)

	def fit_glm(self):

		#fit the GLM
		self.glm = self.glm.fit(self.func,self.events)

		#grab the design matrix
		self.design_matrix = self.glm.design_matrices_[0]

	def display_design_matrix(self):

		#self explanatory
		plot_design_matrix(self.design_matrix)
		plt.show()

	def set_contrasts(self):

		#create the contrast matrix with one 1 in each row corresponding to its intercept
		#(np.eye takes care of this)
		self.contrast_matrix = np.eye(self.design_matrix.shape[1])

		#fill out the rest of the contrast matrix with variables from the design matrix
		self.contrasts = dict([(column, self.contrast_matrix[i])
			for i, column in enumerate(self.design_matrix.columns)])

		#set up contrasts, these will be pretty stable over the life of FearCon
		self.CSplus_minus_CSmin = self.contrasts['CS+'] - self.contrasts['CS-']
		self.CSmin_minus_CSplus = self.contrasts['CS-'] - self.contrasts['CS+']

	def glm_stats(self,contrast_,save_title):

		#generate effect maps
		self.effect_map = self.glm.compute_contrast(contrast_, output_type='effect_size')
		self.effect_map = self.coerce_4d(self.effect_map)
		
		#generate z_score map
		self.z_map = self.glm.compute_contrast(contrast_,output_type='z_score')
		self.z_map = self.coerce_4d(self.z_map)

		self.save_stats(save_title)

	def coerce_4d(self,img):
		
		#coerce both of them to be 4D arrays
		data = img.get_data().view()
		data.shape = data.shape + (1, )
		img = new_img_like(img, data, img.affine)

		return img

	def save_stats(self,title):

		#save both
		nib.save(self.effect_map, os.path.join(self.lev1_dir,'%s_eff_map.nii.gz'%(title)))

		nib.save(self.z_map, os.path.join(self.lev1_dir,'%s_z_map.nii.gz'%(title)))
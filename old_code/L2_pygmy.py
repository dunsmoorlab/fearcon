import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib

from nilearn.plotting import plot_stat_map, plot_anat, plot_img, plot_glass_brain
from nilearn.image import mean_img, image

from nistats.second_level_model import SecondLevelModel
from nistats.reporting import plot_design_matrix
from nilearn.input_data import MultiNiftiMasker

from scipy.stats import norm

from glob import glob

from fc_config import data_dir, get_subj_dir, get_bold_dir, nifti_paths, py_run_key, sub_args
from preprocess_library import meta


class second_level(object):


	def __init__(self, phase=None, day1_mem=False, interaction=False, stat_type='z_map', display=False):


		#first check to make sure the output dir is there
		self.group_pyGLM_dir = os.path.join(data_dir,'group_pyGLM')
		if not os.path.exists(self.group_pyGLM_dir):
			os.mkdir(self.group_pyGLM_dir)

		self.group_phase_dir = os.path.join(self.group_pyGLM_dir,py_run_key[phase])
		if not os.path.exists(self.group_phase_dir):
			os.mkdir(self.group_phase_dir)

		self.determine_contrasts(phase=phase, day1_mem=day1_mem, interaction=interaction)

		for contrast in self.contrasts:

			self.fetch_first_levels(phase,contrast,stat_type)
			
			if display:
				plt.ion()
				self.display_first_levels(contrast)

			self.glm_level2, self.design_matrix = self.init_level2(self.first_levels, smoothing=8)
			
			if display:
				plot_design_matrix(self.design_matrix)


			self.glm_level2, self.z_map = self.fit_level2(self.glm_level2, self.first_levels, self.design_matrix)

			if display:
				self.display_second_level(contrast)

			self.save_level2(contrast,stat_type)


	def determine_contrasts(self,phase, day1_mem, interaction):

		if interaction:
			self.contrasts = ['CS+_hit__CS-_miss']
			# self.contrasts = ['CS+_hit','CS+_miss','CS-_hit','CS-_miss']
			# self.contrasts = ['CS+_miss','CS-_miss']
		elif day1_mem:
			self.contrasts = ['Hit___Miss','Miss___Hit']
		else:
			self.contrasts = ['CSplus___CSmin','CSmin___CSplus']


	def fetch_first_levels(self,phase,contrast,stat_type):

		#first we need to warp the first level to standard space so they can be compared
		for sub in sub_args:
			
			subj = meta(sub)
			first_level = os.path.join(subj.model_dir, 'pyGLM', py_run_key[phase], contrast + '_%s.nii.gz'%(stat_type))
			print(sub, first_level)
			
			registered_level1 = os.path.join(subj.model_dir, 'pyGLM', py_run_key[phase], 'MNI_' + contrast + '_%s.nii.gz'%(stat_type))
			if not os.path.exists(registered_level1):
				print('warping level1 to MNI space')
				struct_warp_img = os.path.join(subj.reg_dir, 'struct2std_warp.nii.gz')
				func2struct = os.path.join(subj.reg_dir, 'func2struct.mat')
				os.system('applywarp --in=%s --ref=$FSL_DIR/data/standard/MNI152_T1_1mm_brain.nii.gz --out=%s --premat=%s --warp=%s'%(
							first_level, registered_level1, func2struct, struct_warp_img))


		self.first_levels = glob('%s/Sub[0-9][0-9][0-9]/model/pyGLM/%s/MNI_%s_%s.nii.gz'%(data_dir,py_run_key[phase],contrast,stat_type))


	def display_first_levels(self,contrast):

		fig, axes = plt.subplots(nrows=5,ncols=4)
		for sub, zmap in enumerate(self.first_levels):
			plot_glass_brain(zmap, colorbar=True, threshold=3, title='S%s'%(sub_args[sub]),
								axes=axes[int(sub/4),int(sub % 4)], 
								plot_abs=False, display_mode='z')
		fig.suptitle('%s'%(contrast))


	def init_level2(self, first_levels, smoothing=8):

		design_matrix = pd.DataFrame([1] * len(first_levels), columns=['intercept'])

		MNI_mask = '/usr/local/fsl/data/standard/MNI152_T1_1mm_brain_mask.nii.gz'

		glm_level2 = SecondLevelModel(mask=MNI_mask, smoothing_fwhm=smoothing)

		return glm_level2, design_matrix

	def fit_level2(self, glm_level2, first_levels, design_matrix):

		print('fitting second level GLM')

		glm_level2 = glm_level2.fit(first_levels, design_matrix=design_matrix)

		z_map = glm_level2.compute_contrast(output_type='z_score')

		return glm_level2, z_map

	def display_second_level(self,contrast,pval=0.01):

		self.z_thresh = norm.isf(pval)

		display = plot_glass_brain(self.z_map, threshold=self.z_thresh, colorbar=True, plot_abs=False,
									display_mode='z', title='Level 2 %s'%(contrast))
		plt.show()


	def save_level2(self,contrast,stat_type):

		nib.save(self.z_map, os.path.join(self.group_phase_dir,'%s_level2_%s.nii.gz'%(contrast,stat_type)))

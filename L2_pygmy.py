import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib

from nilearn.plotting import plot_stat_map, plot_anat, plot_img, plot_glass_brain
from nilearn.image import mean_img, image

from nistats.second_level_model import SecondLevelModel
from nistats.reporting import plot_design_matrix

from scipy.stats import norm

from glob import glob

from fc_config import data_dir, get_subj_dir, get_bold_dir, nifti_paths, py_run_key, sub_args


class second_level(object):


	def __init__(self,phase=None,display=False):


		#first check to make sure the output dir is there
		self.group_pyGLM_dir = os.path.join(data_dir,'group_pyGLM')
		if not os.path.exists(self.group_pyGLM_dir):
			os.mkdir(self.group_pyGLM_dir)

		self.group_phase_dir = os.path.join(self.group_pyGLM_dir,py_run_key[phase])
		if not os.path.exists(self.group_phase_dir):
			os.mkdir(self.group_phase_dir)




		self.determine_contrasts(phase)

		for contrast in self.contrasts:

			self.fetch_first_levels(phase,contrast)
			
			if display:
				plt.ion()
				self.display_first_levels(contrast)

			self.init_level2()
		
			self.fit_level2()

			if display:
				self.display_second_level(contrast)

			self.save_level2(contrast)


	def determine_contrasts(self,phase):

		if phase == 'baseline' or phase == 'fear_conditioning':

			self.contrasts = ['CSplus___CSmin','CSmin___CSplus']


	def fetch_first_levels(self,phase,contrast):

		self.first_levels = glob('%s/Sub[0-9][0-9][0-9]/model/pyGLM/%s/%s_z_map.nii.gz'%(data_dir,py_run_key[phase],contrast))


	def display_first_levels(self,contrast):

		fig, axes = plt.subplots(nrows=4,ncols=4)
		for sub, zmap in enumerate(self.first_levels):
			plot_glass_brain(zmap, colorbar=True, threshold=3, title='S%s'%(sub_args[sub]),
								axes=axes[int(sub/4),int(sub % 4)], 
								plot_abs=False, display_mode='z')
		fig.suptitle('%s'%(contrast))


	def init_level2(self,smoothing=8):

		self.design_matrix = pd.DataFrame([1] * len(self.first_levels), columns=['intercept'])

		self.glm_level2 = SecondLevelModel(smoothing_fwhm=smoothing)


	def fit_level2(self):

		self.glm_level2 = self.glm_level2.fit(self.first_levels, design_matrix=self.design_matrix)

		self.z_map = self.glm_level2.compute_contrast(output_type='z_score')


	def display_second_level(self,contrast,pval=0.01):

		self.z_thresh = norm.isf(pval)

		display = plot_glass_brain(self.z_map, threshold=self.z_thresh, colorbar=True, plot_abs=False,
									display_mode='z', title='Level 2 %s'%(contrast))
		plt.show()


	def save_level2(self,contrast):

		nib.save(self.z_map, os.path.join(self.group_phase_dir,'%s_level2_z_map.nii.gz'%(contrast)))

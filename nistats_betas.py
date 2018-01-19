import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib

#these are variables that are specific to my project, but have descriptive names
#replace them here with things that make sense for your own data structure
#take a look at fc_config.py if you need more info
from fc_config import data_dir, get_subj_dir, get_bold_dir, nifti_paths, py_run_key

#this is my own script, won't work for anything but FearCon
from glm_timing import glm_timing

from nistats.first_level_model import FirstLevelModel



class generate_lss_betas(object):

	#takes in subject, phase (which run), tr, and hemodynamic response function
	#recommend leaving 'display' off (False), other wise you will see each design matrix which could be a lot 
	def __init__(self,subj=0,phase=None,tr=0,hrf_model=None,display=False):

		self.subj = subj
		self.fsub = 'Sub{0:0=3d}'.format(self.subj)

		print('asiudghiasubfd')

		#point to the subject directory
		self.subj_dir = get_subj_dir(subj)

		#point to the bold directory within the subject directory
		self.bold_dir = get_bold_dir(self.subj_dir)


		#initialize output directories
		'''
		self.pyGLM_dir = os.path.join(self.subj_dir,'model','pyGLM')

		self.lev1_dir = os.path.join(self.pyGLM_dir,py_run_key[phase])

		if not os.path.exists(self.pyGLM_dir):

			os.mkdir(self.pyGLM_dir)

		if not os.path.exists(self.lev1_dir):
		
			os.mkdir(self.lev1_dir)
		'''

		#generate the events structure
		#this is going to be different for every experiment, and will require you to write your own code
		#the format should be a pandas DataFrame, with the columns: Duration, Onset, and Condition
		self.events = glm_timing(subj,phase).phase_events()


	def load_bold(self,phase):

		#need path to motion corrected functional run
		self.func = os.path.join(self.bold_dir,nifti_paths[phase])


	def create_mean(self):

		#create the mean functional image, which is only needed for graphing,
		#this also should be maybe be moved to a wrapper script if not needed consistently
		self.mean_func = image.mean_img(self.func)



	#init glm
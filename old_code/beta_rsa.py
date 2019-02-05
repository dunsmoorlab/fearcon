from lighthouse import rsa
from glm_timing import glm_timing
from glob import glob

import os
import numpy as np
import pandas as pd

import nibabel as nib
from nilearn.masking import apply_mask

from fc_config import data_dir, init_dirs, fsub, phase2rundir


class beta_rsa(object):

	test_runs=['baseline','fear_conditioning','extinction','extinction_recall','memory_run_1','memory_run_2','memory_run_3']

	def __init__(self, subj, k):
		
		self.rsa = rsa(subj)

		self.rsa.cs_lookup()

		self.load_loc_betas()
		self.load_test_betas()
		self.mask_runs()

		self.rsa.feature_reduction(k=k, train_dat=self.loc_betas, train_lab=self.loc_labels, reduce_dat=self.test_betas)
		self.rsa.unique_stims(test_labels=self.test_beta_labels)
		self.rsa.mean_patterns(test_labels=self.test_beta_labels, test_data=self.test_betas)
		self.rsa.compare_mean_pats()


	def load_loc_betas(self):

		#point to the betas
		loc_beta_files = { phase: glob('%s%sls-s_betas/beta_***.nii.gz'%(self.rsa.bold_dir,phase2rundir[phase])) for phase in rsa.loc_runs }

		#generate the trial labels
		loc_beta_labels = { phase: glm_timing(self.rsa.subj,phase).phase_events(con=True) for phase in rsa.loc_runs }

		#sort for and keep just animals and tools
		loc_beta_labels = { phase: loc_beta_labels[phase].loc[loc_beta_labels[phase].isin(['animals','tools'])] for phase in rsa.loc_runs }

		#generate the list of betas to actually load
		loc_betas2load = { phase: [loc_beta_files[phase][beta] for beta in loc_beta_labels[phase].index] for phase in rsa.loc_runs}

		#load in the beta images
		self.loc_betas = { phase: nib.concat_images(loc_betas2load[phase]) for phase in rsa.loc_runs }

		#concat the 2 actually images and the labels
		self.loc_betas = nib.concat_images([self.loc_betas[rsa.loc_runs[0]],self.loc_betas[rsa.loc_runs[1]]],axis=3)
		#print(self.loc_betas.header)

		#create the labels scruture so that its an array
		self.loc_labels = np.concatenate( (loc_beta_labels[rsa.loc_runs[0]].values, loc_beta_labels[rsa.loc_runs[1]].values ), axis=0 )

		#take the 's' off of the labels and switch animal/tool to CS+/-
		for i, label in enumerate(self.loc_labels):
			if label[-1] == 's':
				self.loc_labels[i] = label[:-1]

		self.loc_labels[self.loc_labels == self.rsa.csplus] = 'CS+'
		self.loc_labels[self.loc_labels == self.rsa.csminus] = 'CS-'


	def load_test_betas(self):

		#point to the beta files
		test_beta_files = { phase: glob('%s%sls-s_betas/beta_***.nii.gz'%(self.rsa.bold_dir,phase2rundir[phase])) for phase in self.test_runs }

		#load them in and concatenate within runs
		self.test_betas = { phase: nib.concat_images(test_beta_files[phase]) for phase in test_beta_files }

		#load in the labels for each phase, making them arrays in the process
		self.test_beta_labels = { phase: np.array(glm_timing(self.rsa.subj,phase).phase_events(con=True)) for phase in self.test_betas }
		self.test_beta_labels = self.count_stims(labels=self.test_beta_labels)

	def mask_runs(self):

		#do some masking
		self.mask = '%s/mask/LOC_VTC_mask.nii.gz'%(self.rsa.subj_dir)

		#for phase in self.test_betas:
		#	print(phase)
		#	self.test_betas[phase] = apply_mask(self.test_betas[phase], mask)

		self.test_betas = { phase: apply_mask(self.test_betas[phase], self.mask) for phase in self.test_betas }

		self.loc_betas = apply_mask(self.loc_betas, self.mask)

	#so in the end don't actually need this, because I went back and fixed my day 2 motion correction
	# def hack_day2_affine(self):

	# 	self.refvol = nib.load('%srefvol.nii.gz'%(self.rsa.bold_dir))

	# 	correct_these = [phase for phase in phase2rundir if 'day2' in phase2rundir[phase] and phase in self.test_betas.keys()]
		
	# 	for phase in correct_these:

	# 			self.test_betas[phase].affine[:] = self.refvol.affine

	# 	self.loc_betas.affine[:] = self.refvol.affine

	#paragraph that adds appropriate CS_000 number afterwards
	def count_stims(self,labels=None):

		for phase in labels:
			
			csp = 1
			csm = 1
			
			for i, label in enumerate(labels[phase]):

				if label == 'CS+':

					labels[phase][i] = 'CS+_{0:0=2d}'.format(csp)
					csp += 1

				elif label == 'CS-':

					labels[phase][i] = 'CS-_{0:0=2d}'.format(csm)
					csm += 1

		return labels



'''
In [181]: print(b1.header['dim'])
[ 3 76 76 48  1  1  1  1]

In [182]: print(run8.header['dim'])
[  4  76  76  48 240   1   1   1]

In [183]: print(y.header['dim'])
[ 4 76 76 48 64  1  1  1]
'''
		#self.loc_betas = { phase2: nib.load(loc_beta_files[phase2][beta]) if beta.isin(loc_betas2load[phase2]) for phase2 in rsa.loc_runs }
		#self.loc_betas = {phase: for phase in loc_runs}
		#for phase in rsa.loc_runs:
			
		#	self.loc_betas[phase]
	#load in the betas

	#do rsa

	#need to fix the glm_timing script to get betas for the other phases...
import os
import numpy as np
import pandas as pd
import nibabel as nib
from glob import glob

from fc_config import data_dir, sub_args, fsub, init_dirs, phase2rundir, dc2nix_run_key


class preproc(object):

	#subj takes either 'all' or an array of subject numbers e.g. [1,2,3]
	def __init__(self, subject):

		#determine what the subejcts are
		self.subj = meta(subject)		
		print('Preprocessing '+self.subj.fsub)


	def convert_dicom(self):

		#collect all the phases in the raw folder
		#runs_to_convert = { i: file for i, file in enumerate(os.listdir(raw)) if '.' not in file and file in run_key }

		for phase in dc2nix_run_key:
		#convert and zip all phases using dcm2niix in bash
			convert_cmd = 'dcm2niix -o %s -z y -m y %s'%(self.subj.bold_dir, os.path.join(self.subj.raw, phase))
			os.system(convert_cmd)

		#clean out the .json files
		dust = { file for file in os.listdir(self.subj.bold_dir) if '.json' in file}
		#and delete them
		{ os.remove(self.subj.bold_dir + trash_file) for trash_file in dust }

		#rename files from PHASE to run000
		{ os.rename(self.subj.bold_dir + file , self.subj.bold_dir + dc2nix_run_key[file[:-7]] + '.nii.gz') for file in os.listdir(self.subj.bold_dir) if file[:-7] in dc2nix_run_key}

		#move the t1mprage to the anatomy folder
		os.rename(self.subj.bold_dir + 'struct.nii.gz', self.subj.anatomy + '/struct.nii.gz')

		#move the runs to their specific folder within the day folders
		run_loc = glob('%s/day*/run00*/'%(self.subj.bold_dir))
		mov_runs = glob('%s/run00*.nii.gz'%(self.subj.bold_dir))

		{ os.rename(run, run_loc[i] + run[-13:]) for i, run in enumerate(mov_runs) }
		#{ os.rename(bold + run, run_loc[i] + run) for i,run in enumerate(os.listfile(bold)) if 'run' in run}
	

	def recon_all(self):

		print('Running freesurfer recon-all...')

		recon_all_cmd = 'recon-all -s %s/%sfs -i %s/struct.nii.gz -all'%(self.subj.fsub, self.subj.fsub, self.subj.anatomy)

		os.system(recon_all_cmd)


	def skull_strip(self):

		print('strip da skull')
		print('Creating whole brain mask from Freesurfer brainmask.mgz')

		skull_strip_cmd = 'mri_convert %s/mri/brainmask.mgz -rl %s/struct.nii.gz %s/struct_brain.nii.gz'%(self.subj.fs_dir, self.subj.anatomy, self.subj.anatomy)

		os.system(skull_strip_cmd)
		

	def fsl_mcflirt(self):

		self.check_refvol()

		print('starting fsl motion correction...')

		vols_to_mc = glob('%s/day*/run***/run***.nii.gz'%(self.subj.bold_dir))

		for vol in vols_to_mc:
			
			out_vol = vol[:-13]+'mc_'+vol[-13:]

			mcflirt_cmd = 'mcflirt -in %s -out %s -reffile %s'%(vol, out_vol, self.subj.refvol)
			
			print(vol[-13:])
			
			os.system(mcflirt_cmd)


	def check_refvol(self):

		if not os.path.exists(self.subj.refvol):
			
			print('Creating reference volume via fsl...')

			#os.system('fslmaths ' + subj.day1 + 'run001/run001.nii.gz -Tmean ' + bold + 'refvol')
			os.system('fslmaths %s%srun001.nii.gz -Tmean %s refvol'%(self.subj.bold_dir, phase2rundir['baseline'], self.subj.bold_dir) )


		elif os.path.exists(self.subj.refvol):

			print('refvol already exists, moving on')




class meta(object):

	def __init__(self, sub):

		self.subj_dir, self.bold_dir = init_dirs(sub)

		self.fsub = fsub(sub)

		self.fs_dir = os.path.join(self.subj_dir, self.fsub+'fs')

		self.anatomy = os.path.join(self.subj_dir, 'anatomy')

		self.day1 = os.path.join(self.bold_dir,'day1')
		self.day2 = os.path.join(self.bold_dir,'day2')

		self.refvol = os.path.join(self.bold_dir,'refvol.nii.gz')

		self.raw = os.path.join(self.subj_dir,'raw')


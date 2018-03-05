import os
import numpy as np
import pandas as pd
import nibabel as nib

from glob import glob
from scipy.signal import detrend
from scipy.stats import zscore
from nilearn.masking import apply_mask
from nilearn import signal

from fc_config import data_dir, sub_args, fsub, init_dirs, phase2rundir, dc2nix_run_key, mvpa_prepped, nifti_paths, PPA_fs_prepped



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

	def fsl_reg(self):

		self.check_refvol()

		#first register the functional to the anatomical
		os.system('flirt -in %s -ref %s -dof 6 -omat %sfunc2struct.mat'%(self.subj.refvol, self.subj.struct_brain, self.subj.reg_dir))

		#then register the structural to the standard
		os.system('flirt -in %s -ref $FSL_DIR/data/standard/MNI152_T1_1mm_brain.nii.gz -omat %sstruct2std.mat'%(self.subj.struct_brain, self.subj.reg_dir))

		#then run fnirt (this takes a while)
		os.system('fnirt --in=%s --ref=$FSL_DIR/data/standard/MNI152_T1_1mm_brain.nii.gz --aff=%sstruct2std.mat --cout=%sstruct_warp'%(self.subj.struct_brain, self.subj.reg_dir, self.subj.reg_dir))


	def check_refvol(self):

		if not os.path.exists(self.subj.refvol):
			
			print('Creating reference volume via fsl...')

			#os.system('fslmaths ' + subj.day1 + 'run001/run001.nii.gz -Tmean ' + bold + 'refvol')
			os.system('fslmaths %s%srun001.nii.gz -Tmean %s refvol'%(self.subj.bold_dir, phase2rundir['baseline'], self.subj.bold_dir) )


		elif os.path.exists(self.subj.refvol):

			print('refvol already exists, moving on')


	def create_mask(self):

		#this code needs to be heavily re-written
		pass


	def mvpa_prep(self, mask=None, save_dict=None):

		#handle default args
		if mask is None:
			mask = self.subj.maskvol

		if mask is 'PPA_fs':
			mask = self.subj.PPA_fs_mask

		if save_dict is None:
			save_dict = mvpa_prepped

		#this is a new thing im trying, so it needs to be compared to my other method of detrending and z-scoring
		#this uses signal theory to standardize the data, which uses Sum Squares instead of z-score
		#ensure finite removes any NaN if they're there

		#follow up 2/5: while the individual data points between this method and my manual method vary on the order of .00000####,
		#after running many MVPA decoding schemas the two data sets produced IDENTICAL classification results on numereous different conditions,
		#so I'm going with these because I trust it more!

		print('Prepping runs for MVPA with nilearn signal.clean()')


		#load the runs
		mc_runs = { phase: nib.load('%s%s'%(self.subj.bold_dir, nifti_paths[phase])) for phase in nifti_paths }

		print('Applying Mask')
		#have to mask it _first_
		mc_runs = { phase: apply_mask(mc_runs[phase], mask) for phase in mc_runs }

		print('Cleaning signal')
		mc_runs = { phase: signal.clean(mc_runs[phase], detrend=True, standardize=True, t_r=2, ensure_finite=True) for phase in mc_runs }
		
		print('Saving')
		{ np.savez_compressed( '%s%s'%(self.subj.bold_dir, save_dict[phase]),  mc_runs[phase] ) for phase in mc_runs }



class meta(object):

	def __init__(self, sub):

		self.num = sub

		self.subj_dir, self.bold_dir = init_dirs(sub)

		self.fsub = fsub(sub)

		self.fs_dir = os.path.join(self.subj_dir, self.fsub+'fs')

		self.anatomy = os.path.join(self.subj_dir, 'anatomy')

		self.struct_brain = os.path.join(self.anatomy, 'struct_brain.nii.gz')

		self.day1 = os.path.join(self.bold_dir,'day1')
		self.day2 = os.path.join(self.bold_dir,'day2')



		self.raw = os.path.join(self.subj_dir,'raw')

		self.mask = os.path.join(self.subj_dir, 'mask')


		self.refvol = os.path.join(self.bold_dir,'refvol.nii.gz')
		self.maskvol = os.path.join(self.mask, 'LOC_VTC_mask.nii.gz')

		self.PPA_fs_mask = os.path.join(self.mask, 'PPA_fs_mask.nii.gz')
		
		self.meta = pd.read_csv(os.path.join(self.subj_dir,'behavior','%s_elog.csv'%(self.fsub)))

		self.cs_lookup()

		self.glm_mask = os.path.join(self.mask, 'glm_masking')
		if not os.path.exists(self.glm_mask):
			os.mkdir(self.glm_mask)

		self.PPAmaskvol = os.path.join(self.glm_mask, 'PPA_mask.nii.gz')

		self.reg_dir = os.path.join(self.subj_dir,'fsl_reg')
		if not os.path.exists(self.reg_dir):
			os.mkdir(self.reg_dir)


	#see what the csplus and csminus are for this sub
	def cs_lookup(self):	
		if self.meta['DataFile.Basename'][0][0] == 'A':
			self.csplus = 'animal'
			self.csminus = 'tool'
		elif self.meta['DataFile.Basename'][0][0] == 'T':
			self.csplus = 'tool'
			self.csminus = 'animal'



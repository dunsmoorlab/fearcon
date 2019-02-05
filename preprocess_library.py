import os
import numpy as np
import pandas as pd
import nibabel as nib
from sys import platform
import copy
from subprocess import Popen

from glob import glob
from scipy.signal import detrend
from scipy.stats import zscore
from nilearn.masking import apply_mask
from nilearn import signal
from shutil import copyfile
import sys
from fc_config import *
from nilearn.input_data import NiftiMasker

'''
3/8
things to do
1. slice time correction with:
In [12]: with open('BASELINE.json') as json_data:
    ...:     _j = json.load(json_data)
    ...:     print(_j)
2. rework motion
3. highpass filter
4. actually check everything
5. die
'''

class preproc(object):

	#subj takes either 'all' or an array of subject numbers e.g. [1,2,3]
	def __init__(self, subject):

		#determine what the subejcts are
		self.subj = meta(subject)		
		print('Preprocessing '+ self.subj.fsub)


	def convert_dicom(self):

		for phase in dc2nix_run_key:
		# #convert and zip all phases using dcm2niix in bash
			convert_cmd = 'dcm2niix -o %s -z y -m y %s'%(self.subj.bold_dir, os.path.join(self.subj.raw, phase))
			os.system(convert_cmd)

		#clean out the .json files
		dust = { file for file in os.listdir(self.subj.bold_dir) if '.json' in file}
		#and delete them
		{ os.remove(self.subj.bold_dir + trash_file) for trash_file in dust }

		# rename files from PHASE to run000
		{ os.rename(self.subj.bold_dir + file , self.subj.bold_dir + dc2nix_run_key[file[:-7]] + '.nii.gz') for file in os.listdir(self.subj.bold_dir) if file[:-7] in dc2nix_run_key}
		#this line is only needed if OSIRX fucks up the names again
		{ os.rename(self.subj.bold_dir + file , self.subj.bold_dir + dc2nix_run_key_1[file.split('--')[1]] + '.nii.gz') for file in os.listdir(self.subj.bold_dir) if len(file) > 10}

		#move the t1mprage to the anatomy folder
		os.rename(self.subj.bold_dir + 'struct.nii.gz', self.subj.anatomy + '/struct.nii.gz')

		#move the runs to their specific folder within the day folders
		run_loc = glob('%s/day*/run00*/'%(self.subj.bold_dir))
		mov_runs = glob('%s/run00*.nii.gz'%(self.subj.bold_dir))

		#sort them to work out the wierd placement bug
		run_loc.sort()
		mov_runs.sort()
		
		{ os.rename(run, run_loc[i] + run[-13:]) for i, run in enumerate(mov_runs) }	

	def recon_all(self):

		print('Running freesurfer recon-all...')

		recon_all_cmd = 'recon-all -s %s/%sfs -i %s/struct.nii.gz -all'%(self.subj.fsub, self.subj.fsub, self.subj.anatomy)

		os.system(recon_all_cmd)


	def convert_freesurfer(self):
		src = self.subj.mri
		dest = self.subj.anatomy

		src_names = ['orig', 'brainmask', 'aparc+aseg']
		dest_names = ['orig', 'orig_brain_auto', 'aparc+aseg']

		for i in range(len(src_names)):
			src_file = os.path.join(src, src_names[i] + '.mgz')
			dest_file = os.path.join(dest, dest_names[i] + '.nii.gz')

			#maybe write in a check that exits if the FS file is not there?
			os.system('mri_convert %s %s'%(src_file, dest_file))
		
		for i in range(len(dest_names)):
			dest_file = os.path.join(dest, dest_names[i] + '.nii.gz')
			os.system('fslreorient2std %s %s'%(dest_file, dest_file))
		
		# mask for original brain extraction
		brain_auto = os.path.join(dest, 'orig_brain_auto.nii.gz')
		mask_auto = os.path.join(dest, 'brainmask_auto.nii.gz')
		os.system('fslmaths %s -thr 0.5 -bin %s'%(brain_auto, mask_auto))

		# smooth and threshold the identified tissues; fill any remaining holes
		parcel = os.path.join(dest, 'aparc+aseg.nii.gz')
		mask_surf = os.path.join(dest, 'brainmask_surf.nii.gz')
		os.system('fslmaths %s -thr 0.5 -bin -s 0.25 -bin -fillh26 %s' % (parcel, mask_surf))


		# take intersection with original mask (assumed to include all cortex,
		# so don't want to extend beyond that)
		#this is the best brain mask
		mask =  os.path.join(dest, 'brainmask.nii.gz')
		os.system('fslmaths %s -mul %s -bin %s' % (mask_surf, mask_auto, mask))

		# create a brain-extracted image based on the orig image from
		# freesurfer (later images have various normalization things done that
		# won't match the MNI template as well)
		orig = os.path.join(dest, 'orig.nii.gz')
		output = os.path.join(dest, 'orig_brain.nii.gz')
		os.system('fslmaths %s -mas %s %s' % (orig, mask, output))

		cortex = os.path.join(dest, 'ctx.nii.gz')
		# cortex
		os.system('fslmaths %s -thr 1000 -bin %s' % (parcel, cortex))

		l_wm = os.path.join(dest, 'l_wm.nii.gz')
		r_wm = os.path.join(dest, 'r_wm.nii.gz')
		wm = os.path.join(dest, 'wm.nii.gz')

		os.system('fslmaths %s -thr 2 -uthr 2 -bin %s' % (parcel, l_wm))
		os.system('fslmaths %s -thr 41 -uthr 41 -bin %s' % (parcel, r_wm))
		os.system('fslmaths %s -add %s -bin %s' % (l_wm, r_wm, wm))


		#move things that we wont use again to a new folder:
		things_to_move = ['brainmask_auto.nii.gz', 'brainmask_surf.nii.gz', 'l_wm.nii.gz', 'orig_brain_auto.nii.gz', 'r_wm.nii.gz']
		move2dir = os.path.join(dest, 'freesurfer_non_registered')
		if not os.path.exists(move2dir):
			os.mkdir(move2dir)
		for thing in things_to_move:
			os.rename(os.path.join(dest, thing), os.path.join(move2dir, thing))

	#this is new (August 2018) code to fix the orig_brain not being 00 outside of the brain
	def fix_orig_brain(self):

		orig = os.path.join(self.subj.anatomy,'orig.nii.gz')
		orig_brain = os.path.join(self.subj.anatomy,'orig_brain.nii.gz')
		old_orig = os.path.join(self.subj.anatomy,'old_orig_brain.nii.gz')
		brainmask = os.path.join(self.subj.anatomy,'brainmask.nii.gz')
		
		os.system('cp %s %s'%(orig_brain, old_orig))
		os.system('fslmaths %s -mas %s %s'%(orig, brainmask, orig_brain))
	
	#Register FreeSurfer output to original anatomical spaces
	def register_freesurfer(self):

		dest = self.subj.anatomy

		# register orig to highres
		os.system('antsRegistration -d 3 -r [{ref},{src},1] -t Rigid[0.1] -m MI[{ref},{src},1,32,Regular,0.25] -c [1000x500x250x100,1e-6,10] -f 8x4x2x1 -s 3x2x1x0vox -n BSpline -w [0.005,0.995] -o {xfm}'.format(
			ref=os.path.join(dest, 'struct.nii.gz'), src=os.path.join(dest, 'orig.nii.gz'), 
			xfm=os.path.join(dest, 'orig-struct')))
		o2h = os.path.join(dest, 'orig-struct0GenericAffine.mat')

		images = ['orig', 'orig_brain', 'brainmask', 'aparc+aseg', 'ctx', 'wm']
		labels = [False, False, True, True, True, True]

		struct = os.path.join(dest, 'struct.nii.gz')
		for image, label in zip(images, labels):
			if label:
				interp = 'NearestNeighbor'
			else:
				interp = 'BSpline'

			input_img = os.path.join(dest, image + '.nii.gz')
			output_img = os.path.join(dest, image + '.nii.gz')
			#transform to the space of the struct image
			os.system('antsApplyTransforms -i {} -o {} -r {} -t {} -n {}'.format(input_img, output_img, struct, o2h, interp))

	def register_anat2mni(self):
		
		dest = self.subj.anatomy

		template = '/usr/local/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz'

		reg_data = os.path.join(dest, 'antsreg', 'data')
		reg_xfm = os.path.join(dest, 'antsreg', 'transforms')
		if not os.path.exists(reg_data):
			os.system('mkdir -p %s'%(reg_data))
		if not os.path.exists(reg_xfm):
			os.system('mkdir -p %s'%(reg_xfm))
		
		xfm_base = os.path.join(reg_xfm, 'orig-template_')
		highres = os.path.join(dest, 'orig_brain.nii.gz')

		os.system('ANTS 3 -m CC[{template},{highres},1,5] -t SyN[0.25] -r Gauss[3,0] -o {xfm} -i 30x90x20 --use-Histogram-Matching  --number-of-affine-iterations 10000x10000x10000x10000x10000 --MI-option 32x16000'.format(
			template=template, highres=highres, xfm=xfm_base))



	#Register the brain mask to functional space refvol,
	def epi_reg(self):
		
		highres = os.path.join(self.subj.anatomy, 'orig.nii.gz')
		highres_brain = os.path.join(self.subj.anatomy, 'orig_brain.nii.gz')
		highres_mask = os.path.join(self.subj.anatomy, 'brainmask.nii.gz')
		wm_mask = os.path.join(self.subj.anatomy, 'wm.nii.gz')

		for phase in phase2rundir:
			epi_dir = os.path.join(self.subj.bold_dir, phase2rundir[phase])
			epi_input = os.path.join(self.subj.bold_dir, avg_mc_paths[phase])
			epi_output = os.path.join(epi_dir, 'be_avg_mc_%s.nii.gz'%(py_run_key[phase]))

			fm_dir = os.path.join(epi_dir, 'fm')
			if not os.path.exists(fm_dir):
				os.mkdir(fm_dir)

			copyfile(wm_mask, os.path.join(fm_dir, 'epireg_fast_wmseg.nii.gz'))

			out_base = os.path.join(fm_dir, 'epireg')

			epireg_cmd = 'epi_reg --wmseg=%s --echospacing=%.06f --pedir=y- -v --epi=%s --t1=%s --t1brain=%s --out=%s --noclean' % (
				wm_mask, 0.000589987, epi_input, highres, highres_brain, out_base)

			os.system(epireg_cmd)

			#take the inverse of the functional 2 anatomical registration
			os.system('convert_xfm %s -inverse -omat %s'%(os.path.join(fm_dir, 'epireg.mat'), os.path.join(fm_dir, 'epireg_inv.mat')))
			

			# transform the anatomical brain mask into functional space
			# using the anatomical 2 functioncal _inv.mat
			mask_reg = os.path.join(fm_dir, 'brainmask')
			os.system('flirt -in %s -ref %s -applyxfm -init %s -out %s -interp nearestneighbour' % (
				highres_mask, epi_input, os.path.join(fm_dir, 'epireg_inv.mat'),
				mask_reg))

			# dilate to make a tighter brain extraction than the liberal one
			# originally used for the functionals
			os.system('fslmaths %s -kernel sphere 3 -dilD %s' % (mask_reg, mask_reg))

			# mask the input
			os.system('fslmaths %s -mas %s %s' % (epi_input, mask_reg, epi_output))


			#clean out some stuff
			clean_list = ['epireg_fast_wmseg.nii.gz','epireg_fast_wmedge.nii.gz','brainmask.nii.gz','epireg.nii.gz']
			for thing in clean_list:
				os.remove(os.path.join(fm_dir, thing))		



	def create_refvol(self):

		for day in day_dict:

			reg_dir = os.path.join(self.subj.bold_dir, day, 'recursive_registration')
			if not os.path.exists(reg_dir):
				os.mkdir(reg_dir)

			for level in day_dict[day]:
				print(day)

				for i, pair in enumerate(day_dict[day][level]):
					print(pair)

					A = pair[0]
					B = pair[1]
					name = pair[2]

					pair_dir = os.path.join(reg_dir, name)
					if not os.path.exists(pair_dir):
						os.mkdir(pair_dir)

					prefix = pair_dir + os.sep + name
					
					if level == 'level1':
						A_epi = os.path.join(self.subj.bold_dir, be_avg_mc_paths[A])
						B_epi = os.path.join(self.subj.bold_dir, be_avg_mc_paths[B])

					else:
						A_epi = os.path.join(reg_dir, A, A + '_mid.nii.gz')
						B_epi = os.path.join(reg_dir, B, B + '_mid.nii.gz')
					
					reg_cmd = 'unbiased_pairwise_registration.sh -d 3 -f %s -m %s -o %s'%(
						A_epi, B_epi, prefix)
					os.system(reg_cmd)
					#get rid of the mid points to save space
					# if level != 'level1':
						# os.remove(A_epi)
						# os.remove(B_epi)


		refvol_dir = os.path.join(self.subj.bold_dir, 'refvol')
		if not os.path.exists(refvol_dir):
			os.mkdir(refvol_dir)
		prefix = refvol_dir + os.sep + 'refvol'
		day1 = os.path.join(self.subj.bold_dir, 'day1', 'recursive_registration', 'day1', 'day1_mid.nii.gz')
		day2 = os.path.join(self.subj.bold_dir, 'day2', 'recursive_registration', 'day2', 'day2_mid.nii.gz')
		os.system('unbiased_pairwise_registration.sh -d 3 -f %s -m %s -o %s'%(
						day1, day2, prefix))
		os.rename(os.path.join(refvol_dir, 'refvol_mid.nii.gz'), os.path.join(refvol_dir, 'refvol.nii.gz'))

		#register the anatomical to the refvol
		highres = os.path.join(self.subj.anatomy, 'orig.nii.gz')
		highres_brain = os.path.join(self.subj.anatomy, 'orig_brain.nii.gz')
		highres_mask = os.path.join(self.subj.anatomy, 'brainmask.nii.gz')
		wm_mask = os.path.join(self.subj.anatomy, 'wm.nii.gz')

		epi_dir = self.subj.refvol_dir
		epi_input = self.subj.refvol
		epi_output = self.subj.refvol_be

		fm_dir = os.path.join(epi_dir, 'fm')
		if not os.path.exists(fm_dir):
			os.mkdir(fm_dir)

		copyfile(wm_mask, os.path.join(fm_dir, 'epireg_fast_wmseg.nii.gz'))

		out_base = os.path.join(fm_dir, 'epireg')

		epireg_cmd = 'epi_reg --wmseg=%s --echospacing=%.06f --pedir=y- -v --epi=%s --t1=%s --t1brain=%s --out=%s --noclean' % (
			wm_mask, 0.000589987, epi_input, highres, highres_brain, out_base)

		os.system(epireg_cmd)

		#take the inverse of the functional 2 anatomical registration
		os.system('convert_xfm %s -inverse -omat %s'%(os.path.join(fm_dir, 'epireg.mat'), os.path.join(fm_dir, 'epireg_inv.mat')))
		

		# transform the anatomical brain mask into functional space
		# using the anatomical 2 functioncal _inv.mat
		mask_reg = os.path.join(fm_dir, 'brainmask')
		os.system('flirt -in %s -ref %s -applyxfm -init %s -out %s -interp nearestneighbour' % (
			highres_mask, epi_input, os.path.join(fm_dir, 'epireg_inv.mat'),
			mask_reg))

		# dilate to make a tighter brain extraction than the liberal one
		# originally used for the functionals
		os.system('fslmaths %s -kernel sphere 3 -dilD %s' % (mask_reg, mask_reg))

		# mask the input
		os.system('fslmaths %s -mas %s %s' % (epi_input, mask_reg, epi_output))


		#clean out some stuff
		clean_list = ['epireg_fast_wmseg.nii.gz','epireg_fast_wmedge.nii.gz']
		for thing in clean_list:
			os.remove(os.path.join(fm_dir, thing))
	
	#WIP
	#take functional input and transform to anatomical space
	def epi2anat(self):
			
			epi_anat = os.path.join(epi_dir, 'avg_mc_' + py_run_key[phase] + '_brain_anant' )

			# os.system('flirt -in %s -ref %s -applyxfm -init %s -out %s -interp nearestneighbour'%(
			# 	epi_output, highres_brain, os.path.join(fm_dir, 'epireg.mat'), epi_anat)) 


	#actually apply motion correction to each run using the reference volume
	def prep_final_epi_part1(self):

		for phase in phase2rundir.keys():

			#make the output directory
			reg_xfm = os.path.join(self.subj.bold_dir, phase2rundir[phase], 'antsreg', 'transforms')
			reg_data = os.path.join(self.subj.bold_dir, phase2rundir[phase], 'antsreg', 'data')			
			os.system('mkdir -p %s' % reg_data)
			os.system('mkdir -p %s' % reg_xfm)

			#point to the source directory and the reference volume directory
			srcdir = os.path.join(self.subj.bold_dir, phase2rundir[phase])
			refdir = self.subj.refvol_dir

			#load the brain extracted avg motion corrected volume
			srcvol = os.path.join(srcdir, 'be_avg_mc_' + py_run_key[phase] + '.nii.gz')
			#and the reference volume
			refvol = self.subj.refvol_be
			#point to the actual raw bold
			bold = os.path.join(self.subj.bold_dir, raw_paths[phase])
			#convert the motion parameters to a single thing that can be read in by FSL
			os.system('cat %smc_%s.nii.gz.mat/MAT* > %smc_%s.cat'%(srcdir, py_run_key[phase], srcdir, py_run_key[phase]))
			mcf_file = os.path.join(srcdir, 'mc_%s.cat'%(py_run_key[phase]))

			#load the brainmask for the refvol
			mask = os.path.join(refdir, 'fm', 'brainmask.nii.gz')


			# output files
			bold_reg = os.path.join(srcdir, 'reg_%s.nii.gz'%(py_run_key[phase]))
			bold_init = os.path.join(srcdir, 'bold_reg_init.nii.gz')

			# nonlinear registration to the reference run
			xfm_base = os.path.join(reg_xfm, '%s-refvol_' % (py_run_key[phase]))
			itk_file = xfm_base + '0GenericAffine.mat'
			txt_file = xfm_base + '0GenericAffine.txt'
			os.system('antsRegistrationSyN.sh -d 3 -m {mov} -f {fix} -o {out} -n 4 -t s'.format(
				mov=srcvol, fix=refvol, out=xfm_base))

			# convert the affine part to FSL format
			os.system('ConvertTransformFile 3 %s %s' % (itk_file, txt_file))

		#normally would continue here but we have to go to TACC to actually convert what we need to convert...

	def prep_final_epi_part2(self, get_files=False):

		#do this on TACC
		#os.system('c3d_affine_tool -itk %s -ref %s -src %s -ras2fsl -o %s' % (txt_file, refvol, srcvol, reg_file))

		for phase in phase2rundir.keys():

			print(phase)
			srcdir = os.path.join(self.subj.bold_dir, phase2rundir[phase])
			bold = os.path.join(self.subj.bold_dir, raw_paths[phase])
			refvol = self.subj.refvol_be
			bold_init = os.path.join(srcdir, 'bold_reg_init')
			mcf_file = os.path.join(srcdir, 'mc_%s.cat'%(py_run_key[phase]))
			reg_xfm = os.path.join(self.subj.bold_dir, phase2rundir[phase], 'antsreg', 'transforms')
			reg_file = os.path.join(reg_xfm, '%s-refvol.mat' % (py_run_key[phase]))
			bold_reg = os.path.join(srcdir, 'reg_%s.nii.gz'%(py_run_key[phase]))
			output = os.path.join(srcdir, 'pp_%s.nii.gz'%(py_run_key[phase]))
			refdir = self.subj.refvol_dir
			mask = os.path.join(refdir, 'fm', 'brainmask.nii.gz')


			if get_files:
				copyfile(os.path.join(data_dir,'reg_convert',self.subj.fsub, py_run_key[phase] + '-refvol.mat'), reg_file)
				os.system('cat %s/MAT_* > %s'%(os.path.join(srcdir, 'mc_%s.nii.gz.mat'%(py_run_key[phase])), mcf_file))

			# apply motion correction, unwarping, and affine co-registration
			cmd1='applywarp --in=%s --out=%s --ref=%s --premat=%s --postmat=%s --interp=spline --paddingsize=1'%(bold, bold_init, refvol, mcf_file, reg_file)
			os.system(cmd1)

			xfm_base = os.path.join(reg_xfm, '%s-refvol_' % (py_run_key[phase]))
			warp = xfm_base + '1Warp.nii.gz'
			os.system('antsApplyTransforms -d 3 -e 3 -i {}.nii.gz -o {} -r {} -t {} -n BSpline'.format(
				bold_init, bold_reg, refvol, warp))

			os.system('fslmaths %s -mas %s %s' % (
				bold_reg, mask, output))

			os.remove(os.path.join(srcdir, 'bold_reg_init.nii.gz'))
			if os.path.exists(os.path.join(srcdir, 'mc_%s.nii.gz'%(py_run_key[phase]))):
				os.remove(os.path.join(srcdir, 'mc_%s.nii.gz'%(py_run_key[phase])))
			os.remove(bold_reg)


	def mc_first_pass(self):

		print('starting fsl motion correction...')

		#look into using N4biasFieldCorrection
		for phase in phase2rundir.keys():

			in_vol = os.path.join(self.subj.bold_dir, raw_paths[phase])
			#dont actually output a volume here, just get the parameters
			out_vol = os.path.join(self.subj.bold_dir, temp_mc_paths[phase])

			mcflirt_cmd = 'mcflirt -in %s -out %s -mats -plots '%(in_vol, out_vol)
			
			print(in_vol[-13:])
			
			os.system(mcflirt_cmd)

			avg_vol = os.path.join(self.subj.bold_dir, avg_mc_paths[phase][:-7])

			avg_cmd = 'fslmaths %s -Tmean %s'%(out_vol, avg_vol)

			os.system(avg_cmd)

			os.remove(out_vol)
	
	def fsl_slicer(self):


		pass


	def fsl_reg(self):

		#first register the functional to the anatomical
		func2struct = os.path.join(self.subj.reg_dir, 'func2struct.mat')
		os.system('flirt -in %s -ref %s -dof 6 -omat %s'%(self.subj.refvol_be, self.subj.orig_brain, func2struct))

		struct2std = os.path.join(self.subj.reg_dir, 'struct2std.mat')
		#then register the structural to the standard
		os.system('flirt -in %s -ref $FSL_DIR/data/standard/MNI152_T1_1mm_brain.nii.gz -dof 6 -omat %s'%(self.subj.orig_brain, struct2std))

		struct_warp = os.path.join(self.subj.reg_dir, 'struct2std_warp')
		#then run fnirt (this takes a while)
		os.system('fnirt --in=%s --ref=$FSL_DIR/data/standard/MNI152_T1_1mm_brain.nii.gz --aff=%s --cout=%s'%(self.subj.orig_brain, struct2std, struct_warp))
		
		
		struct_warp_img = os.path.join(self.subj.reg_dir, 'struct2std_warp.nii.gz')
		# os.system('applywarp --in=%s --ref=$FSL_DIR/data/standard/MNI152_T1_1mm_brain.nii.gz --out=%s --premat=%s --warp=%s'%(
			# self.subj.refvol_be, os.path.join(self.subj.reg_dir, 'test'), func2struct, struct_warp_img))

	def check_refvol(self):

		if not os.path.exists(self.subj.refvol):
			
			print('Creating reference volume via fsl...')

			#os.system('fslmaths ' + subj.day1 + 'run001/run001.nii.gz -Tmean ' + bold + 'refvol')
			os.system('fslmaths %s%srun001.nii.gz -Tmean %srefvol'%(self.subj.bold_dir, phase2rundir['baseline'], self.subj.bold_dir) )


		elif os.path.exists(self.subj.refvol):

			print('refvol already exists, moving on')


	def check_bbreg(self, overwrite=False):

		self.check_refvol()

		if overwrite:

			print('running bbregister')

			reg_cmd = 'bbregister --s %s/%sfs --mov %s --init-fsl --bold --reg %s'%(self.subj.fsub, self.subj.fsub, self.subj.refvol_be, self.subj.fs_regmat)

			os.system(reg_cmd)

		else:
			print('overwrite=False')

	def create_mask(self, overwrite=False):
		#ok so looks like sometimes OSX does some fucky shit with new subprocesses,
		#basically there is a library file that freesurfer needs,
		#but the python environment changes the defualt location where to look for it
		fs_home = os.getenv('FREESURFER_HOME')
		if platform == 'darwin':
			env = copy.deepcopy(os.environ)
			ld_path = os.path.join(fs_home, 'lib', 'gcc', 'lib')
			if 'DYLD_LIBRARY_PATH' not in env:
				env['DYLD_LIBRARY_PATH'] = ld_path
			else:
				env['DYLD_LIBRARY_PATH'] = 'DYLD_LIBRARY_PATH' + ':' + ld_path

		#populate all of the label files from each hemisphere's annotate file
		if overwrite:

			# lh_pop_cmd = ['mri_annotation2label', '--s', '%s/%sfs'%(self.subj.fsub,self.subj.fsub), '--hemi', 'lh', '--outdir', '%s'%(self.subj.fs_label)]
			# rh_pop_cmd = ['mri_annotation2label', '--s', '%s/%sfs'%(self.subj.fsub,self.subj.fsub), '--hemi', 'rh', '--outdir', '%s'%(self.subj.fs_label)]
			
			Popen(lh_pop_cmd,env=env).wait()
			Popen(rh_pop_cmd,env=env).wait()

		self.check_bbreg(overwrite=overwrite)

		if overwrite:

			print('generating masks')
			
			VTC_lh=['mri_label2vol', '--subject', '%s/%sfs'%(self.subj.fsub,self.subj.fsub) , 
					'--label', '%slh.fusiform.label'%(self.subj.fs_label) ,
					'--label', '%slh.parahippocampal.label'%(self.subj.fs_label) ,
					'--label', '%slh.inferiortemporal.label'%(self.subj.fs_label) ,
					'--label', '%slh.lingual.label'%(self.subj.fs_label) ,
					'--temp', '%s'%(self.subj.refvol_be) ,
					'--reg', '%s'%(self.subj.fs_regmat) ,
					'--proj', 'frac', '0', '1', '.1' ,
					'--fillthresh', '.5' ,
					'--hemi', 'lh' ,
					'--o', '%slh_VTC.nii.gz'%(self.subj.mask)]

			VTC_rh=['mri_label2vol', '--subject', '%s/%sfs'%(self.subj.fsub,self.subj.fsub) , 
					'--label', '%srh.fusiform.label'%(self.subj.fs_label) ,
					'--label', '%srh.parahippocampal.label'%(self.subj.fs_label) ,
					'--label', '%srh.inferiortemporal.label'%(self.subj.fs_label) ,
					'--label', '%srh.lingual.label'%(self.subj.fs_label) ,
					'--temp', '%s'%(self.subj.refvol_be) ,
					'--reg', '%s'%(self.subj.fs_regmat) ,
					'--proj', 'frac', '0', '1', '.1' ,
					'--fillthresh', '.5' ,
					'--hemi', 'rh' ,
					'--o', '%srh_VTC.nii.gz'%(self.subj.mask)]

			LOC_lh=['mri_label2vol', '--subject', '%s/%sfs'%(self.subj.fsub,self.subj.fsub) , 
					'--label', '%slh.lateraloccipital.label'%(self.subj.fs_label) ,
					'--temp', '%s'%(self.subj.refvol_be) ,
					'--reg', '%s'%(self.subj.fs_regmat) ,
					'--proj', 'frac', '0', '1', '.1' ,
					'--fillthresh', '.5' ,
					'--hemi', 'lh' ,
					'--o', '%slh_LOC.nii.gz'%(self.subj.mask)]
			
			LOC_rh=['mri_label2vol', '--subject', '%s/%sfs'%(self.subj.fsub,self.subj.fsub) , 
					'--label', '%srh.lateraloccipital.label'%(self.subj.fs_label) ,
					'--temp', '%s'%(self.subj.refvol_be) ,
					'--reg', '%s'%(self.subj.fs_regmat) ,
					'--proj', 'frac', '0', '1', '.1' ,
					'--fillthresh', '.5' ,
					'--hemi', 'rh' ,
					'--o', '%srh_LOC.nii.gz'%(self.subj.mask)]

			combine_VTC = 'fslmaths %slh_VTC.nii.gz -add %srh_VTC.nii.gz -bin %sVTC_mask.nii.gz'%(self.subj.mask,self.subj.mask,self.subj.mask)
			combine_LOC = 'fslmaths %slh_LOC.nii.gz -add %srh_LOC.nii.gz -bin %sLOC_mask.nii.gz'%(self.subj.mask,self.subj.mask,self.subj.mask)
			combine_LOC_VTC = 'fslmaths %sLOC_mask.nii.gz -add %sVTC_mask.nii.gz -bin %sLOC_VTC_mask.nii.gz'%(self.subj.mask,self.subj.mask,self.subj.mask)

			#run all the cmds
			print('lh VTC mask')
			Popen(VTC_lh,env=env).wait()

			print('rh VTC mask')
			Popen(VTC_rh,env=env).wait()

			print('lh LOC mask')
			Popen(LOC_lh,env=env).wait()

			print('rh LOC mask')
			Popen(LOC_rh,env=env).wait()

			print('combining VTC')
			os.system(combine_VTC)

			print('combining LOC')
			os.system(combine_LOC)

			print('combining LOC_VTC')
			os.system(combine_LOC_VTC)



		else:
			print('mask already exists')

		os.system('flirt -in %sLOC_VTC_mask.nii.gz -ref %s -out %s'%(
				self.subj.mask, self.subj.refvol_be, self.subj.maskvol))

	def new_mask(self):

		srcdir = self.subj.anatomy
		outdir = os.path.join(self.subj.anatomy, 'new_mask')
		if not os.path.exists(outdir):
			os.mkdir(outdir)
		aparc_aseg = os.path.join(srcdir,'aparc+aseg.nii.gz')

		lh_fusiform = [os.path.join(outdir,'lh_fusiform.nii.gz'),1007]
		rh_fusiform = [os.path.join(outdir,'rh_fusiform.nii.gz'),2007]
		
		lh_parahippocampal = [os.path.join(outdir,'lh_parahippocampal.nii.gz'),1016]
		rh_parahippocampal = [os.path.join(outdir,'rh_parahippocampal.nii.gz'),2016]
		
		lh_inferiortemporal = [os.path.join(outdir,'lh_inferiortemporal.nii.gz'),1009]
		rh_inferiortemporal = [os.path.join(outdir,'rh_inferiortemporal.nii.gz'),2009]
		
		lh_lingual = [os.path.join(outdir,'lh_lingual.nii.gz'),1013]
		rh_lingual = [os.path.join(outdir,'rh_lingual.nii.gz'),2013]
		
		lh_lateraloccipital = [os.path.join(outdir,'lh_lateraloccipital.nii.gz'),1011]
		rh_lateraloccipital = [os.path.join(outdir,'rh_lateraloccipital.nii.gz'),2011]

		for roi in [lh_fusiform,rh_fusiform,lh_parahippocampal,rh_parahippocampal,
					lh_inferiortemporal,rh_inferiortemporal,lh_lingual,rh_lingual,
					lh_lateraloccipital,rh_lateraloccipital]:

			os.system('fslmaths %s -thr %s -uthr %s %s'%(
				aparc_aseg, roi[1], roi[1], roi[0]))

		out_mask = os.path.join(outdir, 'LOC_VTC_mask.nii.gz')

		os.system('fslmaths %s -add %s -add %s -add %s -add %s -add %s -add %s -add %s -add %s -add %s -bin %s'%(
			lh_fusiform[0], rh_fusiform[0], lh_parahippocampal[0], rh_parahippocampal[0], lh_inferiortemporal[0], rh_inferiortemporal[0],
			lh_lingual[0], rh_lingual[0], lh_lateraloccipital[0], rh_lateraloccipital[0], out_mask))

		struct = os.path.join(srcdir, 'struct.nii.gz')
		o2h = os.path.join(srcdir, 'orig-struct0GenericAffine.mat')
		interp = 'NearestNeighbor'
		
		#register the mask to freesurfer_structural space, which is what was used to register to functional space
		os.system('antsApplyTransforms -i {} -o {} -r {} -t {} -n {}'.format(out_mask, out_mask, struct, o2h, interp))

		anat2func = os.path.join(self.subj.refvol_dir, 'fm', 'epireg_inv.mat')

		#register and resample mask to reference brain extracted functional
		os.system('flirt -in %s -ref %s -applyxfm -init %s -out %s -interp nearestneighbour'%(
			out_mask, self.subj.refvol_be, anat2func, out_mask))

	def LOC_mask4dylan(self):

		srcdir = self.subj.anatomy
		outdir = os.path.join(self.subj.anatomy, 'new_mask')
		if not os.path.exists(outdir):
			os.mkdir(outdir)
		aparc_aseg = os.path.join(srcdir,'aparc+aseg.nii.gz')

		lh_lateraloccipital = [os.path.join(outdir,'lh_lateraloccipital.nii.gz'),1011]
		rh_lateraloccipital = [os.path.join(outdir,'rh_lateraloccipital.nii.gz'),2011]

		out_mask = os.path.join(outdir, 'LOC_mask.nii.gz')

		os.system('fslmaths %s -add %s -bin %s'%(
			lh_lateraloccipital[0], rh_lateraloccipital[0], out_mask))

		struct = os.path.join(srcdir, 'struct.nii.gz')
		o2h = os.path.join(srcdir, 'orig-struct0GenericAffine.mat')
		interp = 'NearestNeighbor'
		
		#register the mask to freesurfer_structural space, which is what was used to register to functional space
		os.system('antsApplyTransforms -i {} -o {} -r {} -t {} -n {}'.format(out_mask, out_mask, struct, o2h, interp))

		anat2func = os.path.join(self.subj.refvol_dir, 'fm', 'epireg_inv.mat')

		#register and resample mask to reference brain extracted functional
		os.system('flirt -in %s -ref %s -applyxfm -init %s -out %s -interp nearestneighbour'%(
			out_mask, self.subj.refvol_be, anat2func, out_mask))

	def mvpa_prep(self,old=True,roi=None,sd=None):

		if old: save_dict = mvpa_masked_prepped
		if not old: save_dict = sd


		#this is a new thing im trying, so it needs to be compared to my other method of detrending and z-scoring
		#this uses signal theory to standardize the data, which uses Sum Squares instead of z-score
		#ensure finite removes any NaN if they're there

		#follow up 2/5: while the individual data points between this method and my manual method vary on the order of .00000####,
		#after running many MVPA decoding schemas the two data sets produced IDENTICAL classification results on numereous different conditions,
		#so I'm going with these because I trust it more!

		print('Prepping runs for MVPA with nilearn signal.clean()')

		if old: mask = self.subj.ctx_maskvol
		if not old: mask = os.path.join(self.subj.roi,'%s_mask.nii.gz'%(roi))


		masker = NiftiMasker(mask_img=mask, detrend=True,
					high_pass=0.0078, t_r=2, standardize=True)

		print('loading apply NiftiMasker')

		pp_runs = {}
		for phase in nifti_paths:
			if phase == 'localizer_2' and self.subj.num == 107:
				pass
			else:
				phase_ = masker.fit_transform(os.path.join(self.subj.bold_dir, nifti_paths[phase]))
				np.savez_compressed( '%s%s'%(self.subj.bold_dir, save_dict[phase]),  phase_ )
		#load the runs
		#### pp_runs = {phase: masker.fit_transform(os.path.join(self.subj.bold_dir, nifti_paths[phase])) for phase in nifti_paths}
		
		# print('Saving')
		#### { np.savez_compressed( '%s%s'%(self.subj.bold_dir, save_dict[phase]),  pp_runs[phase] ) for phase in pp_runs }

		# pp_runs = {}
		# for phase in nifti_paths:
		# 	if phase == 'localizer_2': pass
		# 	else:
		# 		pp_runs[phase] = masker.fit_transform(os.path.join(subj.bold_dir,nifti_paths[phase]))
		# 		np.savez_compressed('%s%s'%(subj.bold_dir, save_dict[phase]), pp_runs[phase])


	def beta_mvpa_prep(self,roi=None,save_dict=beta_masked_prepped):
		if roi is None: mask = self.subj.ctx_maskvol
		else: mask = os.path.join(self.subj.roi,'%s_mask.nii.gz'%(roi))

		masker = NiftiMasker(mask_img=mask, standardize=True)

		print('masking runs')
		
		if self.subj.num == 107:
			beta_imgs = {}
			phase = 'localizer_1'
			beta_imgs[phase] = masker.fit_transform(os.path.join(self.subj.bold_dir, phase2rundir[phase],'fsl_betas','%s_beta.nii.gz'%(py_run_key[phase]))) 
		else:
			beta_imgs = {phase: masker.fit_transform(os.path.join(self.subj.bold_dir,
				phase2rundir[phase],'fsl_betas','%s_beta.nii.gz'%(py_run_key[phase])))
				for phase in py_run_key if 'localizer' in phase}

		print('Saving')
		{ np.savez_compressed( '%s%s'%(self.subj.bold_dir, save_dict[phase]),  beta_imgs[phase] ) for phase in beta_imgs }

class meta(object):

	def __init__(self, sub):

		if sub == 'gusbrain':
			self.fs_id = 'gusbrain'
			self.fsub = 'gusbrain'
		elif sub == 'fs':
			self.fs_id = 'fsaverage'
			self.fsub = 'fsaverage'
		else:
			
			self.num = sub

			self.subj_dir, self.bold_dir = init_dirs(sub)

			self.fsub = fsub(sub)

			self.fs_id = self.fsub + 'fs'

			self.fs_dir = os.path.join(self.subj_dir, self.fs_id)

			if os.path.isdir(self.fs_dir):
				self.fs_label = self.fs_dir + os.sep + 'annot2label/'
				self.fs_reg = self.fs_dir + os.sep + 'reg' + os.sep
			
				if not os.path.exists(self.fs_label):
					os.mkdir(self.fs_label)
				if not os.path.exists(self.fs_reg):
					os.mkdir(self.fs_reg)

				self.fs_regmat = self.fs_reg + os.sep + 'RegMat.dat'

				self.mri = os.path.join(self.fs_dir,'mri')

			self.anatomy = os.path.join(self.subj_dir, 'anatomy')

			self.orig_brain = os.path.join(self.anatomy, 'orig_brain.nii.gz')

			self.day1 = os.path.join(self.bold_dir,'day1')
			self.day2 = os.path.join(self.bold_dir,'day2')



			self.raw = os.path.join(self.subj_dir,'raw')

			self.mask = os.path.join(self.subj_dir, 'mask/')
			self.roi = os.path.join(self.mask,'roi')


			self.refvol_dir = os.path.join(self.bold_dir,'refvol')
			self.refvol_be = os.path.join(self.refvol_dir, 'be_refvol.nii.gz')
			self.refvol = os.path.join(self.refvol_dir, 'refvol.nii.gz')

			self.maskvol = os.path.join(self.mask, 'LOC_VTC_mask.nii.gz')
			self.ctx_maskvol = os.path.join(self.anatomy, 'new_mask', 'LOC_VTC_mask.nii.gz')

			self.brainmask = os.path.join(self.refvol_dir, 'fm', 'brainmask.nii.gz')

			# self.PPA_fs_mask = os.path.join(self.mask, 'PPA_fs_mask.nii.gz')
			meta_file = os.path.join(self.subj_dir,'behavior','%s_elog.csv'%(self.fsub))
			if os.path.exists(meta_file):
				self.meta = pd.read_csv(meta_file)

				self.cs_lookup()

			# self.glm_mask = os.path.join(self.mask, 'glm_masking')
			# if not os.path.exists(self.glm_mask):
				# os.mkdir(self.glm_mask)

			# self.PPAmaskvol = os.path.join(self.glm_mask, 'PPA_mask.nii.gz')

			self.reg_dir = os.path.join(self.subj_dir,'fsl_reg')
			if not os.path.exists(self.reg_dir):
				os.mkdir(self.reg_dir)

			self.model_dir = os.path.join(self.subj_dir, 'model')


	#see what the csplus and csminus are for this sub
	def cs_lookup(self):	
		if self.meta['DataFile.Basename'][0][0] == 'A':
			self.csplus = 'animal'
			self.csminus = 'tool'
		elif self.meta['DataFile.Basename'][0][0] == 'T':
			self.csplus = 'tool'
			self.csminus = 'animal'



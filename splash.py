#Augustin Hennings 2017
#Dunsmoor/Lewis-Peacock Labs

import os
import argparse
import getpass
import numpy as np
import nibabel as nib
import scipy as sp
import sys
import copy
from scipy.signal import detrend
from sys import platform
from shutil import copyfile
from glob import glob
from subprocess import Popen, PIPE

#OK, so as Remy pointed out there is a much easier work around to keeping data synced across devices,
#without having to edit the crap out of freesurfer so that it will be ok with /path with spaces/
#solution: use /GoogleDrive/
#as it is now, im going to just comment out the redundent stuff, but eventually I should go back and delete them


#import the local config file
from fc_config import subj_dir, data_dir
from fc_config import get_sub_args
from fc_config import phase2location, mvpa_prepped, run_key

#build the python arg_parser
parser = argparse.ArgumentParser(description='Function arguments')

#add arguments
parser.add_argument('-s', '--subj', nargs='+', help='Subject Number', default=0, type=str)
parser.add_argument('-d', '--convert', help='Convert Dicom to nifti', default=False, type=bool)
parser.add_argument('-rec', '--recon', help='Run freesurfer recon-all', default=False, type=bool)
parser.add_argument('-mc', '--motion_correct', help='Run motion correction', default=False, type=bool)
parser.add_argument('-sk', '--skull_strip', help='Skull strip with Freesurfer', default=False, type=bool)
parser.add_argument('-m', '--mask', help='Generate masks using freesurfer', default=False, type=bool)
parser.add_argument('-t', '--transform', help='Detrend using nipype & z-score', default=False, type=bool)
parser.add_argument('-sr','--prep_runs', nargs='+', help='Convert 4D arrays to 2D arrays and apply mask, by single runs', default='', type=str)
parser.add_argument('-all','--prep_all_data', help='Prep all data for MVPA', default= False, type=bool)
parser.add_argument('-gzip', '--compress', help='Compress old runs to save space', default=False, type=bool)
#parse them
args=parser.parse_args()


#variables in ALL_CAPS are specfically for use in the raw #bash environment
if args.subj == ['all']:
	sub_args = get_sub_args()
else:
	sub_args = args.subj

for sub in sub_args:
	sub = int(sub)
	SUBJ = 'Sub{0:0=3d}'.format(sub)

	print('splish splash Sub{0:0=3d}'.format(sub))

	fs_dir = data_dir + SUBJ + 'fs/'


	#tell python where the raw folder is
	raw = data_dir + SUBJ + '/raw/'

	bold = data_dir + SUBJ + '/bold/'

	day1 = bold + 'day1/'
	day2 = bold + 'day2/'

	#anatomy folder for t1mprage
	anatomy = data_dir + SUBJ + '/anatomy/'

	#find the mask folder
	mask = data_dir + SUBJ + '/mask/'
	if 'mask' not in os.listdir(data_dir + SUBJ):
		os.mkdir(mask)

	#directory for labels once made
	if os.path.isdir(fs_dir):
		label = fs_dir + 'annot2label/'
		if 'annot2label' not in os.listdir(fs_dir):
			os.mkdir(label)

	#registration populates a lot of files so need its own folder
	if os.path.isdir(fs_dir):
		registration = fs_dir + 'reg/'
		if 'reg' not in os.listdir(fs_dir):
			os.mkdir(registration)


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


	#convert dicom to nifti
	if args.convert == True:
		#collect all the phases in the raw folder
		runs_to_convert = { i: file for i, file in enumerate(os.listdir(raw)) if '.' not in file and file in run_key }

		#convert and zip all phases using dcm2niix in bash
		{ ( os.system('cd ' + data_dir + SUBJ + '/raw/' + runs_to_convert[i]),
			os.system(
			'dcm2niix '
			'-o ' + bold + ' '
			'-z y -m y '
			'-f %f ' + raw + runs_to_convert[i]),
			os.system('cd ..')
			) for i in runs_to_convert }

		#clean out the .json files
		dust = { file for file in os.listdir(bold) if '.json' in file}
		#and delete them
		{ os.remove(bold + i) for i in dust }

		#rename files from PHASE to run000
		{ os.rename(bold + file ,bold + run_key[file[:-7]] + '.nii.gz') for file in os.listdir(bold) if file[:-7] in run_key}

		#move the t1mprage to the anatomy folder
		os.rename(bold + 'struct.nii.gz', anatomy + 'struct.nii.gz')

		#move the runs to their specific folder within the day folders
		run_loc = glob('%s/day*/run00*/'%(bold))
		mov_runs = glob('%s/run00*.nii.gz'%(bold))

		{ os.rename(run, run_loc[i] + run[-13:]) for i, run in enumerate(mov_runs) }
		#{ os.rename(bold + run, run_loc[i] + run) for i,run in enumerate(os.listfile(bold)) if 'run' in run}

	if args.recon == True:
		os.system('recon-all -s %sfs -i %s/struct.nii.gz -all'%(SUBJ, anatomy))	

	#skull strip for GLM
	if args.skull_strip == True:

		print('Creating whole brain mask from Freesurfer brainmask.mgz')
		#create a whole brain mask using mri_convert of the brainmask.mgz

		os.system(
			'mri_convert' + ' ' +
			fs_dir + 'mri/brainmask.mgz' + ' ' +
			'-rl' + ' ' + anatomy + 'struct.nii.gz' + ' ' +
			anatomy + 'struct_brain.nii.gz'
			)

	if args.mask == True:
		
		#populate all of the label files from each hemisphere's annotate file
		if len(os.listdir(label)) < 2:
			lh_pop = Popen(['mri_annotation2label', '--s', '%sfs'%(SUBJ), '--hemi','lh','--outdir',label],env=env)
			lh_pop.wait()
			rh_pop = Popen(['mri_annotation2label', '--s', '%sfs'%(SUBJ), '--hemi','rh','--outdir',label],env=env)
			rh_pop.wait()

		#create the reference run (mean of the 1st run) if it does not exist
		if 'refvol.nii.gz' not in os.listdir(bold):
			os.system('fslmaths ' + day1 + 'run001/run001.nii.gz -Tmean ' + bold + 'refvol')

		#create registration matrix if it does not exist
		if 'RegMat.dat' not in os.listdir(registration):
			os.system('bbregister --s %sfs --mov %srefvol.nii.gz --init-fsl --bold --reg %sRegMat.dat'%(SUBJ,bold,registration))

		#convert the labels associated with ventral temporal cortex into a mask for each hemisphere
		#right now .3 fill threshold seems to be better, .1 was getting a lot of voxels that looked like they were outside the brain
		#maybe also want to play around with frac
		#ALSO, may want to look at adding in Lateral occipital cortex (LOC), resulting in a joint LOC_VT mask á la ComCon
		if 'VTC_mask.nii.gz' not in os.listdir(mask):
			print('Generating VTC mask')
			#left hemishpere
			lhd1_mask = Popen(['mri_label2vol', '--subject', '%sfs'%(SUBJ),
			'--label', '%slh.fusiform.label'%(label),
			'--label', '%slh.parahippocampal.label'%(label),
			'--label', '%slh.inferiortemporal.label'%(label),
			'--label', '%slh.lingual.label'%(label),
			'--temp', '%srefvol.nii.gz'%(bold),
			'--reg', '%sRegMat.dat'%(registration),
			'--proj', 'frac', '0', '1', '.1',
			'--fillthresh', '.3',
			'--hemi', 'lh',
			'--o', '%slh_VTC.nii.gz'%(mask)],
			env=env)
			lhd1_mask.wait()

			#right hemisphere
			rhd1_mask = Popen(['mri_label2vol', '--subject', '%sfs'%(SUBJ),
			'--label', '%srh.fusiform.label'%(label),
			'--label', '%srh.parahippocampal.label'%(label),
			'--label', '%srh.inferiortemporal.label'%(label),
			'--label', '%srh.lingual.label'%(label),
			'--temp', '%srefvol.nii.gz'%(bold),
			'--reg', '%sRegMat.dat'%(registration),
			'--proj', 'frac', '0', '1', '.1',
			'--fillthresh', '.3',
			'--hemi', 'rh',
			'--o', '%srh_VTC.nii.gz'%(mask)],
			env=env)
			rhd1_mask.wait()
			
			#add the hemispheres together
			print('Combining hemispheres via fslmaths...')
			os.system('fslmaths %slh_VTC.nii.gz -add %srh_VTC.nii.gz -bin %sVTC_mask.nii.gz'%(mask,mask,mask))

	#	else if 'VTC_mask.nii.gz' in os.listdir(mask):
	#		print('VTC mask already exists')

		if 'LOC_mask.nii.gz' not in os.listdir(mask):
			print('Generating LOC mask')

			lhd2_mask = Popen(['mri_label2vol', '--subject', '%sfs'%(SUBJ),
			'--label', '%slh.lateraloccipital.label'%(label),
			'--temp', '%srefvol.nii.gz'%(bold),
			'--reg', '%sRegMat.dat'%(registration),
			'--proj', 'frac', '0', '1', '.1',
			'--fillthresh', '.3',
			'--hemi', 'lh',
			'--o', '%slh_LOC.nii.gz'%(mask)],
			env=env)
			lhd2_mask.wait()
			
			rhd2_mask = Popen(['mri_label2vol', '--subject', '%sfs'%(SUBJ),
			'--label', '%srh.lateraloccipital.label'%(label),
			'--temp', '%srefvol.nii.gz'%(bold),
			'--reg', '%sRegMat.dat'%(registration),
			'--proj', 'frac', '0', '1', '.1',
			'--fillthresh', '.3',
			'--hemi', 'rh',
			'--o', '%srh_LOC.nii.gz'%(mask)],
			env=env)
			rhd2_mask.wait()

			print('Combining hemispheres via fslmaths...')
			os.system('fslmaths %slh_LOC.nii.gz -add %srh_LOC.nii.gz -bin %sLOC_mask.nii.gz'%(mask,mask,mask))

		if 'LOC_VTC_mask.nii.gz' not in os.listdir(mask):
			print('Combining LOC & VTC masks')
			os.system('fslmaths %sLOC_mask.nii.gz -add %sVTC_mask.nii.gz -bin %sLOC_VTC_mask.nii.gz'%(mask,mask,mask))

		else:
			print('All masks already exist')

	#run motion correction
	if args.motion_correct == True:

		#create the reference run (mean of the 1st run) if it does not exist
		if 'refvol.nii.gz' not in os.listdir(bold):
			os.system('fslmaths ' + day1 + 'run001/run001.nii.gz -Tmean ' + bold + 'refvol')

		#find all the runs
		runs_to_mc = glob('%s/day*/run00*/run00*.nii.gz'%(bold))

		#make the names with mc_run000.nii.gz
		mc_runs = [ ( run[:-13] + 'mc_' + run[-13:] ) for run in runs_to_mc ]

		#combine those two things into a dictionary for ease
		mc_index = { run:mc_runs[i] for i, run in enumerate(runs_to_mc) }

		#run mcflirt (FSL motion correction)
		[ ( print('Motion correcting %s'%(run[-13:-7])),
			os.system('mcflirt -in %s -out %s -refvol %srefvol.nii.gz'%(run,mc_index[run],bold))
			) for run in runs_to_mc ]


	if args.transform == True:
		print('Transforming:')
		
		#identify the runs to work
		run_names = { i:file for i, file in enumerate(glob('%s/day*/run00*/mc_run00*.nii.gz'%(bold))) }
		print('Found %s runs...'%(len(run_names)))

		#load the nifti info
		print('Loading niftis...')
		run_metas = { run: nib.load(run_names[run]) for run in run_names  }
		
		#load the actual array data
		print('Getting data...')
		run_imgs = { run: run_metas[run].get_data() for run in run_metas }
		
		#detrend along 4th axis
		print('Detrending...')
		detr_imgs = { run: detrend(run_imgs[run], axis=3) for run in run_imgs }

		print('Z-scoring...')
		#zscore them along 4th axis
		z_imgs = { run: sp.stats.mstats.zscore(detr_imgs[run], axis=3) for run in detr_imgs }

		#convert back to nifti so they can be saved
		#nii_imgs = { run: nib.Nifti1Image(z_imgs[run]) }

		#for now im ok with just saving the data as numpy arrays and gunzipping them,
		#this makes sense because the first pass of my MPVA analysis will be done with scikit and not PyMVPA
		print('Saving as numpy array with prefix "zd_"...')
		{ ( np.savez_compressed('%s'%(run_names[run][:-16] + 'zd_' + run_names[run][-16:-7]), z_imgs[run] ) ) for run in z_imgs }

		#Zip 'em up!
		#print('Gunzipping...')
		#this worked but just saved them to data_dir, go figure
		#{( os.system('tar -cvzf %s %s'%(file[-15:] + '.gz', file)) ) for file in glob('%s/day*/run00*/zd_mc_run00*.npy'%(bold)) }

		#{ ( os.system('tar %s'%(run_names[run][:-16] + 'zd_' + run_names[run][-16:-7] + '.npy')) )for run in run_names }

	#prep single runs for MVPA
	for phase in args.prep_runs:
		#load in the run so that we can apply the mask
		raw = ('%s/%s/bold/%s'%(data_dir,SUBJ,phase2location[phase]))

		print('Loading %s...'%(phase))
		data = np.load(raw)

		#load in the mask as a binary array
		mask = nib.load('%s/%s/mask/LOC_VTC_mask.nii.gz'%(data_dir,SUBJ)).get_data()

		#find the locations of the non-zero mask bits
		roi_ind = np.where(mask!=0)
		print('loaded mask, ROI contains %s voxels'%(roi_ind[0].shape[0]))
		#flatten the array
		roi_ind = np.ascontiguousarray(roi_ind)
		#transpose it
		roi_ind = np.ascontiguousarray(roi_ind[0:3,:].T)


		#filter 'data' to include only voxels as selected by the mask, as well as flattening the data array to be Voxel x Timepoint
		#this is the same thing as sample x feature in machine learning speak
		#first create a new array called 'roi' that is the correct shape
		print('Flattening data & applying mask...')
		dim = [data.shape[3],roi_ind.shape[0]]
		roi = np.empty(dim)
		#then, for every TR
		for sample in range(roi.shape[0]):
			#and for every voxel in that TR THAT IS ALSO IN THE MASK
			for feature, voxel in enumerate(np.rollaxis(roi_ind,axis=0)):
				#collect its index
				indx = np.append(voxel,sample)
				#and add it to the roi array
				roi[sample, feature] = data[indx[0],indx[1],indx[2],indx[3]]

		print('Saving output to subjects bold directory...')
		#save that array!

		np.savez_compressed('%s/%s/bold/%s/prepped_%s'%(data_dir,SUBJ,phase2location[phase][:-16],phase2location[phase][18:-4]), roi)



	#not sure when I'll need this tbh
	if args.prep_all_data == True:
		
		for phase in phase2location:


			#load in the run so that we can apply the mask
			raw = ('%s/%s/bold/%s'%(data_dir,SUBJ,phase2location[phase]))

			print('Loading %s...'%(phase))
			data = np.load(raw)

			#load in the mask as a binary array
			mask = nib.load('%s/%s/mask/LOC_VTC_mask.nii.gz'%(data_dir,SUBJ)).get_data()

			#find the locations of the non-zero mask bits
			roi_ind = np.where(mask!=0)
			print('loaded mask, ROI contains %s voxels'%(roi_ind[0].shape[0]))
			#flatten the array
			roi_ind = np.ascontiguousarray(roi_ind)
			#transpose it
			roi_ind = np.ascontiguousarray(roi_ind[0:3,:].T)


			#filter 'data' to include only voxels as selected by the mask, as well as flattening the data array to be Voxel x Timepoint
			#this is the same thing as sample x feature in machine learning speak
			#first create a new array called 'roi' that is the correct shape
			print('Flattening data & applying mask...')
			dim = [data.shape[3],roi_ind.shape[0]]
			roi = np.empty(dim)
			#then, for every TR
			for sample in range(roi.shape[0]):
				#and for every voxel in that TR THAT IS ALSO IN THE MASK
				for feature, voxel in enumerate(np.rollaxis(roi_ind,axis=0)):
					#collect its index
					indx = np.append(voxel,sample)
					#and add it to the roi array
					roi[sample, feature] = data[indx[0],indx[1],indx[2],indx[3]]

			print('Saving output to subjects bold directory...')
			#save that array!

			np.savez_compressed('%s/%s/bold/%s/prepped_%s'%(data_dir,SUBJ,phase2location[phase][:-16],phase2location[phase][18:-4]), roi)

	if args.compress == True:

		for phase in phase2location:

			old = np.load('%s%s'%(bold,phase2location[phase]))

			np.savez_compressed('%s%s'%(bold,phase2location[phase][:-4]), old)





























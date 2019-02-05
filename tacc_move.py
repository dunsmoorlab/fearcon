from preprocess_library import preproc, meta
from fc_config import data_dir, sub_args, fsub, init_dirs, phase2rundir, dc2nix_run_key, mvpa_prepped, nifti_paths, raw_paths, avg_mc_paths, py_run_key
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
from fc_config import day_dict

tacc_subs = [23,24,25,26,122,123,124,125]

tacc_dir = '$WORK/FearCon/'

# for sub in tacc_subs:

# 	fsub = 'Sub{0:0=3d}'.format(sub)

# 	subj_dir = os.path.join(tacc_dir, fsub)
# 	print(subj_dir)
# 	# os.system('mkdir -p %s'%(subj_dir))

# 	img_dir = os.path.join(subj_dir, 'bold')
# 	print(img_dir)
# 	# os.system('mkdir -p %s'%(img_dir))

# 	reg_dir = os.path.join(subj_dir, 'reg')
# 	print(reg_dir)
# 	# os.system('mkdir -p %s'%(reg_dir))
	

for sub in tacc_subs:

	subj = meta(sub)

	tacc_temp = os.path.join(subj.subj_dir,'tacc_temp')
	print(tacc_temp)
	os.system('mkdir -p %s'%(tacc_temp))

	refvol = subj.refvol_be

	copyfile(refvol, os.path.join(tacc_temp, refvol[-16:]))


	epi = glob('%s/day*/run***/be_avg_mc_run***.nii.gz'%(subj.bold_dir))
	txt = glob('%s/day*/run***/antsreg/transforms/run***-refvol_0GenericAffine.txt'%(subj.bold_dir))

	for i in range(len(epi)):
		copyfile(epi[i], os.path.join(tacc_temp, epi[i][-23:]))
		copyfile(txt[i], os.path.join(tacc_temp, txt[i][-32:]))







	# for phase in phase2rundir:

	# 	srcdir = os.path.join(subj.bold_dir, phase2rundir[phase])
	# 	srcvol = os.path.join(srcdir, 'be_avg_mc_' + py_run_key[phase] + '.nii.gz')

	# 	reg_xfm = os.path.join(self.subj.bold_dir, phase2rundir[phase], 'antsreg', 'transforms')
	# 	xfm_base = os.path.join(reg_xfm, '%s-refvol_' % (py_run_key[phase]))
	# 	txt_file = xfm_base + '0GenericAffine.txt'
	

	# txt_file, refvol, srcvol, reg_file))

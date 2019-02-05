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

subj = meta(1)


phase = 'extinction_recall'

reg_xfm = os.path.join(subj.bold_dir, phase2rundir[phase], 'antsreg', 'transforms')
reg_data = os.path.join(subj.bold_dir, phase2rundir[phase], 'antsreg', 'data')
os.system('mkdir -p %s' % reg_data)
os.system('mkdir -p %s' % reg_xfm)

srcdir = os.path.join(subj.bold_dir, phase2rundir[phase])
refdir = subj.refvol_dir

srcvol = os.path.join(srcdir, 'be_avg_mc_' + py_run_key[phase] + '.nii.gz')
refvol = subj.refvol_be
bold = os.path.join(subj.bold_dir, raw_paths[phase])
os.system('cat %smc_%s.mat/MAT* > %smc_%s.cat'%(srcdir, py_run_key[phase], srcdir, py_run_key[phase]))
mcf_file = os.path.join(srcdir, 'mc_%s.cat'%(py_run_key[phase]))

mask = os.path.join(refdir, 'fm', 'brainmask.nii.gz')


# output files
bold_reg = os.path.join(srcdir, 'reg_%s.nii.gz'%(py_run_key[phase]))
# bold_reg_avg = impath(srcdir, 'bold_reg_avg')
# bold_reg_avg_cor = impath(srcdir, 'bold_reg_avg_cor')
# bias = impath(srcdir, 'bold_reg_avg_bias')
output = os.path.join(srcdir, 'be_mc_reg_%s.nii.gz'%(py_run_key[phase]))
bold_init = os.path.join(srcdir, 'bold_reg_init.nii.gz')
bold_init_avg = os.path.join(srcdir, 'bold_reg_init_avg.nii.gz')

# nonlinear registration to the reference run
xfm_base = os.path.join(reg_xfm, '%s-refvol_' % (py_run_key[phase]))
itk_file = xfm_base + '0GenericAffine.mat'
txt_file = xfm_base + '0GenericAffine.txt'
os.system('antsRegistrationSyN.sh -d 3 -m {mov} -f {fix} -o {out} -n 4 -t s'.format(
	mov=srcvol, fix=refvol, out=xfm_base))

# convert the affine part to FSL format
reg_file = os.path.join(reg_xfm, '%s-refvol.mat' % (py_run_key[phase]))
os.system('ConvertTransformFile 3 %s %s' % (itk_file, txt_file))


os.system('c3d_affine_tool -itk %s -ref %s -src %s -ras2fsl -o %s' % (
	txt_file, refvol, srcvol, reg_file))

# apply motion correction, unwarping, and affine co-registration
os.system('applywarp -i %s -r %s -o %s --premat=%s --postmat=%s --interp=spline --rel --paddingsize=1' %
	(bold, refvol, bold_init, mcf_file, reg_file))

warp = xfm_base + '1Warp.nii.gz'
os.system('antsApplyTransforms -d 3 -e 3 -i {} -o {} -r {} -t {} -n BSpline'.format(
	bold_init, bold_reg, refvol, warp))

os.remove(bold_init)

os.system('fslmaths %s -mas %s %s' % (
	bold_reg, mask, output))








# for phase in ['extinction_recall']:#phase2rundir.keys():

# 	in_vol = os.path.join(subj.bold_dir, raw_paths[phase])
# 	out_vol1 = os.path.join(subj.bold_dir, in_vol[:-13] + 'mc2refvol_be' + in_vol[-13:])

# 	mcflirt_cmd1 = 'mcflirt -in %s -out %s -reffile %s'%(in_vol, out_vol1, subj.refvol_be)

# 	print(in_vol[-13:])

# 	os.system(mcflirt_cmd1)
# 	print('2')
# 	out_vol2 = os.path.join(subj.bold_dir, in_vol[:-13] + 'mc2refvol' + in_vol[-13:])
# 	mcflirt_cmd2 = 'mcflirt -in %s -out %s -reffile %s'%(in_vol, out_vol2, subj.refvol)
# 	os.system(mcflirt_cmd2)
# 	print('3')

# 	reg_cmd1 = 'flirt -in %s -ref %s -omat %s -out %s'%(os.path.join(subj.bold_dir,nifti_paths[phase]), subj.refvol_be,
# 		os.path.join(subj.bold_dir, phase2rundir[phase], '%s2refvol.mat'),
# 		os.path.join(subj.bold_dir, phase2rundir[phase], 'mc_reg_ref_be.nii.gz' ))
# 	os.system(reg_cmd1)
# 	print('4')
# 	reg_cmd2 = 'flirt -in %s -ref %s -omat %s -out %s'%(os.path.join(subj.bold_dir,nifti_paths[phase]), subj.refvol,
# 		os.path.join(subj.bold_dir, phase2rundir[phase], '%s2refvol.mat'),
# 		os.path.join(subj.bold_dir, phase2rundir[phase], 'mc_reg_ref.nii.gz' ))
# 	os.system(reg_cmd2)

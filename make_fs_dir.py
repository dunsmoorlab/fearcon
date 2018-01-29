from fc_config import data_dir, sub_args, fsub

import os
import sys


fs_dir = '/Users/ach3377/Documents/FearCon_Freesurfer'

if not os.path.exists(fs_dir):
	os.mkdir(fs_dir)


for sub in sub_args:

	nsub = fsub(sub)

	sub_fs = os.path.join(data_dir,nsub,nsub + 'fs/')

	os.system('ln -s %s %s'%(sub_fs, fs_dir))

#from glob import glob

#masks = glob('%s/Sub[0-9][0-9][0-9]/mask/LOC_VTC_mask.nii.gz'%(data_dir))

#from nilearn.image import mean_img

#mean_mask = mean_img(masks)

#mean2 = mean_mask.get_data()

#dele = mean2 < 1

#mean2[dele] = 0

#import nibabel as nib

#mean2.header = mean_mask.header

#nib.save(mean2,data_dir)

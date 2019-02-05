import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib

from nilearn.plotting import plot_stat_map, plot_anat, plot_img
from nilearn.image import mean_img, image, new_img_like
from nilearn.input_data import NiftiMasker

from nistats.first_level_model import FirstLevelModel
from nistats.reporting import plot_design_matrix

from fc_config import *

from glm_timing import glm_timing

from preprocess_library import meta

sub = 1
subj = meta(sub)
phase = 'fear_conditioning'
events = glm_timing(sub,phase).phase_events()
func = os.path.join(subj.bold_dir,nifti_paths[phase])

sub = 1
subj = meta(sub)
phase = 'fear_conditioning'
events = glm_timing(sub,phase).phase_events()
func = os.path.join(subj.bold_dir,nifti_paths[phase])
masker = NiftiMasker(mask_img=subj.brainmask, smoothing_fwhm=5, detrend=True,
                    high_pass=0.0078, t_r=2, standardize=True).fit(imgs=func)

motion_confounds = np.loadtxt(os.path.join(subj.bold_dir, phase2rundir[phase], 'motion_assess', 'confound.txt'))
if len(motion_confounds) > 0:

    if len(motion_confounds.shape) == 2:
        _columns = range(motion_confounds.shape[1])
    elif len(motion_confounds.shape) == 1:
        _columns = range(1)

    motion_confounds_df = pd.DataFrame(motion_confounds, columns=_columns)

else: 
    motion_confounds_df = None


glm = FirstLevelModel(t_r=2, mask=masker, slice_time_ref=.5, hrf_model='glover + derivative', signal_scaling=False, minimize_memory=False, verbose=2, noise_model='ols')
glm = glm.fit(func, events, confounds=motion_confounds_df)
design_matrix = glm.design_matrices_[0]

for key in glm.results_[0].keys():
	print(glm.results_[0][key].predicted.shape)

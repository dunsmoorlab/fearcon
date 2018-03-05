from fc_config import data_dir, sub_args, nifti_paths, PPA_prepped
from pygmy import first_level
from glm_timing import glm_timing
from preprocess_library import meta
from L2_pygmy import second_level

import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import os

from nilearn.plotting import plot_stat_map, plot_glass_brain
from glob import glob
from nilearn.input_data import NiftiMasker
from scipy.stats import ttest_1samp
from nistats.utils import z_score
from nistats.thresholding import map_threshold

from nilearn.datasets import load_mni152_template
from nilearn.image import resample_to_img, mean_img
from nilearn.masking import apply_mask, unmask
from nilearn import signal

class loc_glm(object):
	#to do is save the not - scene contrast and do the same to make sure were actually getting different voxels
	def __init__(self,subj):

		#import subject meta
		self.subj = meta(subj)

		print('Running 1st level glm on %s Localizer'%(self.subj.num))
		#load the motion corrected runs
		self.loc_runs = { phase: first_level.load_bold(self, phase) for phase in nifti_paths if 'localizer' in phase }

		#apply the mask
		# self.loc_runs = { phase: apply_mask(self.loc_runs[phase], self.subj.maskvol, ensure_finite=True) for phase in self.loc_runs }
		
		#and then unmask it so its only the mask voxels
		# self.loc_runs = { phase: unmask(self.loc_runs[phase], self.subj.maskvol) for phase in self.loc_runs }
		
		#run the glm
		self.results = { phase: self.run_glm(phase) for phase in self.loc_runs }
		
		self.vis_results()

	def run_glm(self, phase):

		#initialize the glm, settings found in pygmy.py
		glm = first_level.init_glm(self)
		
		#load the events
		events = self.loc_events(self.subj.num, phase)

		#fit the glm
		glm, design_matrix = first_level.fit_glm(self, glm, self.loc_runs[phase], events)

		#set the contrast
		contrast = self.loc_contrasts(design_matrix, 'scene', 'not')

		#collect the zmap
		z_map = first_level.glm_stats(self, glm, contrast)

		#save it
		nib.save(z_map, os.path.join(self.subj.glm_mask, '%s_scene_zmap.nii.gz'%(phase)))
		
		return z_map

	#this over that
	def loc_contrasts(self, design_matrix, this, that):
		#set up contrast
		contrast_matrix = np.eye(design_matrix.shape[1])

		#populate it
		contrasts = dict([(column, contrast_matrix[i])
					for i, column in enumerate(design_matrix.columns)])

		#take this - that
		contrast = contrasts[this] - contrasts[that]
		self.peepthis = contrast
		return contrast

		#the goal here is to find voxels that are more active to scenes compared to anything else


	def loc_events(self, subj, phase):

		#load the glm timing
		events = glm_timing(subj, phase).phase_events()

		#combine across scenes
		scenes = ['indoor','outdoor']

		for scene in scenes:
			
			events.trial_type.loc[events.trial_type == scene] = 'scene'

		events.trial_type.loc[events.trial_type != 'scene'] = 'not'

		return events

	def vis_results(self):

		for phase in self.results:

			plot_stat_map(self.results[phase], bg_img = self.subj.refvol, threshold=2.33, display_mode='z', cut_coords=3,
            	 black_bg=True, title='%s: Scene > Not'%(phase))


class combine_loc(object):

	def __init__(self, subj, display=False):

		self.subj = meta(subj)

		print('2nd level on %s'%(self.subj.num))

		locl1 = glob('%slocalizer_*_scene_zmap.nii.gz'%(self.subj.glm_mask + os.sep))

		self.lev1 = locl1
		
		if display:
			self.display_first_levels(locl1)

		l2_glm, design_matrix = second_level.init_level2(self, locl1, smoothing=5)

		l2_glm, z_map = second_level.fit_level2(self, l2_glm, locl1, design_matrix)

		nib.save(z_map, os.path.join(self.subj.glm_mask, 'combined_loc_scene_zmap.nii.gz'))

		self.view_res(z_map)

		#self.cluster(locl1)

	def display_first_levels(self, first_levels):

		fig, axes = plt.subplots(nrows=1, ncols=2)
		for run, zmap in enumerate(first_levels):
			plot_glass_brain(zmap, colorbar=True, threshold=2.58, title='localizer_%s scene > other'%(run + 1),
				axes=axes[run],
				plot_abs=True, display_mode='z')


	def view_res(self, res):

		plot_stat_map(res, bg_img=self.subj.refvol, threshold=2.33, display_mode='z', cut_coords=3,
						black_bg=True, title='Combined Localizer Scene > Not')


	def cluster(self, first_levels):

		masker = NiftiMasker(smoothing_fwhm=5, memory='nilearn_cache', memory_level=1)

		first_levels = masker.fit_transform(first_levels)

		_, p_values = ttest_1samp(first_levels, 0)

		z_map = masker.inverse_transform(z_score(p_values))

		thresh_map1, thresh1 = map_threshold(z_map, threshold=.005, height_control='fpr',
								cluster_threshold=10)

		thresh_map2, thresh2 = map_threshold(z_map, threshold=.05, height_control='fpr')

		display = plot_stat_map(z_map, title='Raw_z_map')
		
		plot_stat_map(thresh_map1, cut_coords=display.cut_coords, 
						title='Thresholded z map, fpr <.005, clusters > 10 voxels',
						threshold=thresh1)

		plot_stat_map(thresh_map2, cut_coords=display.cut_coords,
                       title='Thresholded z map, expected fdr = .05',
                       threshold=thresh2)


class combine_subs(object):

	def __init__(self):

		print('Combining all subs 2nd levels')

		lev3 = self.get_sub_res('combined_loc_scene_zmap.nii.gz')
		#lev3 = glob('%sSub***/mask/glm_masking/%s'%(data_dir,'combined_loc_scene_zmap.nii.gz'))

		l3_glm, design_matrix = second_level.init_level2(self, lev3, smoothing=5)

		l3_glm, z_map = second_level.fit_level2(self, l3_glm, lev3, design_matrix)

		#z_map, z = map_threshold(z_map, threshold=.001, height_control='fpr')

		nib.save(z_map, os.path.join(data_dir, 'group_pyGLM', 'level3_loc_scene_zmap.nii.gz'))


	def get_sub_res(self, thing_to_get):

		template = load_mni152_template()
		
		level2 = glob('%sSub***/mask/glm_masking/%s'%(data_dir,thing_to_get))

		level2 = {i: resample_to_img(img, template) for i, img in enumerate(level2)}

		return list(level2.values())

#does all the things listed above
def run_all():
	for sub in sub_args:
		loc_glm(sub)
		combine_loc(sub)
	combine_subs()


def make_sub_mask():

	sub_masks = glob('%sSub***/mask/LOC_VTC_mask.nii.gz'%(data_dir))

	



	std_mask = os.path.join(data_dir, 'group_pyGLM', 'level3_loc_scene_zmap.nii.gz')

	for sub in sub_args:

		subj = meta(sub)

		PPA_mask = resample_to_img(std_mask, subj.refvol)

		nib.save(PPA_mask,os.path.join(subj.glm_mask, 'PPA_mask.nii.gz'))

		#idk this isnt working right now but jsut do it manually in terminal
		#os.system('fslmaths %sgroup_pgGLM/level3_loc_scene_zmap.nii.gz -bin %sgroup_pyGLM/std_thr_scene_mask.nii.gz'%(data_dir, data_dir))

def glm_mvpa_prep():

	#this is a new thing im trying, so it needs to be compared to my other method of detrending and z-scoring
	#this uses signal theory to standardize the data, which uses Sum Squares instead of z-score
	#ensure finite removes any NaN if they're there

	#follow up 2/5: while the individual data points between this method and my manual method vary on the order of .00000####,
	#after running many MVPA decoding schemas the two data sets produced IDENTICAL classification results on numereous different conditions,
	#so I'm going with these because I trust it more!
	for sub in sub_args:

		subj = meta(sub)

		print('Prepping runs for MVPA with nilearn signal.clean()')


		#load the runs
		mc_runs = { phase: nib.load('%s%s'%(subj.bold_dir, nifti_paths[phase])) for phase in nifti_paths }

		print('Applying Mask')
		#have to mask it _first_
		mc_runs = { phase: apply_mask(mc_runs[phase], subj.PPAmaskvol) for phase in mc_runs }

		print('Cleaning signal')
		mc_runs = { phase: signal.clean(mc_runs[phase], detrend=True, standardize=True, t_r=2, ensure_finite=True) for phase in mc_runs }
		
		print('Saving')
		{ np.savez_compressed( '%s%s'%(self.subj.bold_dir, PPA_prepped[phase]),  mc_runs[phase] ) for phase in mc_runs }













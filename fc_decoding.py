import os
import numpy as np
import pandas as pd
import nibabel as nib

from glob import glob
from nilearn import signal
from nilearn.masking import apply_mask, unmask


from fc_config import data_dir, sub_args, init_dirs, nifti_paths
from preprocess_library import meta
from glm_timing import glm_timing


class decode(object):

	def __init__(self,subj,imgs='tr'):

		self.subj = meta(subj)

		self.load_localizer(imgs)

		self.loc_dat = self.mask()

		self.loc_dat = self.clean_signal()
	
	#takes in 'tr' or 'beta' for imgs
	def load_localizer(self,imgs):

		#if you want to load in the tr by tr data (not beta)
		if imgs == 'tr':

			#load in the motion corrected niftis
			loc_runs = { phase: '%s%s'%(self.subj.bold_dir, nifti_paths[phase]) for phase in nifti_paths if 'localizer' in phase }

			self.loc_dat = { phase: nib.load(loc_runs[phase]) for phase in loc_runs }


	#apply the mask du jour
	def mask(self, maskvol=None, data=None):

		if maskvol is None:
			maskvol = self.subj.maskvol

		#this is in the xval script so the defaul data is the localizer	
		if data is None:
			data = self.loc_dat

		#apply the mask
		data = { phase: apply_mask(data[phase], maskvol) for phase in data }		

		return data


	def clean_signal(self, data=None):
		
		#this is in the xval script so the defaul data is the localizer	
		if data is None:
			data = self.loc_dat

		#this is a new thing im trying, so it needs to be compared to my other method of detrending and z-scoring
		#this uses signal theory to standardize the data, which uses Sum Squares instead of z-score
		#ensure finite removes any NaN if they're there
		data = { phase: signal.clean(data[phase], detrend=True, standardize=True, t_r=2, ensure_finite=True) for phase in data }

		return data


	def better_tr_labels(self,data=None):

		#this is in the xval script so the defaul data is the localizer	
		if data is None:
			data = self.loc_dat

		out_labels = dict.fromkeys(list(data.keys()))

		#for phase in data:
		for phase in data:

			#grab the onset information
			events = glm_timing(self.subj.num,phase).phase_events()

			#calculate the TR in which the stims are first presented, as well as the remainder
			events['start_tr'], events['start_rem'] = divmod(events.onset, 2)
			events['end_tr'], events['end_rem'] = divmod(events.onset+events.duration, 2)


			#create the label structure
			labels = pd.Series(index=np.arange(data[phase].shape[0]))


			#for every event in the run:
			for trial in events.index:

				#i did this just to avoid lots of typing, makes it look cleaner
				start_tr = events.start_tr[trial]
				start_rem = events.start_rem[trial]

				end_tr = events.end_tr[trial]
				end_rem = events.end_rem[trial]
				
				label = events.trial_type[trial]

				#if the stimulus starts and ends within the same TR, then label that TR!
				if start_tr == end_tr:
					labels[start_tr] = label

				#otherwise if the stimulus starts pretty early in the starting TR,
				#but loses .3s of it in the next TR, go ahead and label the start TR
				elif start_rem <= 1.3:
					labels[start_tr] = label

				#otherwise, if it ends within a second of the next TR, label the next one
				elif end_rem <= .5:
					labels[end_tr] = label

				#and then also the label the next one for this condition
				elif start_rem >= 1.5 and end_rem >= .5:
					labels[end_tr] = label						


			out_labels[phase] = labels

			for label in labels.unique():
				print(label, len(np.where((labels==label) == True)[0]))

		return out_labels

'''
if the ending time for this stimulus goes beyond BOTH the TR window (.3 of the next TR)
as well as past the 


elif onset + duration > (window[-1] + .5):

self.labels[tr+1] = label

#something I need to check is how fixing the wholes in the label changes things
'''





















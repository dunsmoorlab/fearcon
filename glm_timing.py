import os
import argparse
import numpy as np
import pandas as pd
from toolz import interleave

from fc_config import mvpa_prepped, dataz, data_dir, get_subj_dir, get_bold_dir

class glm_timing(object):

	#this is a hacky line I need because of e-prime
	first_line = {'baseline': 0, 'fearconditioning': 48, 'extinction': 96}


	def __init__(self,subj=0,phase=None):

		#initalize directories
		self.subj = subj

		self.fsub = 'Sub{0:0=3d}'.format(self.subj)

		self.subj_dir = get_subj_dir(subj)

		self.bold_dir = get_bold_dir(self.subj_dir)

		#load the e-prime meta data csv
		self.meta = pd.read_csv(os.path.join(self.subj_dir,'behavior',self.fsub + '_elog.csv'))

		self.phase = phase
		
		#correct for another dumb e-prime thing
		if self.phase == 'fear_conditioning':
			self.phase = 'fearconditioning'

		self.phase_events()
	

	def phase_events(self):

		#isoloate the metadata for this phase
		#handle the case for extinction first
		if self.phase == 'extinction':
			self.phase_meta = self.ext_meta()
		else:
			self.phase_meta = self.meta[self.meta.phase == self.phase]
		
		#find the start time
		self.phase_start = self.phase_meta['InitialITI.OnsetTime'][self.first_line[self.phase]]

		#create the output structure
		self.phase_timing = pd.DataFrame([],columns=['duration','onset','trial_type'])

		#find the onsets
		self.phase_timing.onset = self.phase_meta['stim.OnsetTime'] - self.phase_start

		#find the durations
		self.phase_timing.duration = self.phase_meta['stim.Duration']

		#find the trial types
		self.phase_timing.trial_type = self.phase_meta.cstype

		#convert onset and duration to seconds
		self.phase_timing.onset /= 1000
		self.phase_timing.duration /= 1000

		#make sure that the index is correct (starting at 0 for each phase)
		if self.phase_timing.index[0] != 0:
			self.phase_timing = self.phase_timing.reset_index(drop=True)

		return self.phase_timing


	def ext_meta(self):
		
		self.phase3_stims = pd.Series(0)
		self.phase3_unique_loc = pd.Series(0)
		q = 0

		for loc, unique in enumerate(self.meta.stims[self.meta.phase == 'extinction']):
			if not any(stim == unique for stim in self.phase3_stims):
				self.phase3_stims[q] = unique
				self.phase3_unique_loc[q] = loc + 96
				q = q + 1

		return self.meta.loc[self.phase3_unique_loc,]


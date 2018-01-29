import os
import argparse
import numpy as np
import pandas as pd
from toolz import interleave

from fc_config import mvpa_prepped, dataz, data_dir, get_subj_dir, get_bold_dir

class glm_timing(object):

	#this is a hacky line I need because of e-prime
	first_line = {'baseline': 0, 'fearconditioning': 48, 'extinction': 96, 'extinctionRecall': 384,
					'memory_run_1': 408, 'memory_run_2': 488, 'memory_run_3': 568,
					'localizer_1': 0, 'localizer_2':0 }


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
		if self.phase == 'extinction_recall':
			self.phase = 'extinctionRecall'

		self.phase_meta = self.get_phase_meta()


		#self.phase_events()

	def get_phase_meta(self):
		#isoloate the metadata for this phase
		#handle the case for extinction first
		if self.phase == 'extinction':
			phase_meta = self.ext_meta()
		
		elif self.phase[0:3] == 'mem':
			phase_meta = self.meta[self.meta.ExperimentName == '%s%s%s'%(self.meta.ExperimentName[408][:-4],'Run',int(self.phase[-1]) + 1)]
		
		elif self.phase[0:3] == 'loc':
			phase_meta = pd.read_csv(os.path.join(self.subj_dir,'behavior','run_logs','localizer_%s_meta.csv'%(self.phase[-1])))
			phase_meta = phase_meta[phase_meta.Procedure == 'LocStim']
		
		else:
			phase_meta = self.meta[self.meta.phase == self.phase]
		
		return phase_meta

	#create and return a pandas DataFrame for event timing
	#if con=True, then returns only the trial condition type
	def phase_events(self,con=False):

		
		#find the start time
		self.phase_start = self.phase_meta['InitialITI.OnsetTime'][self.first_line[self.phase]]

		#create the output structure
		self.phase_timing = pd.DataFrame([],columns=['duration','onset','trial_type'])

		if self.phase[0:3] == 'mem':
			#find the onsets
			self.phase_timing.onset = self.phase_meta['oldnew.OnsetTime'] - self.phase_start

			#find the durations
			self.phase_timing.duration = 3000

			#find the trial types
			self.phase_timing.trial_type = self.phase_meta.cstype


		if self.phase[0:3] == 'loc':

			#find the onsets
			self.phase_timing.onset = self.phase_meta['stim1.OnsetTime'] - self.phase_start

			#find the durations
			self.phase_timing.duration = 1000
			
			#find the trial types
			self.phase_timing.trial_type = self.phase_meta.stims

			self.phase_timing.trial_type = [self.find_between(cat,'/','/') for cat in self.phase_timing.trial_type]


		else:
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

		self.phase_timing = self.phase_timing.reset_index(drop=True)

		if con:
			return self.phase_timing.trial_type
		else:
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

	def find_between(self, string, first, last):

		try:
			start = string.index(first) + len(first)
			end = string.index(last, start)
			
			found_it = string[start:end]

			if found_it == 'localizer_scenes':
				return 'outdoor'

			else:
				return found_it

		except ValueError:
			return 'string search fucked up'


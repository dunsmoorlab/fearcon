import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from math import sqrt
import seaborn as sns

from fc_config import data_dir, sub_args, pretty_graph
from glm_timing import glm_timing
from preprocess_library import meta

class scr_sort():

	#dictionary of what the first part of the 'Analysis' column is called in autonomate
	auto_dict = { 'extinction_recall': 
					{4.0: 'er_scr_4s',
					 4.5: 'er_scr_45s',
					 5.0: 'er_scr_5s'}
				 }

	p_auto_dict = {}

	def __init__(self, phase, p=False):

		if p:
			#load the ptsd group thing
			pass
		#load in analyzed scr for this phase
		else:
			self.scr_meta = pd.read_csv('%s/Group_SCR/%s/%s_batch_analyzed.csv'%(data_dir,phase,phase))

		#clean up the subject names
		for i, _name in enumerate(self.scr_meta.File):
			self.scr_meta.loc[i,'File'] = self.scr_meta.File[i][:6]

		#go seperate out all the subject results by the right trial length
		for sub in sub_args:
			self.sub_sort(sub, phase)

	def sub_sort(self, sub, phase):

		#load subj meta
		subj = meta(sub)

		#isolate scr for this sub
		sub_scr = self.scr_meta.loc[np.where(self.scr_meta.File == subj.fsub)]
		
		#load in trial events
		sub_events = glm_timing(sub,phase).phase_events()

		#create output strucuture, using index from sub_events
		sub_out = pd.DataFrame([],columns=['Sub','Phase','Trial','CStype','Duration','t2p','Onset','scr_Duration'], index=sub_events.index)

		#fill in some things
		sub_out.Sub = subj.fsub
		sub_out.Phase = phase

		#now loop through the events and find all the fun info
		for i in sub_events.index:
			
			trial = i+1
			_duration = sub_events.duration.loc[i]
			
			sub_out.Duration.loc[i] = _duration
			sub_out.Trial.loc[i] = trial
			sub_out.CStype.loc[i] = sub_events.trial_type.loc[i]

			_trial = sub_scr.loc[sub_scr.Analysis == scr_sort.auto_dict[phase][_duration]].loc[sub_scr.Event == trial]

			sub_out.loc[i,'t2p'] = sub_scr.loc[sub_scr.Analysis == scr_sort.auto_dict[phase][_duration]].loc[sub_scr.Event == trial].t2pValue.values[0]
			sub_out.Onset.loc[i] = sub_scr.loc[sub_scr.Analysis == scr_sort.auto_dict[phase][_duration]].loc[sub_scr.Event == trial].Onset.values[0]
			sub_out.scr_Duration.loc[i] = sub_scr.loc[sub_scr.Analysis == scr_sort.auto_dict[phase][_duration]].loc[sub_scr.Event == trial].Duration.values[0]
		
		for i, scr in enumerate(sub_out.t2p):
			sub_out.loc[i,'t2p'] = sqrt(scr)

		sub_out.to_csv('%s/SCR/%s_analyzed.csv'%(subj.subj_dir,phase))


class scr_stats():

	n_trials = {'extinction_recall': 24}

	def __init__(self, phase):
		
		self.trial_range = range(scr_stats.n_trials[phase])
		self.true_range = list(range(1, self.trial_range[-1] + 2))
		self.phase = phase

		self.collect_sub_scr()
		# self.stats_dict()
		# self.determine_non_responders()

	def collect_sub_scr(self):

		self.sub_dfs = {}

		for sub in sub_args:
			subj = meta(sub)
			self.sub_dfs[sub] = pd.read_csv('%s/SCR/%s_analyzed.csv'%(subj.subj_dir, self.phase))

	def stats_dict(self):

		trial_res = {}

		for trial in self.trial_range:

			trial_res[trial] = {}

			for sub in sub_args:

				trial_res[trial][sub] = self.sub_dfs[sub].t2p[trial]

		trial_df = pd.DataFrame([], index=sub_args, columns=trial_res.keys())

		for _trial in trial_df.columns:
			for sub in sub_args:
				trial_df[_trial].loc[sub] = trial_res[_trial][sub]

		
		self.true_range = list(range(1, self.trial_range[-1] + 2))
		trial_df.set_axis(self.true_range, axis='columns', inplace=True)
		
		#collect the wide format df
		self.scr_wide = trial_df.copy()

		#then make a tall one
		trial_df.reset_index(inplace=True)
		trial_df.rename(columns={'index':'subject'},inplace=True)

		trial_df = trial_df.melt(id_vars='subject')
		trial_df.rename(columns={'variable':'trial','value':'scr'}, inplace=True)

		self.scr_tall = trial_df.copy()


	def determine_non_responders(self):

		self.nr_map = {}

		for trial in self.true_range:

			self.nr_map[trial] = {}

			self.nr_map[trial]['nr'] = np.array(self.scr_wide.index[np.where(self.scr_wide[trial] == 0)[0]])
			self.nr_map[trial]['resp'] = np.array(self.scr_wide.index[np.where(self.scr_wide[trial] != 0)[0]])


#takes in wide format data and a phase string for a title
def scr_boxplot(self, data, phase):
	ax = sns.boxplot(data=data)
	pretty_graph(ax=ax, xlab='Trial #', ylab='square root SCR', main='%s SCR'%(phase))

	




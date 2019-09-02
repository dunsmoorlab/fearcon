import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from math import sqrt
import seaborn as sns

from fc_config import *
from glm_timing import glm_timing
from preprocess_library import meta
from wesanderson import wes_palettes

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

	def __init__(self, phase='extinction_recall'):
		
		self.trial_range = range(scr_stats.n_trials[phase])
		self.true_range = list(range(1, self.trial_range[-1] + 2))
		self.phase = phase

		self.collect_sub_scr()
		# self.stats_dict()
		# self.determine_non_responders()a

	def collect_sub_scr(self):

		self.sub_dfs = {}

		# for sub in sub_args:
		# 	subj = meta(sub)
		# 	self.sub_dfs[sub] = pd.read_csv('%s/SCR/%s_analyzed.csv'%(subj.subj_dir, self.phase))
		
		self.raw = pd.read_csv(os.path.join(data_dir,'Group_SCR','extinction_recall','manual_extinction_recall_scr.csv'))
		self.raw.set_index(['subject'],inplace=True)
		# self.raw = np.sqrt(self.raw)

		self.scr = pd.DataFrame([],index=pd.MultiIndex.from_product(
			[all_sub_args,['CS+','CS-'],range(1,13)],
			names=['subject','condition','trial']),
			columns=['raw','group','half','quarter'])

		for sub in all_sub_args:
			
			for cond in ['CS+','CS-']:

				subj = meta(sub)
				events = glm_timing(sub,'extinction_recall').phase_events(con=True)

				cs_map = np.where(events == cond)[0] + 1

				# cs_cols = [col for col in self.scr.columns if cond in col]
				for i,val in enumerate(cs_map):
					self.scr.loc[(sub,cond,int(i+1)),'raw'] = self.raw.loc[sub,str(cs_map[i])]
					# scr.loc[(sub,cond,int(i+1)),'raw'] = raw.loc[sub,str(cs_map[i])]
		self.scr.raw = self.scr.raw.astype(float)
		# square root transform
		self.scr['scr'] = self.scr.raw.apply(np.sqrt)

		self.scr.loc[sub_args,'group'] = 'control'
		self.scr.loc[p_sub_args,'group'] = 'ptsd'
		
		self.scr['phase'] = 'renewal'
		
		self.out = pd.DataFrame([],index=all_sub_args,columns=['scr','group'])
		self.out.index.name = 'subject'
		for sub in all_sub_args:

			self.out.scr.loc[sub] = self.scr.scr.loc[(sub,'CS+',[1,2,3,4])].mean()

		self.out.group[sub_args] = 'control'
		self.out.group[p_sub_args] = 'ptsd'

		self.out.to_csv(os.path.join(data_dir,'graphing','SCR','rnw_4CSP.csv'))

		self.scr.reset_index(inplace=True)
		self.scr['scr'] = self.scr['scr'].astype(float)

		self.scr = self.scr.set_index('trial')
		self.scr.loc[(1,2,3,4),'half'] = 1
		self.scr.loc[(1,2,3,4),'quarter'] = 5

		self.scr.loc[(5,6,7,8,9,10,11,12),'half'] = 2
		self.scr.loc[(5,6,7,8,9,10,11,12),'quarter'] = 6
		
		self.scr.reset_index(inplace=True)
		self.scr.to_csv(os.path.join(data_dir,'graphing','SCR','day2_all_scr.csv'),index=False)		
	def graph(self):
		sns.set_context('poster')
		sns.set_style('whitegrid',{'axes.facecolor':'.9','figure.facecolor':'.9'})	
		c = sns.catplot(x='trial',y='scr',hue='condition',col='group',data=self.scr,kind='box',
			palette=[wes_palettes['Darjeeling1'][3],wes_palettes['Darjeeling1'][-1]])
		# plt.tight_layout()
		c1 = sns.catplot(x='trial',y='scr',hue='condition',col='group',data=self.scr,kind='point',
			palette=[wes_palettes['Darjeeling1'][3],wes_palettes['Darjeeling1'][-1]])
	
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

	




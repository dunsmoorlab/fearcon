#the are python
import os
import argparse
import numpy as np
import pandas as pd

import sys

# these are all local code
from fc_config import *
from fc_behavioral import recognition_memory
from preprocess_library import meta


class glm_timing(object):

	#this is a hacky line I need because of e-prime, provides the correct starting line number
	first_line = {'baseline': 0, 'fearconditioning': 48, 'extinction': 96, 'extinctionRecall': 384,
					'memory_run_1': 408, 'memory_run_2': 488, 'memory_run_3': 568,
					'localizer_1': 0, 'localizer_2':0 }


	def __init__(self,subj=0,phase=None):

		# if subj == 107 and phase == 'localizer_2': 
		# 	self.sub_exit()

		#initalize directories
		self.subj = meta(subj)

		self.phase = phase
		self.phase_name = phase
		
		#correct for another dumb e-prime thing
		if self.phase == 'fear_conditioning':
			self.phase = 'fearconditioning'
		if self.phase == 'extinction_recall':
			self.phase = 'extinctionRecall'

		self.phase_meta = self.get_phase_meta()

	def sub_exit(self):
		return 'Sub107 no localizer_2'			

	def get_phase_meta(self):
		#isoloate the metadata for this phase
		#handle the case for extinction first
		if self.phase == 'extinction':
			phase_meta = self.ext_meta()
		
		elif self.phase[0:3] == 'mem':
			phase_meta = self.subj.meta[self.subj.meta.ExperimentName == '%s%s%s'%(self.subj.meta.ExperimentName[408][:-4],'Run',int(self.phase[-1]) + 1)]
		
		elif self.phase[0:3] == 'loc':
			phase_meta = pd.read_csv(os.path.join(self.subj.subj_dir,'behavior','run_logs','localizer_%s_meta.csv'%(self.phase[-1])))
			phase_meta = phase_meta[phase_meta.Procedure == 'LocStim']
		
		else:
			phase_meta = self.subj.meta[self.subj.meta.phase == self.phase]
		
		return phase_meta

	def fsl_timing(self,resp=False,pm=False):

		if 'localizer' in self.phase: events = self.loc_blocks()
		if 'memory' in self.phase: events = self.mem_events()
		else: events = self.phase_events(resp=resp) 
		dest = os.path.join(self.subj.model_dir, 'GLM/onsets', py_run_key[self.phase_name])

		if pm:
			events['PM'] = np.nan
			ev = pd.read_csv(os.path.join(data_dir,'Group_GLM','beta_pm_mvpa_ev.csv'))
			sub_ev = ev.loc[np.where(ev.subject == self.subj.num)[0]]
			sub_ev.set_index(['condition','trial'],inplace=True)
			sub_ev.drop(labels='subject',axis=1,inplace=True)
			sub_ev['mc'] = np.nan
			for con in ['CS+','CS-']: sub_ev['mc'][con] = sub_ev['evidence'][con] - sub_ev['evidence'][con].mean()
			tmap = {'CS+':np.where(events.trial_type == 'CS+')[0], 'CS-':np.where(events.trial_type == 'CS-')[0]}
			for con in ['CS+','CS-']:
				for i, trial in enumerate(tmap[con]):
					events['PM'][trial] = sub_ev['mc'][con][i+1]

					whole = events[events.trial_type == con]
					whole.reset_index(inplace=True)
					whole = whole[['onset','duration','PM']]
					resp_string = 'beta_pm_'
					whole.to_csv( os.path.join(dest, '%sall_%s.txt'%(resp_string, con)),
							sep='\t', float_format='%.8e', index=False, header=False)
			return
		
		elif not pm: events['PM'] = 1

		if 'memory' in self.phase:
			conds = events.encode.unique()
			events.drop(['memcond','response'],axis=1,inplace=True)
			for hc in [True,False]:
				events_ = events.copy()
				if hc:
					events_.drop(['acc'],axis=1,inplace=True)
					acc = 'hc_acc'
				elif not hc:
					events_.drop(['hc_acc'],axis=1,inplace=True)
					acc = 'acc'

				mem_conds = events_[acc].unique()
				for mem_con in mem_conds:
					mem_cond_all = events_[events_[acc] == mem_con]
					for con in conds:
						con_all = mem_cond_all[mem_cond_all.encode == con]
						for cs in ['CS+','CS-']:
							con_cs = con_all[con_all.trial_type == cs]
							con_cs.reset_index(inplace=True)
							con_cs = con_cs[['onset','duration','PM']]
							
							if hc:mem_name = 'hc_' + mem_con
							else:mem_name = mem_con
							con_cs.to_csv(os.path.join(dest, '%s_%s_%s.txt'%(mem_name,con,cs)),
									sep='\t', float_format='%.8e', index=False, header=False)


		else: 
			conds = events.trial_type.unique()

			for con in conds:
				whole = events[events.trial_type == con]
				whole.reset_index(inplace=True)
				whole = whole[['onset','duration','PM']]
				
				if resp: resp_string = 'resp_'
				else: resp_string = '' 

				whole.to_csv( os.path.join(dest, '%sall_%s.txt'%(resp_string, con)),
							sep='\t', float_format='%.8e', index=False, header=False)

				# if 'localizer' not in self.phase or 'memory' not in self.phase:
				if 'call' in self.phase:
					# early = whole.loc[:whole.shape[0]/2 - 1]
					# late = whole.loc[whole.shape[0]/2:]
					early = whole.loc[:3]
					late = whole.loc[4:]
				else:
					early = whole.loc[:whole.shape[0]/2 - 1]
					late = whole.loc[whole.shape[0]/2:]

					early.to_csv( os.path.join(dest, '%searly_%s.txt'%(resp_string, con)),
							sep='\t', float_format='%.8e', index=False, header=False)
				
					late.to_csv( os.path.join(dest, '%slate_%s.txt'%(resp_string, con)),
							sep='\t', float_format='%.8e', index=False, header=False)

			#need to add some for extinction recall so we can look at really early



	#create and return a pandas DataFrame for event timing
	#if con=True, then returns only the trial condition type
	def phase_events(self,con=False,mem=False,intrx=False,resp=False,er_start=False,shock=False,stims=False):

		
		#find the start time
		self.phase_start = self.phase_meta['InitialITI.OnsetTime'][self.first_line[self.phase]]

		#create the output structure
		self.phase_timing = pd.DataFrame([],columns=['onset','duration','trial_type'])

		if self.phase[0:3] == 'mem':
			#find the onsets
			self.phase_timing.onset = self.phase_meta['oldnew.OnsetTime'] - self.phase_start

			#find the durations
			self.phase_timing.duration = 3000

			#find the trial types
			self.phase_timing.trial_type = self.phase_meta.cstype


		elif self.phase[0:3] == 'loc':

			#find the onsets
			self.phase_timing.onset = self.phase_meta['stim1.OnsetTime'] - self.phase_start

			#find the durations
			self.phase_timing.duration = 1000
			
			#find the trial types
			self.phase_timing.trial_type = self.phase_meta.stims

			self.phase_timing.trial_type = [self.find_between(cat,'/','/') for cat in self.phase_timing.trial_type]

			#this is literally just here to get rid of the 's' on animals and tools
			dumb_thing = {'animals':'animal','tools':'tool','scrambled':'scrambled','indoor':'indoor','outdoor':'outdoor'}

			self.phase_timing.trial_type = [dumb_thing[label] for label in self.phase_timing.trial_type]

		else:
			#find the onsets
			self.phase_timing.onset = self.phase_meta['stim.OnsetTime'] - self.phase_start

			#find the durations
			if resp:
				self.phase_timing.duration = self.phase_meta['stim.RT']	
			else:
				self.phase_timing.duration = self.phase_meta['stim.Duration']

			#find the trial types
			self.phase_timing.trial_type = self.phase_meta.cstype

			if mem:
				if self.phase == 'fearconditioning':
					day1_phase = 'fear_conditioning'
				else:
					day1_phase = self.phase
				day1_memory = recognition_memory.collect_mem_dat(self, self.subj.num, hch=True, exp_day1=day1_phase)
				day1_memory.index = range(len(day1_memory))

		#convert onset and duration to seconds
		self.phase_timing.onset /= 1000
		self.phase_timing.duration /= 1000

		#make sure that the index is correct (starting at 1 for each phase)
		self.phase_timing = self.phase_timing.reset_index(drop=True)
		if intrx == True:
			self.phase_timing['trial_type'] = self.phase_timing['trial_type'] + '_' + day1_memory
		elif mem:
			self.phase_timing['trial_type'] = day1_memory

		# self.phase_timing.index = list(range(1,self.phase_timing.shape[0] + 1))
		if self.phase == 'extinctionRecall':
			if er_start:
				st = pd.DataFrame({'onset':0,'duration':8,'trial_type':'start'},index=[0])
				self.phase_timing = pd.concat([st,self.phase_timing]).reset_index(drop=True)

		if 'mem' not in self.phase:
			self.proc = self.phase_meta['Procedure[Block]']
			self.proc.index = range(48)

		#make shock timings just straight to fsl format for now
		if self.phase == 'fearconditioning' and shock:

			self.shock = self.phase_timing[self.proc=='CSUS']
			self.shock.onset = self.shock.onset + self.shock.duration
			self.shock.duration = 0
			self.shock['PM'] = 1
			self.shock.drop('trial_type',axis=1,inplace=True)
			self.shock = self.shock[['onset','duration','PM']]
			dest = os.path.join(self.subj.model_dir, 'GLM/onsets', py_run_key[self.phase_name])
			self.shock.to_csv( os.path.join(dest, 'all_shock.txt'),
						sep='\t', float_format='%.8e', index=False, header=False)
			self.shock.index = range(12)
			early = self.shock.loc[:self.shock.shape[0]/2 - 1]
			late = self.shock.loc[self.shock.shape[0]/2:]

			early.to_csv( os.path.join(dest, 'early_shock.txt'),
					sep='\t', float_format='%.8e', index=False, header=False)
		
			late.to_csv( os.path.join(dest, 'late_shock.txt'),
					sep='\t', float_format='%.8e', index=False, header=False)

		if stims:
			self.phase_timing['stim'] = self.phase_meta.stims.values
			self.phase_timing['stim'] = self.phase_timing.stim.str.replace('stims/','')

		if con:
			return self.phase_timing.trial_type
		else:
			return self.phase_timing


	def ext_meta(self):
		
		self.phase3_stims = pd.Series(0)
		self.phase3_unique_loc = pd.Series(0)
		q = 0

		for loc, unique in enumerate(self.subj.meta.stims[self.subj.meta.phase == 'extinction']):
			if not any(stim == unique for stim in self.phase3_stims):
				self.phase3_stims[q] = unique
				self.phase3_unique_loc[q] = loc + 96
				q = q + 1

		return self.subj.meta.loc[self.phase3_unique_loc,]

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

	def betaseries(self,con=False,mem=False,intrx=False,resp=False,er_start=False):

		phase_events = self.phase_events(con=con,mem=mem,intrx=intrx,resp=resp,er_start=er_start)

		np.set_printoptions(precision=8)
			
		phase_events = phase_events[['onset','duration']]

		for trial in phase_events.index:

			vals = phase_events.loc[trial].values

			out = pd.DataFrame([[vals[0],vals[1],1]])

			dest = os.path.join(self.subj.model_dir,'GLM','onsets',
				phase2rundir[self.phase_name][-7:],'betaseries') + os.sep
			
			if not os.path.exists(dest): os.mkdir(dest)

			if er_start: count = trial
			else: count = trial + 1

			# np.savetxt(dest+'trial{0:0=2d}.txt'.format(count), np.transpose(out), fmt='%.8e', delimiter='\t')
			
			out.to_csv(dest+'trial{0:0=2d}.txt'.format(count),sep='\t',
				float_format='%.8e',index=False,header=False)

	def loc_blocks(self, con=False, beta=False):

		if 'localizer' not in self.phase:
			print('this is only for localizer runs, pls stop')
			sys.exit()

		loc_events = self.phase_events()
		
		block_events = pd.DataFrame([],columns=['duration','onset','trial_type'],index=list(range(0,int(len(loc_events.index)/8))))

		block = np.array(range(8))

		#not sure why i made this so complicated since i _could_ hardcode how many blocks there are
		#but basicaly this takes the index of the old loc_events and blocks them by 8
		for i in range(0,int(len(loc_events.index)/8)):

			#block type is type of first stim in block
			block_events.trial_type.loc[i] = loc_events.loc[block].trial_type[block[0]]
			#onset of the block is onset of first stim
			block_events.onset.loc[i] = loc_events.loc[block].onset[block[0]]
			#block duration is the end of the last stim in the block - the onset of the first stim
			block_events.duration.loc[i] = (loc_events.loc[block].onset[block[-1]] + loc_events.loc[block].duration[block[-1]]) - loc_events.loc[block].onset[block[0]]
			block += 8

		#what i am going to do is hardcode some rest blocks
		restloc = np.arange(4,20,5)
		
		rest_events = pd.DataFrame([],columns=['duration','onset','trial_type'],index=range(4))

		rest_events.trial_type = 'rest'
		rest_events.duration = 6
		for rest_trial in rest_events.index:
			rest_events.loc[rest_trial,'onset'] = block_events.loc[restloc[rest_trial]].onset + block_events.loc[restloc[rest_trial]].duration + 3

		block_events = block_events.append(rest_events)
		block_events.sort_values('onset', inplace=True)
		block_events.reset_index(inplace=True)
		block_events.drop('index', axis='columns', inplace=True)

		if con:
			return block_events.trial_type
		
		if beta:

			np.set_printoptions(precision=8)
			
			block_events = block_events[['onset','duration']]

			for trial in block_events.index:

				vals = block_events.loc[trial].values

				out = pd.DataFrame([[vals[0],vals[1],1]])

				dest = os.path.join(self.subj.model_dir,'GLM','onsets',
					phase2rundir[self.phase][-7:],'betaseries') + os.sep
				
				if not os.path.exists(dest): os.mkdir(dest)

				count = trial + 1

				# np.savetxt(dest+'trial{0:0=2d}.txt'.format(count), np.transpose(out), fmt='%.8e', delimiter='\t')
				
				out.to_csv(dest+'trial{0:0=2d}.txt'.format(count),sep='\t',
					float_format='%.8e',index=False,header=False)





		else:
			return block_events


	def mem_events(self,stims=False):
		
		if 'memory' not in self.phase:
			print('this is only for memory runs, pls stop')
			sys.exit()

		mem_events = self.phase_events(stims=stims)
		if stims:
			self.phase_timing['stim'] = self.phase_timing.stim.str.replace('stims2/','')


		phase_index = self.phase_meta.index

		raw_response, memcond, day1_where = recognition_memory.collect_mem_dat(self, self.subj.num, hch=True, exp_res=True)
		
		phase_memcond = memcond[phase_index].copy()
		phase_memcond.index = range(len(phase_memcond))
		mem_events['memcond'] = phase_memcond
		
		phase_where = day1_where[phase_index].copy()
		phase_where.index = range(len(phase_where))
		mem_events['encode'] = phase_where

		phase_response = raw_response[phase_index].copy()
		phase_response.index = range(len(phase_response))
		


		convert_response = pd.Series(np.zeros(len(phase_response)))
		for i, resp in enumerate(phase_response):

			if resp == 1:
				convert_response[i] = 'DN'
			if resp == 2:
				convert_response[i] = 'MN'
			if resp == 3:
				convert_response[i] = 'MO'
			if resp == 4:
				convert_response[i] = 'DO'
			if resp == 0:
				convert_response[i] = 'NR'
		mem_events['response'] = convert_response

		acc = pd.Series(np.zeros(len(convert_response)))
		hc_acc =  pd.Series(np.zeros(len(convert_response)))

		for i, resp in enumerate(convert_response):
			if phase_memcond[i] == 'New':
				if resp == 'DN' or resp == 'MN' or resp == 'NR':
					acc[i] = 'CR'
					hc_acc[i] = 'CR'
				elif resp == 'MO':
					acc[i] = 'FA'
					hc_acc[i] = 'CR'
				elif resp == 'DO':
					acc[i] = 'FA'
					hc_acc[i] = 'FA'
			
			elif phase_memcond[i] == 'Old':
				if resp == 'DN' or resp == 'MN' or resp == 'NR':
					acc[i] = 'M'
					hc_acc[i] = 'M'
				if resp == 'MO':
					acc[i] = 'H'
					hc_acc[i] = 'M'
				if resp == 'DO':
					acc[i] = 'H'
					hc_acc[i] = 'H'
		
		mem_events['acc'] = acc
		mem_events['hc_acc'] = hc_acc

		return mem_events


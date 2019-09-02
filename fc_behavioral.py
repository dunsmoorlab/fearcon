import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys
from scipy import stats
from scipy.stats import ttest_rel, ttest_ind
from fc_config import *
from preprocess_library import meta
from sklearn.metrics import auc

class recognition_memory():


	def __init__(self, p=False, hch=False): 

		if not p:
			# self.sub_args = self.exclude_subs(old=sub_args, exclude=[])
			self.sub_args = self.exclude_subs(old=sub_args, exclude=[18,20])
		elif p:
			self.sub_args = self.exclude_subs(old=working_subs, exclude=[120])
			# self.sub_args = self.exclude_subs(old=p_sub_args, exclude=[])


		self.create_output_structures()

		for sub in self.sub_args:
			# self.collect_roc(sub)
			self.collect_mem_dat(sub, hch=hch)

		self.aucdf.reset_index(inplace=True)
		self.aucdf.auc = self.aucdf.auc.astype(float)

		self.mem_stats()
		#self.vis_group_mem()

	#exclude some subs
	def exclude_subs(self, old=None, exclude=None):

		print('excluding subs %s'%(exclude))

		new_sub_args = np.array(old)

		for sub in exclude:

			new_sub_args = np.delete(new_sub_args, np.where(new_sub_args == sub)[0])

		return list(new_sub_args)


	#create output structures for collecting results
	def create_output_structures(self):

		self.memory_phase = ['baseline','fear_conditioning','extinction','false_alarm']
		self.block_phases = ['baseline','fear_conditioning','extinction']
		cs_condition = ['CS+','CS-']
		self.block_6 = range(1,7,1)

		self.phase_err = pd.DataFrame(index=pd.MultiIndex.from_product(
								[self.memory_phase, cs_condition],
								names=['phase','condition']),
								columns=['cr','err'])

		self.block_cr = pd.DataFrame(index=pd.MultiIndex.from_product(
								[self.sub_args, self.block_phases, self.block_6, cs_condition],
								names=['subject','phase','block','condition']),
								columns=['cr','hit_count','hit_rate'])

		self.f_a = pd.DataFrame(index=pd.MultiIndex.from_product(
								[self.sub_args, ['false_alarm'], cs_condition],
								names=['subject','phase','condition']),
								columns=['cr'])
		
		self.aucdf = pd.DataFrame([],columns=['auc'],
					index=pd.MultiIndex.from_product(
					[self.sub_args,self.block_phases,cs_condition],
					names=['subject','phase','condition']))


		self.block_cr.hit_count = 0
		self.block_cr.sort_index(inplace=True)

	def recog_auc(self,old,new):
		curve=np.zeros([4,2])    
		for i, thr in enumerate([4,3,2,1]):
			throld = [resp for resp in old if resp >= thr]
			thrnew = [resp for resp in new if resp >= thr]
			curve[i,1] = len(throld) / len(old)
			curve[i,0] = len(thrnew) / len(new)

		_auc = auc(curve[:,0],curve[:,1])
		return _auc
	
	def collect_roc(self, sub):
		#this is the number of rows corresponding to just the recognition memroy runs
		phase5 = list(range(408,648))
		#create a series that labels the index of the trials with the name of the phase
		_phase_name = np.array(((['baseline'] * 48) + (['fear_conditioning'] * 48) + (['extinction'] * 48)))
		_block_index = np.tile(np.repeat((1,2,3,4,5,6),8),3)
		day1_index = pd.DataFrame([],index=range(144),columns=['phase','block'])
		day1_index.phase = _phase_name
		day1_index.block = _block_index

		sub_meta = meta(sub).meta
		#collect the raw responses from the recognition test
		respconv = sub_meta['oldnew.RESP'][phase5]
		#fill in any non-responses with 0
		respconv = respconv.fillna(0)
		#collect the old/new attribute for each stim
		memcond = sub_meta.MemCond[phase5]
		#collect the CS type for each stim
		condition = sub_meta.cstype

		#collect stims from baseline and fear conditioning
		phase1_2_stims = sub_meta.stims[0:96]
		phase1_stims = phase1_2_stims[0:48]
		phase2_stims = phase1_2_stims[48:96]
		
		#do the same for extinction, but also undo the dumb row expansion that e-prime does with the scene ITIs
		phase3_stims = pd.Series(0)
		phase3_unique_loc = pd.Series(0)
		q = 0

		#this paragraph goes through all the stims and extinciton
		#and collects unique stim names in their experimental order
		for loc, unique in enumerate(sub_meta.stims[sub_meta.phase == 'extinction']):
			if not any(stim == unique for stim in phase3_stims):
				phase3_stims[q] = unique
				phase3_unique_loc[q] = loc + 96
				q = q + 1

		#give it the correct index numbering
		phase3_stims.index = list(range(96,144))
		
		#concatenate all stims from day1
		day1_stims = pd.Series(np.zeros(144))
		day1_stims[0:96] = phase1_2_stims
		day1_stims[96:144] = phase3_stims

		#collect the stims from day2, phase5 (much easier)
		day2_stims = sub_meta.stims[phase5]

		#get rid of 'stims/' and 'stims2/' in both so they can be compared
		day1_stims = day1_stims.str.replace('stims/','')
		day2_stims = day2_stims.str.replace('stims2/','')

		#lastely, make a variable for all of the day1 conditions, counter-acting how eprime fucks up extinciton
		day1_condition = pd.Series(np.zeros(144))
		day1_condition[0:96] = condition[0:96]
		day1_condition[96:144] = condition[phase3_unique_loc]

		day1df = pd.DataFrame({'stim':day1_stims.values,
							   'phase':day1_index.phase.values})

		day2df = pd.DataFrame({'response':respconv.values,
							   'memcond':memcond.values,
							   'stim':day2_stims.values,
							   'condition':condition[phase5].values,
							   'day1_phase':''})

		day2df = day2df.drop(np.where(day2df.response == 0)[0]).reset_index(drop=True)
		day2df.loc[np.where(day2df.memcond == 'New')[0],'day1_phase'] = 'foil'
		for i in day2df.index:
			if day2df.loc[i,'memcond'] == 'Old':
				day2df.loc[i,'day1_phase'] = day1df.phase[day1df.stim == day2df.loc[i,'stim']].values[0]

		for phase in ['baseline','fear_conditioning','extinction']:
				for cond in ['CS+','CS-']:
					Old = day2df.response[day2df.day1_phase == phase][day2df.condition == cond].values
					New = day2df.response[day2df.day1_phase == 'foil'][day2df.condition == cond].values
					self.aucdf.loc[(sub,phase,cond),'auc'] = self.recog_auc(old=Old,new=New)



	def collect_mem_dat(self, sub, hch=True, exp_res=False, exp_day1=False):

		#this is the number of rows corresponding to just the recognition memroy runs
		phase5 = list(range(408,648))
		#create a series that labels the index of the trials with the name of the phase
		_phase_name = np.array(((['baseline'] * 48) + (['fear_conditioning'] * 48) + (['extinction'] * 48)))
		_block_index = np.tile(np.repeat((1,2,3,4,5,6),8),3)
		day1_index = pd.DataFrame([],index=range(144),columns=['phase','block'])
		day1_index.phase = _phase_name
		day1_index.block = _block_index

		#load in subjects meta data file (e-prime log)
		sub_meta = meta(sub).meta
		#collect the raw responses from the recognition test
		respconv = sub_meta['oldnew.RESP'][phase5]
		#fill in any non-responses with 0
		respconv = respconv.fillna(0)
		#collect the old/new attribute for each stim
		memcond = sub_meta.MemCond[phase5]
		#collect the CS type for each stim
		condition = sub_meta.cstype

		#collect stims from baseline and fear conditioning
		phase1_2_stims = sub_meta.stims[0:96]
		phase1_stims = phase1_2_stims[0:48]
		phase2_stims = phase1_2_stims[48:96]
		
		#do the same for extinction, but also undo the dumb row expansion that e-prime does with the scene ITIs
		phase3_stims = pd.Series(0)
		phase3_unique_loc = pd.Series(0)
		q = 0

		#this paragraph goes through all the stims and extinciton
		#and collects unique stim names in their experimental order
		for loc, unique in enumerate(sub_meta.stims[sub_meta.phase == 'extinction']):
			if not any(stim == unique for stim in phase3_stims):
				phase3_stims[q] = unique
				phase3_unique_loc[q] = loc + 96
				q = q + 1

		#give it the correct index numbering
		phase3_stims.index = list(range(96,144))
		
		#concatenate all stims from day1
		day1_stims = pd.Series(np.zeros(144))
		day1_stims[0:96] = phase1_2_stims
		day1_stims[96:144] = phase3_stims

		#collect the stims from day2, phase5 (much easier)
		day2_stims = sub_meta.stims[phase5]

		#get rid of 'stims/' and 'stims2/' in both so they can be compared
		day1_stims = day1_stims.str.replace('stims/','')
		day2_stims = day2_stims.str.replace('stims2/','')

		#lastely, make a variable for all of the day1 conditions, counter-acting how eprime fucks up extinciton
		day1_condition = pd.Series(np.zeros(144))
		day1_condition[0:96] = condition[0:96]
		day1_condition[96:144] = condition[phase3_unique_loc]

		#now lets look at their responses
		#set up some variables to translate raw responses into meaning
		correct_rejection = np.zeros(0)
		false_alarm = np.zeros(0)
		miss = np.zeros(0)
		hit = np.zeros(0)
		print('%s has %s non-responses'%(sub, len(np.where(respconv == 0)[0])))
		
		#convert raw response into meaning
		#right now this isn't built to seperate out confidence, this is where that would have to happen
		for i in day2_stims.index:
			
			if not hch:

				#if its new, has to be either CR or FA
				if memcond[i] == 'New':
					#non-responses get counted as correct rejection if its new
					if respconv[i] == 1 or respconv[i] == 2 or respconv[i] == 0:
						correct_rejection = np.append(correct_rejection, respconv.index[i-408])
					
					elif respconv[i] == 3 or respconv[i] == 4:
						false_alarm = np.append(false_alarm, respconv.index[i-408])

				elif memcond[i] == 'Old':
				#non-responses get counted as misses if they're old
					if respconv[i] == 1 or respconv[i] == 2 or respconv[i] == 0:
						miss = np.append(miss, respconv.index[i-408])
					
					elif respconv[i] == 3 or respconv[i] == 4:
						hit = np.append(hit, respconv.index[i-408])
		

			if hch:
			
				#if its new, has to be either CR or FA
				if memcond[i] == 'New':
					#non-responses get counted as correct rejection if its new
					if respconv[i] == 1 or respconv[i] == 2 or respconv[i] == 0 or respconv[i] == 3:
						correct_rejection = np.append(correct_rejection, respconv.index[i-408])
					
					elif respconv[i] == 4:
						false_alarm = np.append(false_alarm, respconv.index[i-408])


				elif memcond[i] == 'Old':
				
					if respconv[i] == 1 or respconv[i] == 2 or respconv[i] == 0 or respconv[i] == 3:
						miss = np.append(miss, respconv.index[i-408])
				
					elif respconv[i] == 4:
						hit = np.append(hit, respconv.index[i-408])

		if exp_res:
			day1_where = pd.Series(np.zeros(len(day2_stims)))
			day1_where.index = day2_stims.index
			for i in day2_stims.index:
				stim = day2_stims[i]
				where = np.where(day1_stims == stim)[0]
				if len(where) == 0:
					day1_where[i] = 'foil'
				else:
					day1_where[i] = day1_index.loc[where,'phase'].values[0]
			return respconv, memcond, day1_where
			sys.exit()

		#count up old and new 
		#(I could hardcode these values since they shouldn't change, but I'm paranoid and this is my check_sum)
		old = len(memcond[memcond == 'Old'])
		new = len(memcond[memcond == 'New'])

		#lets break it up into phase
		#since false alarm rate isn't calculate phase by phase, all we need are the location of the hits on day1
		hit_index = []
		[[hit_index.append(i) for i, stim in enumerate(day1_stims) if stim == day2_stims[target]] for target in hit]

		if exp_day1:
			if exp_day1 == 'baseline':
				comp_stims = phase1_stims
			elif exp_day1 == 'fear_conditioning':
				comp_stims = phase2_stims
			elif exp_day1 == 'extinction':
				comp_stims = phase3_stims
		
			day1_memory=pd.Series(np.zeros(len(comp_stims)))
			for i, stim in enumerate(comp_stims.index):
				if stim in hit_index:
					day1_memory[i] = 'hit'
				else:
					day1_memory[i] = 'miss'

			return day1_memory
			sys.exit()


		# calculate overall false alarm rate by condition
		CSplus_false_alarm_rate = len(condition[false_alarm][condition == 'CS+']) / (new /2)
		CSmin_false_alarm_rate = len(condition[false_alarm][condition == 'CS-']) / (new /2)

		self.f_a['cr'][sub]['false_alarm']['CS+'] = CSplus_false_alarm_rate
		self.f_a['cr'][sub]['false_alarm']['CS-'] = CSmin_false_alarm_rate


		#loop through all the hits and tally them up block by block
		for stim in hit_index:
			self.block_cr['hit_count'][sub][day1_index['phase'][stim]][day1_index['block'][stim]][day1_condition[stim]] += 1

		for phase in self.block_phases:
			for block in self.block_6:
				self.block_cr['cr'][sub][phase][block]['CS+'] = ( (self.block_cr['hit_count'][sub][phase][block]['CS+'] / 4) - CSplus_false_alarm_rate)
				self.block_cr['cr'][sub][phase][block]['CS-'] = ( (self.block_cr['hit_count'][sub][phase][block]['CS-'] / 4) - CSmin_false_alarm_rate)
				self.block_cr['hit_rate'][sub][phase][block]['CS+'] = (self.block_cr['hit_count'][sub][phase][block]['CS+'] / 4)
				self.block_cr['hit_rate'][sub][phase][block]['CS-'] = (self.block_cr['hit_count'][sub][phase][block]['CS-'] / 4)

	#combine subject results into group results
	def mem_stats(self):
		#reset first and then melt them and then see if you cant get all 4 phases in the same graph

		self.phase_cr = self.block_cr.unstack(level=2)
		self.phase_cr = self.phase_cr['cr'].mean(axis=1)
		self.phase_cr = self.phase_cr.reset_index()
		self.phase_cr.rename(columns={0:'cr'}, inplace=True)

		self.phase_hr = self.block_cr.unstack(level=2)
		self.phase_hr = self.phase_hr['hit_rate'].mean(axis=1)
		self.phase_hr = self.phase_hr.reset_index()
		self.phase_hr.rename(columns={0:'hit_rate'}, inplace=True)


		self._f_a = self.f_a.copy()
		self._f_a.reset_index(inplace=True)
		self.phase_cr = self.phase_cr.append(self._f_a)

		self.hr_fa = self._f_a.copy()
		self.hr_fa.rename(columns={'cr':'hit_rate'}, inplace=True)
		self.phase_hr = self.phase_hr.append(self.hr_fa)

		_phases = pd.Categorical(self.phase_cr['phase'],
					categories=self.memory_phase, ordered=True)
		self.phase_cr['phase'] = _phases
		self.phase_cr.sort_values(['phase','subject'], inplace=True)
		
		_phases = pd.Categorical(self.phase_hr['phase'],
					categories=self.memory_phase, ordered=True)
		self.phase_hr['phase'] = _phases
		self.phase_hr.sort_values(['phase','subject'], inplace=True)


		_err_ = self.block_cr.unstack(level=(0,-2))
		self.f_a = self.f_a.unstack(level=0)

		for phase in self.memory_phase:
			if phase == 'false_alarm':
				self.phase_err['cr'][phase] = self.f_a['cr'].mean(axis=1)
				self.phase_err['err'][phase] = self.f_a['cr'].sem(axis=1)
			else:
				self.phase_err['cr'][phase] = _err_['cr'].loc[phase].mean(axis=1)
				self.phase_err['err'][phase] = _err_['cr'].loc[phase].sem(axis=1)

		self.phase_err.reset_index(inplace=True)

		self.phase_stats = self.phase_cr.set_index(['phase','condition'])

		self.phase_t = pd.DataFrame(index=self.memory_phase, columns=['tstat','pval'])

		for phase in self.memory_phase:
			
			self.phase_t['tstat'][phase], self.phase_t['pval'][phase] = ttest_rel(self.phase_stats['cr'][phase]['CS+'], self.phase_stats['cr'][phase]['CS-'])

		print(self.phase_t)

	def vis_group_mem(self,title=None):

		fig, pp = plt.subplots()
		pp = sns.pointplot(data=self.phase_cr, x='phase', y='cr',
							hue='condition', kind='point',
							palette='husl',
							dodge=True, join=False)
		pretty_graph(ax=pp, xlab='Phase', ylab='Corrected Recognition', main='Corrected Recognition by Phase')
		# fig.savefig('%s/%s_bootstrap_CR'%(data_dir + 'graphing' + os.sep + 'behavior', title))


		fig2, bx = plt.subplots()
		self.phase_cr['cr'] = self.phase_cr['cr'].astype(np.float)
		bx = sns.boxplot(data=self.phase_cr, x='phase', y='cr',
						hue='condition', palette='husl')
		bx = sns.stripplot(data=self.phase_cr, x='phase', y='cr',
						hue='subject', palette='husl')
		pretty_graph(ax=bx, xlab='Phase', ylab='Corrected Recognition', main='Corrected Recognition by Phase')
		# fig2.savefig('%s/%s_CR_boxplot'%(data_dir + 'graphing' + os.sep + 'behavior', title))


		fig3, sw = plt.subplots()
		sw = sns.factorplot(data=self.phase_cr, x='condition', y='cr',
							col='phase', hue='subject',
							kind='box', palette='hls')

		st = sns.factorplot(data=self.phase_cr, x='condition', y='cr',
							col='phase', hue='subject',
							kind='strip', palette='hls')
		# add annotations one by one with a loop
		for line in range(0,self.phase_cr.shape[0]):
			st.text(self.phase_cr.condition[line]+0.2, self.phase_cr.cr[line], self.phase_cr.subejct[line], horizontalalignment='left', size='medium', color='black', weight='semibold')
		# plt.savefig('%s/%s_CR_swarmplot'%(data_dir + 'graphing' + os.sep + 'behavior', title))
	
		sns.set_style('whitegrid')
		sns.set_style('ticks')
		# sns.set_style(rc={'axes.linewidth':'5'})
		# plt.rcParams['xtick.labelsize'] = 22 
		# plt.rcParams['ytick.labelsize'] = 22

		
		fig, ax = plt.subplots()
		ind = np.arange(3)    # the x locations for the groups
		width = 0.4         # the width of the bars
		
		#for now dont plot false alarm
		self.phase_err_ = self.phase_err[:6]
		csp = self.phase_err_.loc[np.where(self.phase_err_['condition'] == 'CS+')[0]]
		csm = self.phase_err_.loc[np.where(self.phase_err_['condition'] == 'CS-')[0]]

		p1 = ax.bar(ind, csp['cr'], width, yerr=(csp['err']), color=plt.cm.Set1.colors[0], alpha=.8)
		p2 = ax.bar(ind+width, csm['cr'], width, yerr=(csm['err']), color=plt.cm.Set1.colors[1], alpha=.8)
		ax.set_xticks(ind + width / 2)
		ax.set_xticklabels(self.memory_phase)
		ax.set_ylim([0,.7])
		# pretty_graph(ax=ax, xlab='Phase', ylab='Corrected Recognition', main='CR by Phase with SEM', legend=True)
		# ax.legend((p1[0], p2[0]), ('CS+', 'CS-'), fontsize='larger')
		fig.set_size_inches(9, 5.5)
		plt.tight_layout()
		plt.savefig(os.path.join(data_dir,'graphing', 'cns', 'hch_CR_mem.png'))

		#ToDO - graph block results


class shock_expectancy():

	def __init__(self,p=False):

		if p == True:
			self.sub_args = p_sub_args
			self.ptsd = True
		elif p == False:
			self.sub_args = sub_args
			self.ptsd = False
		if p == 'all':
			self.sub_args = all_sub_args
			self.ptsd = False

		self.create_output_structures()
		
		for sub in self.sub_args:
			self.collect_expectancy(sub)

		self.exp_stats()
		# self.vis_phase_exp()

	def create_output_structures(self):
		self.cs_condition = ['CS+','CS-']

		self.exp_phases = ['fear_conditioning','extinction','extinction_recall']

		self.phase_exp = {}
		for sub in self.sub_args:
			self.phase_exp[sub] = {}
			for phase in self.exp_phases:
				self.phase_exp[sub][phase] = {}
				if phase == 'extinction_recall':
					for con in self.cs_condition:
						self.phase_exp[sub][phase][con] = {}
						for i in range(1,13):
							self.phase_exp[sub][phase][con][i] = {}
				else:
					for con in self.cs_condition:
						self.phase_exp[sub][phase][con] = {}
						for i in range(1,25):
							self.phase_exp[sub][phase][con][i] = {}

		self.prop_exp = {}
		for phase in self.exp_phases:
			self.prop_exp[phase] = {}
			if phase == 'extinction_recall':
				for con in self.cs_condition:
					self.prop_exp[phase][con] = {}
					for i in range(1,13):
						self.prop_exp[phase][con][i] = {}
			else:
				for con in self.cs_condition:
					self.prop_exp[phase][con] = {}
					for i in range(1,25):
						self.prop_exp[phase][con][i] = {}

	def collect_expectancy(self, sub):

		sub_meta = meta(sub).meta

		#collect index for fear and ER, and use the paragraph for finding extinction
		phase2_loc = np.where(sub_meta.phase == 'fearconditioning')[0]
		phase4_loc = np.where(sub_meta.phase == 'extinctionRecall')[0]


		#this paragraph goes through all the stims and extinciton
		#and collects unique stim names in their experimental order
		phase3_stims = pd.Series(0)
		phase3_loc = pd.Series(0)
		q = 0
		for loc, unique in enumerate(sub_meta.stims[sub_meta.phase == 'extinction']):
			if not any(stim == unique for stim in phase3_stims):
				phase3_stims[q] = unique
				phase3_loc[q] = loc + 96
				q = q + 1
		phase3_loc = np.array(phase3_loc)

		_phase2 = {'resp':sub_meta['stim.RESP'][phase2_loc], 'condition': sub_meta['cstype'][phase2_loc], 'cs_trial': sub_meta['cstypebytrial'][phase2_loc]}
		phase2 = pd.DataFrame(_phase2)
		phase2.index = range(1,49)
		phase2.fillna(0,inplace=True)

		_phase3 = {'resp':sub_meta['stim.RESP'][phase3_loc], 'condition': sub_meta['cstype'][phase3_loc], 'cs_trial': sub_meta['cstypebytrial'][phase3_loc]}
		phase3 = pd.DataFrame(_phase3)
		phase3.index = range(1,49)
		phase3.fillna(0,inplace=True)

		_phase4 = {'resp':sub_meta['stim.RESP'][phase4_loc], 'condition': sub_meta['cstype'][phase4_loc], 'cs_trial': sub_meta['cstypebytrial'][phase4_loc]}
		phase4 = pd.DataFrame(_phase4)
		phase4.index = range(1,25)
		phase4.fillna(0,inplace=True)

		non_responses = len(np.where(phase2['resp'] == 0)[0]) + len(np.where(phase3['resp'] == 0)[0]) + len(np.where(phase4['resp'] == 0)[0])
		print('%s has %s non-responses'%(sub, non_responses))


		for phase in self.exp_phases:
			
			if phase == 'fear_conditioning':
				for r in phase2.index:
					self.phase_exp[sub][phase][phase2['condition'][r]][int(phase2['cs_trial'][r][-2:])] = {}
					if phase2['resp'][r] == 1:
						self.phase_exp[sub][phase][phase2['condition'][r]][int(phase2['cs_trial'][r][-2:])]['exp'] = 1
					
					elif phase2['resp'][r] == 2 or phase2['resp'][r] == 0:
						self.phase_exp[sub][phase][phase2['condition'][r]][int(phase2['cs_trial'][r][-2:])]['exp'] = 0
			
			if phase == 'extinction':
				for r in phase3.index:
					self.phase_exp[sub][phase][phase3['condition'][r]][int(phase3['cs_trial'][r][-2:])] = {}
					if phase3['resp'][r] == 1:
						self.phase_exp[sub][phase][phase3['condition'][r]][int(phase3['cs_trial'][r][-2:])]['exp'] = 1
					
					elif phase3['resp'][r] == 2 or phase3['resp'][r] == 0:
						self.phase_exp[sub][phase][phase3['condition'][r]][int(phase3['cs_trial'][r][-2:])]['exp'] = 0
			
			if phase == 'extinction_recall':
				for r in phase4.index:
					self.phase_exp[sub][phase][phase4['condition'][r]][int(phase4['cs_trial'][r][-2:])] = {}
					if phase4['resp'][r] == 1:
						self.phase_exp[sub][phase][phase4['condition'][r]][int(phase4['cs_trial'][r][-2:])]['exp'] = 1
					
					elif phase4['resp'][r] == 2 or phase4['resp'][r] == 0:
						self.phase_exp[sub][phase][phase4['condition'][r]][int(phase4['cs_trial'][r][-2:])]['exp'] = 0



	def exp_stats(self):

		self.exp_df = pd.DataFrame.from_dict({(sub, phase, con, trial): self.phase_exp[sub][phase][con][trial]
												for sub in self.phase_exp.keys()
												for phase in self.phase_exp[sub].keys()
												for con in self.phase_exp[sub][phase].keys()
												for trial in self.phase_exp[sub][phase][con].keys()},
												orient='index')
		

		for sub in self.sub_args:
			for phase in self.exp_phases:
				for con in self.cs_condition:
					_hold = np.zeros(self.exp_df['exp'][sub][phase][con].shape[0])
					for i in self.exp_df['exp'][sub][phase][con].index:
						_hold[i-1] = (self.exp_df['exp'][sub][phase][con][:i].sum() / self.exp_df['exp'][sub][phase][con][:i].shape[0])
					for trial, val in enumerate(_hold):
						self.phase_exp[sub][phase][con][trial+1]['cavg'] = val
		
		self.exp_df = pd.DataFrame.from_dict({(sub, phase, con, trial): self.phase_exp[sub][phase][con][trial]
												for sub in self.phase_exp.keys()
												for phase in self.phase_exp[sub].keys()
												for con in self.phase_exp[sub][phase].keys()
												for trial in self.phase_exp[sub][phase][con].keys()},
												orient='index')
		


		self.exp_df.reset_index(inplace=True)
		self.exp_df.rename(columns={'level_0':'subject', 'level_1':'phase', 'level_2':'condition', 'level_3':'trial'}, inplace=True)

		_phases = pd.Categorical(self.exp_df['phase'],
					categories=self.exp_phases, ordered=True)

		self.exp_df['phase'] = _phases
		self.exp_df.sort_values(['phase','subject'], inplace=True)
		self.exp_df.reset_index(inplace=True,drop=True)
		# self.exp_df.drop('index')

		self.exp_df.to_csv(os.path.join(data_dir,'graphing','behavior','expectancy.csv'))

		self.prop_df = self.exp_df.copy()
		#self.prop_df = self.prop_df.drop(columns=['subject'])
		self.prop_df.set_index(['phase','condition','trial'], inplace=True)

		for phase in self.prop_exp:
			for condition in self.prop_exp[phase]:
				for trial in self.prop_exp[phase][condition]:
					self.prop_exp[phase][condition][trial]['avg'] = self.prop_df['exp'][phase][condition][trial].mean()
					self.prop_exp[phase][condition][trial]['err'] = self.prop_df['exp'][phase][condition][trial].sem()

		self.prop_df = pd.DataFrame.from_dict({(phase, con, trial): self.prop_exp[phase][con][trial]
										for phase in self.prop_exp.keys()
										for con in self.prop_exp[phase].keys()
										for trial in self.prop_exp[phase][con].keys()},
										orient='index')
		self.prop_df.reset_index(inplace=True)
		self.prop_df.rename(columns={'level_0':'phase', 'level_1':'condition', 'level_2':'trial'}, inplace=True)

		_phases = pd.Categorical(self.prop_df['phase'],
					categories=self.exp_phases, ordered=True)

		self.prop_df['phase'] = _phases
		self.prop_df.sort_values(['phase','condition','trial'], inplace=True)
		self.prop_df.set_index(['phase','condition','trial'],inplace=True)


	def vis_phase_exp(self):

		# fig, ax = plt.subplots(nrows=1, ncols=3)

		# ax = sns.factorplot(data=self.exp_df, x='trial', y='exp',
		# 			col='phase', hue='condition', kind='point', ci=95, palette='husl')
		# plt.savefig('%s/cavg_shoch_exp'%(data_dir + 'graphing' + os.sep + 'behavior'))

		# ax2 = sns.factorplot(data=self.prop_df, x='trial', y='avg',
		# 		col='phase', hue='condition', kind='point', ci=None, palette='husl')
		# plt.savefig('%s/prop_shoch_exp'%(data_dir + 'graphing' + os.sep + 'behavior'))
		sns.set_style(rc={'axes.linewidth':'5'})
		plt.rcParams['xtick.labelsize'] = 22 
		plt.rcParams['ytick.labelsize'] = 22

		fig, (ax1, ax2) = plt.subplots(1,2,sharey=False)
		fig3, ax3 = plt.subplots()

		for cond, color in zip(['CS+','CS-'], [plt.cm.Set1.colors[0], plt.cm.Set1.colors[1]]):
			ax1.plot(range(1,25), self.prop_df['avg'].loc['fear_conditioning'].loc[cond], marker='o', color=color, markersize=8)

			# ax1.fill_between(range(1,25), self.prop_df['avg'].loc['fear_conditioning'].loc[cond] - self.prop_df['err'].loc['fear_conditioning'].loc[cond],
			# 							self.prop_df['avg'].loc['fear_conditioning'].loc[cond] + self.prop_df['err'].loc['fear_conditioning'].loc[cond],
			# 			label='%s'%(cond), color=color, alpha=.5)
			ax1.set_xticks(np.arange(4,25,4))
			ax2.plot(range(1,25), self.prop_df['avg'].loc['extinction'].loc[cond], marker='o', color=color, markersize=8)
			# ax2.fill_between(range(1,25), self.prop_df['avg'].loc['extinction'].loc[cond] - self.prop_df['err'].loc['extinction'].loc[cond],
			# 							self.prop_df['avg'].loc['extinction'].loc[cond] + self.prop_df['err'].loc['extinction'].loc[cond],
			# 			label='%s'%(cond), color=color, alpha=.5)
			ax2.set_xticks(np.arange(4,25,4))


			ax3.plot(range(1,13), self.prop_df['avg'].loc['extinction_recall'].loc[cond], marker='o', color=color, markersize=8)
			# ax3.fill_between(range(1,13), self.prop_df['avg'].loc['extinction_recall'].loc[cond] - self.prop_df['err'].loc['extinction_recall'].loc[cond],
			# 							self.prop_df['avg'].loc['extinction_recall'].loc[cond] + self.prop_df['err'].loc['extinction_recall'].loc[cond],
			# 			label='%s'%(cond), color=color, alpha=.5)
			ax3.set_xticks(np.arange(2,13,2))
			ax3.set_yticks(np.arange(.2,1.2,.2)) 

		ax1.set_ylim([0,1])
		ax2.set_ylim([0,1])
		ax3.set_ylim([0,1])
		ax1.set_xlim([.5,24.5])
		ax2.set_xlim([.5,24.5])
		ax3.set_xlim([.5,12.5])

		# ax1.set_title('Fear Conditioning',size='xx-large')
		# ax2.set_title('Extinction',size='xx-large')
		# ax3.set_title('Extinction Recall',size='xx-large')

		# ax1.legend(loc='upper right', fontsize='larger')
		# ax2.legend(loc='upper right', fontsize='larger')
		# ax3.legend(loc='upper right', fontsize='larger')

		# ax1.set_ylabel('Proportion of Subjects Expecting a Shock',size = 'x-large')

		# ax1.set_xlabel('Trial',size = 'x-large')
		# ax2.set_xlabel('Trial',size = 'x-large')
		# ax3.set_xlabel('Trial',size = 'x-large')

		fig.set_size_inches(10, 4)
		fig3.set_size_inches(6, 3)
		plt.tight_layout()
		fig.savefig(os.path.join(data_dir,'graphing','cns','day1_shock_expectancy_prop.png'), dpi=300)
		fig3.savefig(os.path.join(data_dir,'graphing','cns','day2_shock_expectancy_prop.png'), dpi=300)

import os
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns

from fc_config import *
from fc_decoding import loc_decode
from preprocess_library import meta
from glm_timing import glm_timing

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from nilearn.input_data import NiftiMasker

from itertools import cycle
from scipy.stats import sem

from scr_analysis import scr_stats
from fc_behavioral import shock_expectancy

class decode():

	decode_runs = ['baseline','fear_conditioning','extinction','extinction_recall','memory_run_1','memory_run_2','memory_run_3']

	def __init__(self, subj=0, imgs=None, k=0, save_dict=mvpa_masked_prepped):

		self.subj = meta(subj)
		print('Decoding %s'%(self.subj.fsub))

		self.loc_dat, self.loc_labels = self.load_localizer(imgs=imgs)

		self.test_dat = self.load_test_dat(runs=decode.decode_runs, save_dict=save_dict)

		self.init_fit_clf(k=k, data=self.loc_dat, labels=self.loc_labels)

		self.clf_res = self.decode_test_data(test_data=self.test_dat)


	def load_localizer(self, imgs=None):

		loc_dat = loc_decode.load_localizer(self, imgs)

		loc_labels = loc_decode.better_tr_labels(self, data=loc_dat, imgs=imgs)
		
		if 'tr' in imgs:
			#shift 3TRs (labels are already shifted in better_tr_labels)
			loc_dat = loc_decode.tr_shift(self, data=loc_dat)

		loc_dat = np.concatenate([loc_dat['localizer_1'], loc_dat['localizer_2']])

		if 'tr' in imgs:
			loc_labels, loc_dat = loc_decode.snr_hack(self, labels=loc_labels, data=loc_dat)

		'''
		#was removing rest at first but we need it to make sense of the deocding data
		if 'tr' in imgs:
			loc_labels, loc_dat = loc_decode.remove_rest(self, labels=loc_labels, data=loc_dat)
		'''
		
		# loc_labels, loc_dat = loc_decode.remove_indoor(self, labels=loc_labels, data=loc_dat)

		loc_labels, loc_dat = loc_decode.downsample_scenes(self, labels=loc_labels, data=loc_dat)

		loc_labels = loc_decode.collapse_scenes(self, labels=loc_labels)

		return loc_dat, loc_labels


	def load_test_dat(self, runs, save_dict):

		dat = { phase: np.load('%s%s'%(self.subj.bold_dir,save_dict[phase]))[dataz] for phase in runs }

		return dat


	def init_fit_clf(self, k=0, data=None, labels=None):

		print('Initializing Classifier')
		self.clf = Pipeline([ ('anova', SelectKBest(f_classif, k=k)), ('clf', LogisticRegression()) ])

		print('Fitting Classifier to localizer')
		self.clf.fit(data, labels)


	def decode_test_data(self, test_data=None):

		proba_res = { phase: self.clf.decision_function(test_data[phase]) for phase in test_data }

		#this is straight from sklearn.linear_model.base.py
		"""Probability estimation for OvR logistic regression.
        Positive class probabilities are computed as
        1. / (1. + np.exp(-self.decision_function(X)));
        multiclass is handled by normalizing that over all classes.
        """
        #the only thing different here is that i am NOT normalizing over all classes
		for phase in proba_res:
			prob = proba_res[phase]
			prob *= -1
			np.exp(prob, prob)
			prob += 1
			np.reciprocal(prob, prob)
			proba_res[phase] = prob


		self.clf_lab = list(self.clf.classes_)

		_res = {phase:{} for phase in test_data}

		for phase in _res:
			
			for i, label in enumerate(self.clf_lab):

				_res[phase][label] = proba_res[phase][:,i]

		labels = list(self.clf.classes_)
		nlab = list(range(0,len(labels)))

		return _res


class group_decode():

	conds = ['csplus','csminus','scene','scrambled','rest']

	def __init__(self, imgs=None, k=0, save_dict=mvpa_masked_prepped):

		print('hi hello')
		self.sub_decode(imgs=imgs, k=k, save_dict=save_dict)
		self.event_res = self.event_results(self.group_results)

	def sub_decode(self, imgs='tr', k=0, save_dict=mvpa_masked_prepped):
		
		group_res = {}

		for sub in sub_args:

			sub_dec = decode(sub, imgs=imgs, k=k, save_dict=save_dict)
			sub_res = sub_dec.clf_res

			group_res[sub] = {}

			for phase in decode.decode_runs:

				group_res[sub][phase] = {}				
			
				group_res[sub][phase] = { label: sub_res[phase][label] for label in sub_dec.clf_lab }

		#make it a self here so it can be referenced later
		self.group_results = group_res

		'''
		this nifty little line creates a dataframe from the nested dict group_res
		the outer index is phase, and the inner index is subject number
		the columns are the labels (scene,animal,etc.)		
		'''
		group_df = pd.DataFrame.from_dict({(phase, sub): group_res[sub][phase]
												for sub in group_res.keys()
												for phase in group_res[sub].keys() }, orient='index')

		#now build a dict that goes dict[phase][condition]:mean,sem
		group_stats = { phase:{} for phase in decode.decode_runs }

		for phase in group_stats:

			group_stats[phase] = { cond:{} for cond in group_decode.conds }

			for cond in group_stats[phase]:
				#animals/tools get sorted here
				if cond == 'csplus':
					group_stats[phase][cond]['ev'] = group_df[sub_dec.subj.csplus][phase].mean()
					#maybe need to compute error here
				elif cond == 'csminus':
					group_stats[phase][cond]['ev'] = group_df[sub_dec.subj.csminus][phase].mean()
					#maybe need to compute error here
				
				#scenes/scrambled/rest here
				else:
					group_stats[phase][cond]['ev'] = group_df[cond][phase].mean()
					#maybe need to compute error here

		self.group_stats = group_stats 

		#make second dict that computes mean and stuff over entire phase

	def event_results(self, sub_res):
		
		event_res = {}	
		
		for sub in sub_res.keys():
			subj = meta(sub)

			event_res[sub] = {}

			#calculate the event related scene evidence for each phase
			for phase in sub_res[sub].keys():

				event_res[sub][phase] = {}

				events = glm_timing(sub, phase).phase_events()

				events['start_tr'], events['start_rem'] = divmod(events.onset, 2)
				events['end_tr'], events['end_rem'] = divmod(events.onset+events.duration, 2)


				#apply hdr shift to the trs because we actually want the events...
				events.start_tr += hdr_shift['start']
				events.end_tr += hdr_shift['start']

				for trial in events.index:

					#for use to get rid of pythonic indexing
					_trial = trial + 1

					event_res[sub][phase][_trial] = {}

					if events.start_rem[trial] <= 1.5:
						
						start = events.start_tr[trial]

					elif events.start_rem[trial] > 1.5:

						start = events.start_tr[trial] + 1				

					end = events.end_tr[trial]


					#right now we are looking 1 TR before the stim comes on through 1 TR after it ends
					#range() is not inclusive on the upper bound hence the +2
					window = range(int(events.start_tr[trial])-3, int(events.end_tr[trial])+2)
					#print(sub, phase, trial, window)
					#have to fix the window if it ever tries to grab a TR that isn't actually there
					#this only happens once im pretty sure
					if window[-1] >= sub_res[sub][phase]['scene'].shape[0]:
						print('fixing window')
						window = range(window[0], sub_res[sub][phase]['scene'].shape[0])


					for cond in group_decode.conds:
						if cond == 'csplus':
							stim_cond = subj.csplus

						elif cond == 'csminus':
							stim_cond = subj.csminus

						else:
							stim_cond = cond

						event_res[sub][phase][_trial][cond] = sub_res[sub][phase][stim_cond][window]
		
		event_df = pd.DataFrame.from_dict( {(phase, cond, trial, sub): event_res[sub][phase][trial][cond]
											for sub in event_res.keys()
											for phase in event_res[sub].keys()
											for trial in event_res[sub][phase].keys()
											for cond in event_res[sub][phase][trial].keys()}, orient='index')

		event_df.index = pd.MultiIndex.from_tuples(event_df.index, names=('phase','cond','trial','sub'))

		event_df.columns = [-3,-2,-1,0,1,2,3,4]

		#not all the trials have the final value so we'll drop it
		event_df.drop(event_df.columns[-1],axis=1,inplace=True)

		self.event_df = event_df


		event_stats = {}

		for phase in decode.decode_runs:

			event_stats[phase] = {}

			for cond in group_decode.conds:

				event_stats[phase][cond] = {}

				event_stats[phase][cond]['avg'] = event_df.loc[phase].loc[cond].mean(axis=0)
				event_stats[phase][cond]['err'] = event_df.loc[phase].loc[cond].sem(axis=0)

		self.event_stats = event_stats
		return event_stats

	def exp_event(self, phase='extinction_recall'):

		print('loading shock_expectancy behavioral data')
		exp_bhv = shock_expectancy().phase_exp

		phase_df = self.event_df.loc[phase]
		phase_df = phase_df.reorder_levels(['sub','trial','cond'])

		exp_ev = {'expect':{}, 'no':{}}

		for sub in sub_args:

			#get the phase conditions and correct for pythonic indexing
			pc = glm_timing(sub, phase).phase_events(con=True)
			pc.index = range(1,pc.shape[0]+1)

			csplus_map = (np.where(pc=='CS+')[0] + 1)

			first_csp = csplus_map[0]

			if exp_bhv[sub][phase]['CS+'][1]['exp'] == 1:

				exp_ev['expect'][sub] = phase_df.loc[sub].loc[first_csp]

			elif exp_bhv[sub][phase]['CS+'][1]['exp'] == 0:

				exp_ev['no'][sub] = phase_df.loc[sub].loc[first_csp]

		print('n_expect = %s; n_no = %s'%(len(exp_ev['expect'].keys()), len(exp_ev['no'].keys())))

		self.exp_ev_df = pd.DataFrame.from_dict({(response, sub, tr): exp_ev[response][sub][tr]
											for response in exp_ev.keys()
											for sub in exp_ev[response].keys()
											for tr in exp_ev[response][sub].keys()},
											orient='index')
		
		self.ev_ = self.exp_ev_df.groupby(level=[0,2]).mean()
		self.ev_.reset_index(inplace=True)
		self.ev_ = self.ev_.melt(id_vars=['level_0','level_1'])
		self.ev_.rename(columns={'level_0':'response', 'level_1':'tr', 'variable':'condition', 'value':'ev'}, inplace=True)
		
		self.err_ = self.exp_ev_df.groupby(level=[0,2]).sem()
		self.err_.reset_index(inplace=True)
		self.err_ = self.err_.melt(id_vars=['level_0','level_1'])
		self.err_.rename(columns={'level_0':'response', 'level_1':'tr', 'variable':'condition', 'value':'err'}, inplace=True)

		self.err_['ev'] = self.ev_['ev']
		self.err_.set_index(['response','condition','tr'],inplace=True)

		fig, ax1 = plt.subplots()
		for cond in ['csplus','rest','scene']:
			ax1.errorbar([-3,-2,-1,0,1,2,3], self.err_['ev'].loc['expect'].loc[cond], yerr=(self.err_['err'].loc['expect'].loc[cond]), label='%s'%(cond))
		fig, ax2 = plt.subplots()
		for cond in ['csplus','rest','scene']:
			ax2.errorbar([-3,-2,-1,0,1,2,3], self.err_['ev'].loc['no'].loc[cond], yerr=(self.err_['err'].loc['no'].loc[cond]), label='%s'%(cond))

		self.exp_ev_df.reset_index(inplace=True)
		self.exp_ev_df.rename(columns={'level_0':'response', 'level_1':'subject', 'level_2':'tr'}, inplace=True)
		self.exp_ev_df = self.exp_ev_df.melt(id_vars=['response','subject','tr'])
		self.exp_ev_df.rename(columns={'variable':'condition', 'value':'evidence'}, inplace=True)

		
		ax = sns.factorplot(data=self.exp_ev_df, x='tr', y='evidence',
							hue='condition', col='response',
							kind='point',dodge=True)
		plt.savefig('%s/NEW_BETA_ER_trial_evidence_with_expectancy'%(data_dir + 'graphing' + os.sep + 'mvpa_analysis'))

	def memory_events(self):
		memory_phases = ['memory_run_1','memory_run_2','memory_run_3']
		
		enc_ev = {}

		for phase in memory_phases:
			
			phase_df = self.event_df.loc[phase]
			phase_df = phase_df.reorder_levels(['sub','trial','cond'])

			enc_ev[phase] = {}

			for sub in sub_args:

				if sub == 18 or sub == 20:
					pass
				else:
				
					enc_ev[phase][sub] = {'baseline':{}, 'fear_conditioning':{}, 'extinction':{}, 'foil':{}}
					#get the phase conditions and correct for pythonic indexing
					pc = glm_timing(sub, phase).mem_events()
					pc.index = range(1,pc.shape[0]+1)

					#have to correct index again
					base_map = (np.where(pc['encode'] == 'extinction')[0] + 1)
					fear_map = (np.where(pc['encode'] == 'extinction')[0] + 1)
					ext_map = (np.where(pc['encode'] == 'extinction')[0] + 1)
					foil_map = (np.where(pc['encode'] == 'extinction')[0] + 1)

					for i, trial in enumerate(base_map):
						enc_ev[phase][sub]['baseline'][i] = phase_df.loc[sub].loc[trial].reset_index().melt(id_vars='cond').set_index(['cond','variable']).reorder_levels(['variable','cond'])
					for i, trial in enumerate(base_map):
						enc_ev[phase][sub]['fear_conditioning'][i] = phase_df.loc[sub].loc[trial].reset_index().melt(id_vars='cond').set_index(['cond','variable']).reorder_levels(['variable','cond'])
					for i, trial in enumerate(base_map):
						enc_ev[phase][sub]['extinction'][i] = phase_df.loc[sub].loc[trial].reset_index().melt(id_vars='cond').set_index(['cond','variable']).reorder_levels(['variable','cond'])
					for i, trial in enumerate(base_map):
						enc_ev[phase][sub]['foil'][i] = phase_df.loc[sub].loc[trial].reset_index().melt(id_vars='cond').set_index(['cond','variable']).reorder_levels(['variable','cond'])
		self.enc_ev = enc_ev
		self.mem_ev_df = pd.DataFrame.from_dict({(phase, sub, encode, trial): enc_ev[phase][sub][encode][trial]
											for phase in enc_ev.keys()
											for sub in enc_ev[phase].keys()
											for encode in enc_ev[phase][sub].keys()
											for trial in enc_ev[phase][sub][encode].keys()},
											orient='index')
		
		self.mem_ev_df.reset_index(inplace=True)
		hold = self.mem_ev_df['index'].apply(pd.Series)
		hold.rename(columns={0:'phase',1:'subject',2:'encode',3:'trial'}, inplace=True)

		self.mem_ev_df = self.mem_ev_df.drop(columns='index')
		self.mem_ev_df = pd.concat([hold, self.mem_ev_df])
		self.mem_ev_df.set_index(['phase','subject','encode','trial'], inplace=True)
		# self.mem_ev_df.rename(columns={'level_0':'phase', 'level_1':'subject', 'level_2':'trial'}, inplace=True)
		# # self.mem_ev_df = self.mem_ev_df.melt(id_vars=['phase','subject','trial'])
		# # # self.mem_ev_df.rename(columns={'variable':'condition', 'value':'evidence'}, inplace=True)

		
		# ax = sns.factorplot(data=self.exp_ev_df, x='tr', y='evidence',
		# 					hue='condition', col='response',
		# 					kind='point',dodge=True)
		# plt.savefig('%s/NEW_BETA_ER_trial_evidence_with_expectancy'%(data_dir + 'graphing' + os.sep + 'mvpa_analysis'))



	def vis_cond_phase(self, phase=None, title=None):
		
		results=self.group_stats

		n_classes = len(group_decode.conds)

		index = range(0, len(results[phase]['scene']['ev']))


		plt.figure()
		colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green'])
		for cond, color in zip(results[phase].keys(), colors):
			plt.plot(index, results[phase][cond]['ev'][index], color=color, lw=2,
				label='%s'%(cond))
		plt.legend()
		plt.title(phase + '; ' + title)
		plt.xlabel('TR')
		plt.ylabel('classifier evidence')
		plt.savefig('%s/%s_%s'%(data_dir + 'graphing' + os.sep + 'mvpa_analysis', phase, title))


	def phase_bar_plot(self, title=None):
		
		results=self.group_stats

		stats = self.get_bar_stats(results)

		sns.barplot(x='phase',y='avg',hue='cond',data=stats)
		
		plt.title(title)


	def get_bar_stats(self, results):

		stats = {}

		for phase in results:
			
			stats[phase] = {}


			for cond in results[phase]:
				stats[phase][cond] = {}
				stats[phase][cond]['avg'] = np.mean(results[phase][cond]['ev'])
				stats[phase][cond]['err'] = sem(results[phase][cond]['ev'])
				

		stats_df = pd.DataFrame.from_dict({(phase, cond): stats[phase][cond]
													for phase in stats.keys()
													for cond in stats[phase].keys() }, orient='index')
		
		stats_df.index.rename(('phase','cond'),inplace=True)

		stats_df.reset_index(inplace=True)


		return stats_df



	def vis_phase(self, cond='scene',index=range(0,50), title=None):

		results=self.group_stats

		# index = list(range(0,phase.extinction_recall.index.shape[0]))

		# fig, ax = plt.subplots()
		n_classes = len(group_decode.conds)

		plt.figure()
		colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red'])
		for phase, color in zip(results.keys(), colors):
	   		plt.plot(index, results[phase][cond]['ev'][index], color=color, lw=2,
	   			label='%s'%(phase))
		plt.legend()
		plt.title(title)
		plt.xlabel('TR')
		plt.ylabel('classifier evidence for %s'%(cond))




	def vis_event_res(self,phase=None,title=None):

		results=self.event_res

		n_classes = len(group_decode.conds)

		index = list(range(-1,4))

		plt.figure()
		colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green'])
		for cond, color in zip(results[phase].keys(), colors):
			plt.errorbar(index, results[phase][cond]['avg'][index], yerr=results[phase][cond]['err'][index], color=color, lw=2, 
				label='%s'%(cond))
		plt.legend()
		plt.xticks([-1,0,1,2,3])
		plt.title(phase + '; ' + title)
		plt.xlabel('TR')
		plt.ylabel('classifier evidence')
		plt.savefig('%s/%s_%s'%(data_dir + 'graphing' + os.sep + 'mvpa_analysis' + os.sep + 'event', phase, title))

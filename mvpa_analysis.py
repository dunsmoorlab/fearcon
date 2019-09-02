import os
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

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
from scipy.stats import sem, ttest_rel, ttest_ind, t

from scr_analysis import scr_stats
from fc_behavioral import shock_expectancy

class decode():

	# decode_runs = ['baseline','fear_conditioning','extinction','extinction_recall']#,'memory_run_1','memory_run_2','memory_run_3']
	decode_runs = ['extinction_recall']

	def __init__(self, subj=0, imgs=None, k=0, save_dict=mvpa_masked_prepped, binarize=False, SC=True, S_DS=True, rmv_rest=False, rmv_scram=True, rmv_ind=False, verbose=False):

		self.verbose = verbose
		self.subj = meta(subj)
		if self.verbose: print('Decoding %s'%(self.subj.fsub))

		self.loc_dat, self.loc_labels = self.load_localizer(imgs=imgs, save_dict=save_dict, binarize=binarize, rmv_rest=rmv_rest, SC=SC, S_DS=S_DS, rmv_scram=rmv_scram, rmv_ind=rmv_ind)

		self.test_dat = self.load_test_dat(runs=decode.decode_runs, save_dict=save_dict)

		if self.verbose: print(self.loc_dat.shape, self.loc_labels.shape)
		self.init_fit_clf(k=k, data=self.loc_dat, labels=self.loc_labels)

		self.clf_res = self.decode_test_data(test_data=self.test_dat,binarize=binarize)


	def load_localizer(self, imgs=None, save_dict=None, stim_binarize=None, binarize=None, SC=None, S_DS=None, rmv_rest=None, rmv_scram=None, rmv_ind=None):

		loc_dat = loc_decode.load_localizer(self, imgs, save_dict)
		if self.verbose: print(loc_dat['localizer_1'].shape)

		loc_labels = loc_decode.better_tr_labels(self, data=loc_dat, imgs=imgs)
		
		if 'tr' in imgs:
			#shift 2TRs (labels are already shifted in better_tr_labels)
			loc_dat = loc_decode.tr_shift(self, data=loc_dat)

		
		if self.subj.num == 107: loc_dat = loc_dat['localizer_1']
		else: loc_dat = np.concatenate([loc_dat['localizer_1'], loc_dat['localizer_2']])
		
		if self.verbose: print(loc_dat.shape)
		if 'tr' in imgs:
			loc_labels, loc_dat = loc_decode.snr_hack(self, labels=loc_labels, data=loc_dat)

		
		#was removing rest at first but we need it to make sense of the deocding data
		if rmv_rest:
			loc_labels, loc_dat = loc_decode.remove_rest(self, labels=loc_labels, data=loc_dat)
	

		#downsample scenes
		if S_DS:
			loc_labels, loc_dat = loc_decode.downsample_scenes(self, labels=loc_labels, data=loc_dat)

		#collapse indoor & outdoor scenes
		if SC:
			loc_labels = loc_decode.collapse_scenes(self, labels=loc_labels)
		
		if rmv_ind:
			loc_labels, loc_dat = loc_decode.remove_indoor(self, labels=loc_labels, data=loc_dat)

		#remove scrambled scenes
		if rmv_scram:
			loc_labels, loc_dat = loc_decode.remove_scrambled(self, labels=loc_labels, data=loc_dat)

		if stim_binarize:
			loc_labels, loc_dat = loc_decode.stim_binarize(self,labels=loc_labels, data=loc_dat)

		if binarize:
			loc_labels, loc_dat = loc_decode.binarize(self,labels=loc_labels, data=loc_dat)
		return loc_dat, loc_labels


	def load_test_dat(self, runs, save_dict):

		dat = { phase: np.load('%s%s'%(self.subj.bold_dir,save_dict[phase]))[dataz] for phase in runs }
		for run in dat:
			if self.verbose: print(dat[run].shape)
		return dat


	def init_fit_clf(self, k=0, data=None, labels=None):

		if self.verbose: print('Initializing Classifier')
		self.clf = Pipeline([ ('anova', SelectKBest(f_classif, k=k)), ('clf', LogisticRegression() ) ])

		if self.verbose: print('Fitting Classifier to localizer')
		self.clf.fit(data, labels)


	def decode_test_data(self, test_data=None, binarize=False):

		_res = {phase:{} for phase in test_data}
		self.clf_lab = list(self.clf.classes_)
		print(list(self.clf.classes_))

		if binarize:
			print('BINARIZING')
			for i, v in enumerate(self.clf.classes_):
				if v == 'scene':scene_loc=i
				if v == 'scrambled':scrambled_loc=i

			for phase in _res:
				_res[phase]['scene'] = self.clf.predict_proba(test_data[phase])[:,scene_loc]
				_res[phase]['scrambled'] = self.clf.predict_proba(test_data[phase])[:,scrambled_loc]
		else:
			
			#this is straight from sklearn.linear_model.base.py
			"""Probability estimation for OvR logistic regression.
	        Positive class probabilities are computed as
	        1. / (1. + np.exp(-self.decision_function(X)));
	        multiclass is handled by normalizing that over all classes.
	        """
	        #the only thing different here is that i am NOT normalizing over all classes
			proba_res = { phase: self.clf.decision_function(test_data[phase]) for phase in test_data }
			for phase in proba_res:
				prob = proba_res[phase]
				prob *= -1
				np.exp(prob, prob)
				prob += 1
				np.reciprocal(prob, prob)
				proba_res[phase] = prob

			#this is normalized
			# proba_res = { phase: self.clf.predict_proba(test_data[phase]) for phase in test_data }

			for phase in _res:
			
				for i, label in enumerate(self.clf_lab):
					if label == 'scene': print(i,label,proba_res[phase][0:4,i])
					_res[phase][label] = proba_res[phase][:,i]

		return _res


class group_decode():

	def __init__(self, conds=None, imgs=None, k=0, save_dict=mvpa_masked_prepped, binarize=False, rmv_rest=False,SC=True, S_DS=True, rmv_ind=False, rmv_scram=True, p=False, verbose=False):
		self.save_dict = save_dict

		self.verbose = verbose
		if binarize:
			self.group_decode_conds = ['scene','scrambled']
		# elif not binarize and rmv_scram:
		# 	self.group_decode_conds = ['csplus','csminus','scene','rest']
		# elif rmv_ind:
		# 	self.group_decode_conds = ['scene','rest']
		# elif not rmv_scram:
		# 	self.group_decode_conds = ['csplus','csminus','scene','rest','scrambled']
		# else:
		# 	self.group_decode_conds = ['scene','rest','scrambled']
		# self.group_decode_conds = ['scene','scrambled','stim','rest']
		# self.group_decode_conds = ['scene','scrambled','stim']
		# self.group_decode_conds = ['csplus','csminus','scene','rest']
		# self.group_decode_conds = ['animal', 'indoor', 'scene', 'scrambled', 'tool','rest']
		# self.group_decode_conds = ['stim', 'indoor', 'scene', 'scrambled']
		else:
			self.group_decode_conds = conds

		if binarize:
			self.comp_cond = 'scrambled'
		elif rmv_rest: 
			self.comp_cond = 'stim'
		elif 'rest' not in self.group_decode_conds:
			self.comp_cond = 'scrambled'
		elif 'scrambled' not in self.group_decode_conds:
			self.comp_cond = 'rest'
		else:
			self.comp_cond = 'rest'

		if p == True:
			self.sub_args = p_sub_args
			self.ptsd = True
		elif p == False:
			self.sub_args = sub_args
			self.ptsd = False
		if p == 'all':
			self.sub_args = all_sub_args
			self.ptsd = False
		print(self.sub_args)
		if self.verbose: print('hi hello')
		self.sub_decode(imgs=imgs, k=k, save_dict=save_dict, binarize=binarize,rmv_rest=rmv_rest, rmv_ind=rmv_ind, SC=SC, S_DS=S_DS, rmv_scram=rmv_scram, verbose=verbose)
		if save_dict == beta_ppa_prepped or save_dict == hippocampus_beta or save_dict == hipp_no_ppa_beta or save_dict == beta_nz_ppa_prepped: 
			self.event_res = self.beta_event_results(self.group_results)
			print('did the beta thing')
		else:
			self.event_res = self.event_results(self.group_results)
		

		if self.verbose: print('loading shock_expectancy behavioral data')
		if p: shock_str = 'ptsd'
		elif not p: shock_str = 'control'
		self.exp_bhv = pickle.load(open(data_dir + 'graphing/behavior/' + shock_str + '_shock_expectancy.p', 'rb'))

	def sub_decode(self, imgs='tr', k=0, save_dict=mvpa_masked_prepped, binarize=False,rmv_rest=False, SC=True, S_DS=True, rmv_scram=True, rmv_ind=False, verbose=False):
		
		group_res = {}

		for sub in self.sub_args:

			sub_dec = decode(sub, imgs=imgs, k=k, save_dict=save_dict, binarize=binarize,rmv_rest=rmv_rest, SC=SC, S_DS=S_DS, rmv_scram=rmv_scram, rmv_ind=rmv_ind, verbose=verbose)
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
		self.group_df = group_df
		#now build a dict that goes dict[phase][condition]:mean,sem
		group_stats = { phase:{} for phase in decode.decode_runs }

		for phase in group_stats:

			group_stats[phase] = { cond:{} for cond in self.group_decode_conds }

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
	def beta_event_results(self, sub_res):

		event_res = {}

		for sub in sub_res.keys():
			subj = meta(sub)

			event_res[sub] = {}

			for phase in sub_res[sub].keys():

				event_res[sub][phase] = {}

				events = glm_timing(sub, phase).phase_events(con=True)

				for trial in events.index:
					_trial = trial + 1

					event_res[sub][phase][_trial] = {}

					for cond in self.group_decode_conds:
						if cond == 'csplus':
							stim_cond = subj.csplus

						elif cond == 'csminus':
							stim_cond = subj.csminus

						else:
							stim_cond = cond

						event_res[sub][phase][_trial][cond] = sub_res[sub][phase][stim_cond][trial]
		
		event_df = pd.DataFrame.from_dict( {(phase, cond, trial, sub): event_res[sub][phase][trial][cond]
											for sub in event_res.keys()
											for phase in event_res[sub].keys()
											for trial in event_res[sub][phase].keys()
											for cond in event_res[sub][phase][trial].keys()}, orient='index')

		event_df.index = pd.MultiIndex.from_tuples(event_df.index, names=('phase','cond','trial','sub'))

		self.event_df = event_df

		event_stats = {}

		for phase in decode.decode_runs:

			event_stats[phase] = {}

			for cond in self.group_decode_conds:

				event_stats[phase][cond] = {}

				event_stats[phase][cond]['avg'] = event_df.loc[phase].loc[cond].mean(axis=0)
				event_stats[phase][cond]['err'] = event_df.loc[phase].loc[cond].sem(axis=0)

		self.event_stats = event_stats
		return event_stats



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
				events.start_tr += 2
				events.end_tr += 2

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
					window = range(int(events.start_tr[trial])-2, int(events.end_tr[trial])+2)
					#print(sub, phase, trial, window)
					#have to fix the window if it ever tries to grab a TR that isn't actually there
					#this only happens once im pretty sure
					if window[-1] >= sub_res[sub][phase]['scene'].shape[0]:
						if self.verbose: print('fixing window')
						window = range(window[0], sub_res[sub][phase]['scene'].shape[0])

					for cond in self.group_decode_conds:
					# for cond in conds:
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

		event_df.columns = [-2,-1,0,1,2,3,4]

		#not all the trials have the final value so we'll drop it
		event_df.drop(event_df.columns[-1],axis=1,inplace=True)

		self.event_df = event_df


		event_stats = {}

		for phase in decode.decode_runs:

			event_stats[phase] = {}

			for cond in self.group_decode_conds:

				event_stats[phase][cond] = {}

				event_stats[phase][cond]['avg'] = event_df.loc[phase].loc[cond].mean(axis=0)
				event_stats[phase][cond]['err'] = event_df.loc[phase].loc[cond].sem(axis=0)

		self.event_stats = event_stats
		return event_stats


	def exp_event(self, phase='extinction_recall', con='CS+', split=False):

		print(con)

		self.exp_split = []

		exp_ev = {'expect':{}, 'no':{}}
		self.nexp = {'expect':{}, 'no':{}}

		trial_range = range(1,5)

		for trial in trial_range:
			phase_df = self.event_df.loc[phase]
			phase_df = phase_df.reorder_levels(['sub','trial','cond'])

			exp_ev['expect'][trial] = {}
			exp_ev['no'][trial] = {}

			for sub in self.sub_args:

				#get the phase conditions and correct for pythonic indexing
				pc = glm_timing(sub, phase).phase_events(con=True)
				pc.index = range(1,pc.shape[0]+1)

				csplus_map = (np.where(pc==con)[0] + 1)

				csp_trial = csplus_map[trial - 1]

				if self.exp_bhv[sub][phase][con][trial]['exp'] == 1:

					exp_ev['expect'][trial][sub] = phase_df.loc[sub].loc[csp_trial]

				elif self.exp_bhv[sub][phase][con][trial]['exp'] == 0:

					exp_ev['no'][trial][sub] = phase_df.loc[sub].loc[csp_trial]

					if trial == 1:
						self.exp_split = np.append(self.exp_split, sub)
			
			self.nexp['no'][trial] = len(exp_ev['no'][trial].keys())
			self.nexp['expect'][trial] = len(exp_ev['expect'][trial].keys())
			
			if trial == 1:
				self.no_group = list(exp_ev['no'][trial].keys())
				self.yes_group = list(exp_ev['expect'][trial].keys())

			if self.verbose: print('%s trial %s: n_expect = %s; n_no = %s'%(con, trial, len(exp_ev['expect'][trial].keys()), len(exp_ev['no'][trial].keys())))

		self.exp_ev_df = pd.DataFrame.from_dict({(response, trial, sub, tr): exp_ev[response][trial][sub][tr]
											for response in exp_ev.keys()
											for trial in exp_ev[response].keys()
											for sub in exp_ev[response][trial].keys()
											for tr in exp_ev[response][trial][sub].keys()},
											orient='index')
		self.big_split = self.exp_ev_df.copy()

		self.ev_ = self.exp_ev_df.groupby(level=[0,1,3]).mean()
		self.ev_.reset_index(inplace=True)
		self.ev_ = self.ev_.melt(id_vars=['level_0','level_1','level_2'])
		self.ev_.rename(columns={'level_0':'response', 'level_1':'trial', 'level_2':'tr', 'variable':'condition', 'value':'ev'}, inplace=True)
		
		self.err_ = self.exp_ev_df.groupby(level=[0,1,3]).sem()
		self.err_.reset_index(inplace=True)
		self.err_ = self.err_.melt(id_vars=['level_0','level_1','level_2'])
		self.err_.rename(columns={'level_0':'response', 'level_1':'trial', 'level_2':'tr', 'variable':'condition', 'value':'err'}, inplace=True)

		self.err_['ev'] = self.ev_['ev']
		self.err_.set_index(['response','trial','condition','tr'],inplace=True)

		self.exp_ev_df.reset_index(inplace=True)
		self.exp_ev_df.rename(columns={'level_0':'response', 'level_1':'trial', 'level_2':'subject', 'level_3':'tr'}, inplace=True)
		self.exp_ev_df = self.exp_ev_df.melt(id_vars=['response','trial','subject','tr'])
		self.exp_ev_df.rename(columns={'variable':'condition', 'value':'evidence'}, inplace=True)

		if split:

			self.big_split.reset_index(inplace=True)
			self.big_split['split'] = 'expect'

			for sub in self.exp_split:
				self.big_split['split'].loc[np.where(self.big_split['level_2'] == sub)[0]] = 'no'
			
			self.big_split.drop(columns=['level_0'],inplace=True)
			self.big_split.rename(columns={'split':'response','level_1':'trial', 'level_2':'subject', 'level_3':'tr'}, inplace=True)
			self.big_split.set_index(['response','trial','subject','tr'],inplace=True)
			
			self.split_ev = self.big_split.groupby(level=[0,1,3]).mean()
			self.split_ev.reset_index(inplace=True)
			self.split_ev = self.split_ev.melt(id_vars=['response','trial','tr'])
			self.split_ev.rename(columns={'variable':'condition', 'value':'ev'}, inplace=True)

			self.split_err = self.big_split.groupby(level=[0,1,3]).sem()
			self.split_err.reset_index(inplace=True)
			self.split_err = self.split_err.melt(id_vars=['response','trial','tr'])
			self.split_err.rename(columns={'variable':'condition', 'value':'err'}, inplace=True)

			self.split_err['ev'] = self.split_ev['ev']
			self.split_err.set_index(['response','trial','condition','tr'],inplace=True)

			self.err_ = self.split_err.copy()
			self.exp_ev_df = self.big_split.copy()
			self.exp_ev_df.reset_index(inplace=True)
			self.exp_ev_df = self.exp_ev_df.melt(id_vars=['response','trial','subject','tr'])
			self.exp_ev_df.rename(columns={'variable':'condition', 'value':'evidence'}, inplace=True)
			# ax = sns.factorplot(data=self.exp_ev_df, x='tr', y='evidence',
			# 					hue='condition', col='response', row='trial',
			# 					kind='point',dodge=True)
	#	plt.savefig('%s/exp_ev_no_scrambled'%(data_dir + 'graphing' + os.sep + 'mvpa_analysis' + os.sep + 'cns'))
	
	def vis_exp_event(self):
		
		# sns.set_style('whitegrid')
		# sns.set_style('ticks')
		# sns.set_style(rc={'axes.linewidth':'2'})
		# plt.rcParams['xtick.labelsize'] = 10 
		# plt.rcParams['ytick.labelsize'] = 10
		# plt.rcParams['axes.labelsize'] = 10
		# plt.rcParams['axes.titlesize'] = 12
		# plt.rcParams['legend.labelspacing'] = .25

		trials = np.unique(self.err_.index.get_level_values('trial'))
		nrows = len(trials)
		
		xaxis_tr = [-2,-1,0,1,2,3]

		fig, ax = plt.subplots(nrows,2)
		for trial in range(nrows):
			for cond, color in zip(['csplus','scene','rest'], ['red','purple','gray','green']):
			# for cond, color in zip(['scene',self.comp_cond], ['mediumblue',plt.cm.Set1.colors[-1]]):	
				if cond == 'csplus':
					labelcond = 'CS+'
				else:
					labelcond = cond

				ax[trial][0].plot(xaxis_tr, self.err_['ev'].loc['no'].loc[trials[trial]].loc[cond], color=color, marker='o', markersize=5, label=labelcond)
				print(np.mean(self.err_['ev'].loc['no'].loc[trials[trial]].loc[cond]))
				ax[trial][0].fill_between(xaxis_tr, self.err_['ev'].loc['no'].loc[trials[trial]].loc[cond] - self.err_['err'].loc['no'].loc[trials[trial]].loc[cond],
				 self.err_['ev'].loc['no'].loc[trials[trial]].loc[cond] + self.err_['err'].loc['no'].loc[trials[trial]].loc[cond],
				 alpha=.5, color=color)

				ax[trial][1].plot(xaxis_tr, self.err_['ev'].loc['expect'].loc[trials[trial]].loc[cond], label='%s'%(labelcond), color=color, marker='o', markersize=5)
				
				ax[trial][1].fill_between(xaxis_tr, self.err_['ev'].loc['expect'].loc[trials[trial]].loc[cond] - self.err_['err'].loc['expect'].loc[trials[trial]].loc[cond],
				 self.err_['ev'].loc['expect'].loc[trials[trial]].loc[cond] + self.err_['err'].loc['expect'].loc[trials[trial]].loc[cond],
				 alpha=.5, color=color)
			
			ax[trial][0].set_title('CS+ Trial %s; Did not expect a shock (N=%s)'%(trials[trial], self.nexp['no'][trials[trial]]))
			ax[trial][1].set_title('CS+ Trial %s; Expected a shock (N=%s)'%(trials[trial], self.nexp['expect'][trials[trial]]))

		ax[1][0].set_xlabel('TR (away from stimulus onset)')
		ax[1][1].set_xlabel('TR (away from stimulus onset)')
		
		ax[0][0].set_ylabel('Classifier Evidence')
		ax[1][0].set_ylabel('Classifier Evidence')
		# plt.gca().invert_yaxis()
		# plt.gca().invert_yaxis()

		# ax[0][0].plot(1,.2,marker='$*$',markersize=10, color='black')
		# ax[1][0].plot(0,.5,marker='$*$',markersize=10, color='black')
		# fig.set_size_inches(10, 6)

		# ax[0][1].legend(loc='upper center', bbox_to_anchor=(0.5, 1), frameon=True, ncol=3,borderaxespad=1)#,prop={'size': 8})
		# ax[0][1].legend(bbox_to_anchor=(0, .02, 1., 1), frameon=True, loc=3, ncol=3,borderaxespad=0,prop={'size': 8})
		# ax[0][0].legend(loc='upper center', bbox_to_anchor=(0.5, 1), frameon=True, ncol=3,borderaxespad=1)#,prop={'size': 8})

		# fig.savefig(os.path.join(data_dir,'graphing','quals','2trial_exp_tr.png'), dpi=500)

		# for _ax in ax.flat:
		# 	ax.label_outer()

			# ax2.set_title('Expected a shock, N=8',size='xx-large')
			# ax1.legend(['CS+','Scene','Rest'],loc='lower left', fontsize='larger')
			# ax1.set_ylabel('Classifier Evidence',size = 'x-large')
			# ax2.set_ylabel('Classifier Evidence',size = 'x-large')
			# ax1.set_xlabel('TR (away from stimulus onset)',size = 'x-large')
			# ax2.set_xlabel('TR (away from stimulus onset)',size = 'x-large')		
		

		# fig, ax2 = plt.subplots(nrows,2)
		# for trial in range(nrows):
		# 	for cond, color in zip(['csplus','scene','rest'], [plt.cm.Set1.colors[0],plt.cm.Set1.colors[3],plt.cm.Set1.colors[-1],'green']):
				
		# 		ax2[trial][0].plot(xaxis_tr, self.split_err['ev'].loc['no_1'].loc[trials[trial]].loc[cond], color=color, marker='o', markersize=8)
				
		# 		ax2[trial][0].fill_between(xaxis_tr, self.split_err['ev'].loc['no_1'].loc[trials[trial]].loc[cond] - self.split_err['err'].loc['no_1'].loc[trials[trial]].loc[cond],
		# 		 self.split_err['ev'].loc['no_1'].loc[trials[trial]].loc[cond] + self.split_err['err'].loc['no_1'].loc[trials[trial]].loc[cond],
		# 		 alpha=.5, color=color, label='%s'%(cond))

		# 		ax2[trial][1].plot(xaxis_tr, self.split_err['ev'].loc['exp_1'].loc[trials[trial]].loc[cond], label='%s'%(cond), color=color, marker='o', markersize=8)
				
		# 		ax2[trial][1].fill_between(xaxis_tr, self.split_err['ev'].loc['exp_1'].loc[trials[trial]].loc[cond] - self.split_err['err'].loc['exp_1'].loc[trials[trial]].loc[cond],
		# 		 self.split_err['ev'].loc['exp_1'].loc[trials[trial]].loc[cond] + self.split_err['err'].loc['exp_1'].loc[trials[trial]].loc[cond],
		# 		 alpha=.5, color=color)
		
		# for trial in range(nrows):
		# 	ax[trial][0].set_ylim([0,1])
		# 	ax[trial][1].set_ylim([0,1])
		# 	ax[trial][0].set_xlim([-2,3])
		# 	ax[trial][1].set_xlim([-2,3])

		# ax1.xaxis.grid(True)
		# ax2.xaxis.grid(True)

		# ax1.set_title('Did not expect a shock, N=11',size='xx-large')
		# ax2.set_title('Expected a shock, N=8',size='xx-large')
		# ax1.legend(['CS+','Scene','Rest'],loc='lower left', fontsize='larger')
		# ax1.set_ylabel('Classifier Evidence',size = 'x-large')
		# ax2.set_ylabel('Classifier Evidence',size = 'x-large')
		# ax1.set_xlabel('TR (away from stimulus onset)',size = 'x-large')
		# ax2.set_xlabel('TR (away from stimulus onset)',size = 'x-large')
		#fig.set_size_inches(12, 5)
		fig.set_size_inches(15,10)
		plt.tight_layout()
		# fig.savefig(os.path.join(data_dir,'graphing','4trial_expectancy_TR_ev.png'), dpi=300)
	
	def more_exp_stats(self):

		self.mstats = self.err_.reset_index()
		
		self.co_scene = {}
		self.co_rest = {}
		self.co_base = {'expect':{}, 'no':{}}
		
		self.map2 = {}

		# for resp in ['expect','no']:
			
		# 	self.map2[resp] = {}

		# 	for tr in self.trs:

		# 		self.map2[resp][tr] = {}

		# 		for cond in ['scene','rest']:

		# 			self.map2[resp][tr][cond] = self.stat_df['evidence'].loc[resp].loc[tr].loc[cond]

		# for tr in seltrs:
		# 	print(tr)
		# 	print(ttest_ind(self.map2['no'][tr]['scene'], self.map2['expect'][tr]['scene']))
		self._no_ = {}
		self._expect_ = {}

		for trial in [1,2,3,4]:
			if self.save_dict == ppa_prepped:
				self.exp_stats(trial_=trial,vis=False)
			else:
				self.beta_exp_stats(trial_=trial)

			self.base_stats = self.ev_base_err.copy()

			for resp in ['expect','no']:

				# self.co_base[resp][trial] = np.concatenate((self.base_stats[0].loc[resp]['-2'],
				# 									self.base_stats[0].loc[resp]['-1'],
				# 									self.base_stats[0].loc[resp]['0'],
				# 									self.base_stats[0].loc[resp]['1'],
				# 									self.base_stats[0].loc[resp]['2'],
				# 									self.base_stats[0].loc[resp]['3']))

				
				#the point here is to average over the 2 trs of the stimulus
				# self.co_base[resp][trial] = np.mean(np.array((self.base_stats.loc[resp].loc['0'],self.base_stats.loc[resp].loc['1'],self.base_stats.loc[resp].loc['2'],self.base_stats.loc[resp].loc['3'],)),axis=0)
				self.co_base[resp][trial] = self.base_stats.loc[resp].loc['0']
				#get rid of NaNs
				self.co_base[resp][trial] = self.co_base[resp][trial][~np.isnan(self.co_base[resp][trial])]
			print(ttest_ind(self.co_base['no'][trial],self.co_base['expect'][trial]))

		# self._no_ = np.concatenate((self.co_base['no'][1], self.co_base['no'][2]))
		# self._expect_ = np.concatenate((self.co_base['expect'][1], self.co_base['expect'][2]))
			self._no_[trial] = self.co_base['no'][trial]
			self._expect_[trial] = self.co_base['expect'][trial]

		# for resp in ['expect','no']:
		# 	self.co_scene[resp] = {}
		# 	self.co_rest[resp] = {}
		# 	for tr in [-2,-1,0,1,2,3]:
				
		# 		self.co_scene[resp] = np.mean((self.err_['ev'].loc[resp].loc[1].loc['scene'], 
		# 			self.err_['ev'].loc[resp].loc[2].loc['scene']),axis=0)

		# 	self.co_rest[resp] = np.mean((self.err_['ev'].loc[resp].loc[1].loc['rest'], 
		# 		self.err_['ev'].loc[resp].loc[2].loc['rest']),axis=0)

		# for resp in ['expect','no']:
		# 	self.co_base[resp] = self.co_scene[resp] - self.co_rest[resp]
		self.tr_no = {}
		self.tr_exp = {}

		for sub in self.sub_args:
			if sub in self.exp_split:
				self.tr_no[sub] = self.group_results[sub]['extinction_recall']['scene']
			else:
				self.tr_exp[sub] = self.group_results[sub]['extinction_recall']['scene']

		self.no_df = pd.DataFrame.from_dict({(sub): self.tr_no[sub]
											for sub in self.tr_no.keys()}, orient='index')

		self.exp_df = pd.DataFrame.from_dict({(sub): self.tr_exp[sub]
											for sub in self.tr_exp.keys()}, orient='index')
	
	def beta_exp_stats(self,trial_=1,vis=False):
		self.stat_df = self.exp_ev_df.set_index(['response','tr','condition','trial'])
		self.trs = [0]

		self.stat_map = {}

		for resp in ['expect','no']:
			
			self.stat_map[resp] = {}

			for tr in self.trs:

				self.stat_map[resp][tr] = {}

				# for cond in ['scene',self.comp_cond,'csplus']:
				for cond in ['scene',self.comp_cond]:
					self.stat_map[resp][tr][cond] = self.stat_df['evidence'].loc[resp].loc[tr].loc[cond].loc[trial_]
				self.pair_ttest = {'no':{0:{},1:{}}, 'expect':{0:{},1:{}}}

		#conduct the paired ttests
		self.pair_ttest['expect'][0]['t_stat'], self.pair_ttest['expect'][0]['p_val'] = ttest_rel(self.stat_map['expect'][0]['scene'], self.stat_map['expect'][0][self.comp_cond])

		self.pair_ttest['no'][0]['t_stat'], self.pair_ttest['no'][0]['p_val'] = ttest_rel(self.stat_map['no'][0]['scene'], self.stat_map['no'][0][self.comp_cond])

		#calculate the baseline as (scene-rest) for each TR in each condition
		self.ev_baseline = {}

		for resp in ['expect','no']:
			
			self.ev_baseline[resp] = {}

			for tr in self.trs:

				# self.ev_baseline[resp][tr] = self.stat_map[resp][tr]['scene'].values - self.stat_map[resp][tr][self.comp_cond].values
				self.ev_baseline[resp][tr] = self.stat_map[resp][tr]['scene'].values

		#set up the output structure for independent
		self.ind_ttest = {}	

		#conduct the independent ttests
		#took off an extra [0]?
		self.ind_ttest[0] = ttest_ind(self.ev_baseline['no'][0], self.ev_baseline['expect'][0], equal_var=True)

		self.pair_df = pd.DataFrame.from_dict({(response,tr): self.pair_ttest[response][tr]
												for response in self.pair_ttest.keys()
												for tr in self.pair_ttest[response].keys()},
												orient='index')
		self.pair_df.reset_index(inplace=True)
		self.pair_df.rename(columns={'level_0':'response', 'level_1':'tr'}, inplace=True)

		self.ind_df = pd.DataFrame.from_dict({(tr): self.ind_ttest[tr]
												for tr in self.ind_ttest.keys()},
												orient='index')

		self.ev_base_err = pd.DataFrame.from_dict({(response,tr): self.ev_baseline[response][tr]
												for response in self.ev_baseline.keys()
												for tr in self.ev_baseline[response].keys()},
												orient='index')
		
		self.ev_base_err.reset_index(inplace=True)
		hodl = self.ev_base_err['index'].apply(pd.Series)
		hodl.rename(columns={0:'response',1:'tr'}, inplace=True)
		self.ev_base_err.drop(['index'],axis=1,inplace=True)
		self.ev_base_err.index = pd.MultiIndex.from_product([['expect','no'],['0']],
												names=['response','tr'])


		_exp = self.ev_base_err.loc['expect']
		_no = self.ev_base_err.loc['no']

		_exp.dropna(axis=1,inplace=True)
		_no.dropna(axis=1,inplace=True)

		self.ev_base_graph = pd.DataFrame(index=pd.MultiIndex.from_product([['expect','no'],['0']],
												names=['response','tr']),
												columns=['avg','err'])
		self.ev_base_graph['avg'].loc['expect'].loc['0'] = _exp.loc['0'].mean()
		self.ev_base_graph['err'].loc['expect'].loc['0'] = sem(_exp.loc['0'])

		self.ev_base_graph['avg'].loc['no'].loc['0'] = _no.loc['0'].mean()
		self.ev_base_graph['err'].loc['no'].loc['0'] = sem(_no.loc['0'])

		self.ev_base_graph.reset_index(inplace=True)
		self.ev_base_graph.set_index(['response','tr'],inplace=True)


	def exp_stats(self,trial_=1,vis=True):
		#grab the raw data
		self.stat_df = self.exp_ev_df.set_index(['response','tr','condition','trial'])
		self.trs = [-2,-1,0,1,2,3]

		#collect just the scene and rest evidence for the different conditions at TR=0,1
		#for now just the first trial!!
		self.stat_map = {}

		for resp in ['expect','no']:
			
			self.stat_map[resp] = {}

			for tr in self.trs:

				self.stat_map[resp][tr] = {}

				# for cond in ['scene',self.comp_cond,'csplus']:
				for cond in ['scene',self.comp_cond]:
					self.stat_map[resp][tr][cond] = self.stat_df['evidence'].loc[resp].loc[tr].loc[cond].loc[trial_]

	
		#set up the output structure for paired
		self.pair_ttest = {'no':{0:{},1:{}}, 'expect':{0:{},1:{}}}

		#conduct the paired ttests
		self.pair_ttest['expect'][0]['t_stat'], self.pair_ttest['expect'][0]['p_val'] = ttest_rel(self.stat_map['expect'][0]['scene'], self.stat_map['expect'][0][self.comp_cond])
		self.pair_ttest['expect'][1]['t_stat'], self.pair_ttest['expect'][1]['p_val'] = ttest_rel(self.stat_map['expect'][1]['scene'], self.stat_map['expect'][1][self.comp_cond])

		self.pair_ttest['no'][0]['t_stat'], self.pair_ttest['no'][0]['p_val'] = ttest_rel(self.stat_map['no'][0]['scene'], self.stat_map['no'][0][self.comp_cond])
		self.pair_ttest['no'][1]['t_stat'], self.pair_ttest['no'][1]['p_val'] = ttest_rel(self.stat_map['no'][1]['scene'], self.stat_map['no'][1][self.comp_cond])

		#calculate the baseline as (scene-rest) for each TR in each condition
		self.ev_baseline = {}

		for resp in ['expect','no']:
			
			self.ev_baseline[resp] = {}

			for tr in self.trs:

				# self.ev_baseline[resp][tr] = self.stat_map[resp][tr]['scene'].values - self.stat_map[resp][tr][self.comp_cond].values
				self.ev_baseline[resp][tr] = self.stat_map[resp][tr]['scene'].values

		#set up the output structure for independent
		self.ind_ttest = {}	

		#conduct the independent ttests
		#took off an extra [0]?
		self.ind_ttest[-1] = ttest_ind(self.ev_baseline['no'][-1], self.ev_baseline['expect'][-1], equal_var=True)
		self.ind_ttest[0] = ttest_ind(self.ev_baseline['no'][0], self.ev_baseline['expect'][0], equal_var=True)
		self.ind_ttest[1] = ttest_ind(self.ev_baseline['no'][1], self.ev_baseline['expect'][1], equal_var=True)


		self.pair_df = pd.DataFrame.from_dict({(response,tr): self.pair_ttest[response][tr]
												for response in self.pair_ttest.keys()
												for tr in self.pair_ttest[response].keys()},
												orient='index')
		self.pair_df.reset_index(inplace=True)
		self.pair_df.rename(columns={'level_0':'response', 'level_1':'tr'}, inplace=True)

		self.ind_df = pd.DataFrame.from_dict({(tr): self.ind_ttest[tr]
												for tr in self.ind_ttest.keys()},
												orient='index')


		self.ev_base_err = pd.DataFrame.from_dict({(response,tr): self.ev_baseline[response][tr]
												for response in self.ev_baseline.keys()
												for tr in self.ev_baseline[response].keys()},
												orient='index')
		
		self.ev_base_err.reset_index(inplace=True)
		hodl = self.ev_base_err['index'].apply(pd.Series)
		hodl.rename(columns={0:'response',1:'tr'}, inplace=True)
		self.ev_base_err.drop(['index'],axis=1,inplace=True)
		self.ev_base_err.index = pd.MultiIndex.from_product([['expect','no'],['-2','-1','0','1','2','3']],
												names=['response','tr'])


		_exp = self.ev_base_err.loc['expect']
		_no = self.ev_base_err.loc['no']

		_exp.dropna(axis=1,inplace=True)
		_no.dropna(axis=1,inplace=True)

		self.ev_base_graph = pd.DataFrame(index=pd.MultiIndex.from_product([['expect','no'],['0','1']],
												names=['response','tr']),
												columns=['avg','err'])
		self.ev_base_graph['avg'].loc['expect'].loc['0'] = _exp.loc['0'].mean()
		self.ev_base_graph['err'].loc['expect'].loc['0'] = sem(_exp.loc['0'])
		self.ev_base_graph['avg'].loc['expect'].loc['1'] = _exp.loc['1'].mean()
		self.ev_base_graph['err'].loc['expect'].loc['1'] = sem(_exp.loc['1'])

		self.ev_base_graph['avg'].loc['no'].loc['0'] = _no.loc['0'].mean()
		self.ev_base_graph['err'].loc['no'].loc['0'] = sem(_no.loc['0'])
		self.ev_base_graph['avg'].loc['no'].loc['1'] = _no.loc['1'].mean()
		self.ev_base_graph['err'].loc['no'].loc['1'] = sem(_no.loc['1'])

		self.ev_base_graph.reset_index(inplace=True)
		self.ev_base_graph.set_index(['response','tr'],inplace=True)

		if vis:
			sns.set_style(rc={'axes.linewidth':'1.5'})
			plt.rcParams['xtick.labelsize'] = 22 
			plt.rcParams['ytick.labelsize'] = 22
			plt.rcParams['axes.labelsize'] = 22
			plt.rcParams['axes.titlesize'] = 26
			plt.rcParams['legend.labelspacing'] = .25
			plt.rcParams['axes.unicode_minus'] = False


			fig, ax1 = plt.subplots()
			ind2 = np.arange(1)    # the x locations for the groups
			width = 0.4         # the width of the bars
			sns.set_style('whitegrid')
			sns.set_style('ticks')
			p1 = ax1.bar(ind2, self.ev_base_graph['avg'].loc['no'].loc['1'], width, yerr=self.ev_base_graph['err'].loc['no'].loc['1'],
				color=plt.cm.Set1.colors[3], alpha=.8)
			p2 = ax1.bar(ind2+width, self.ev_base_graph['avg'].loc['expect'].loc['1'], width, yerr=self.ev_base_graph['err'].loc['expect'].loc['1'],
				edgecolor=plt.cm.Set1.colors[3], color='w', linewidth=4)
			legend = ax1.legend((p1[0], p2[0]), ('No', 'Yes'),fontsize='xx-large',title='Expect a shock?',loc='upper right')
			legend.get_title().set_fontsize('18') #legend 'Title' fontsize
			ax1.set_ylim([-.4,1])
			ax1.set_xticks(ind2 + width / 2)
			ax1.set_xticklabels(('1'))
			ax1.set_title('Differences in Relative Scene Evidence')
			ax1.set_ylabel('Relative Scene Evidence (Scene-Rest)')
			ax1.set_xlabel('TR (away from stimulus onset)')
			ax1.plot(.2,.75,marker='$*$',markersize=28, color='black')
			ax1.text(.14,.72,s=' \u2212',fontsize=32, color='black')
			ax1.text(.2,.72,s=' \u2212',fontsize=32, color='black')
			# ax1.text(.14,.72,s='- -',fontsize=35, color='black')

			fig.set_size_inches(8, 6)
			plt.tight_layout()
			# fig.savefig(os.path.join(data_dir,'graphing','quals','scene_minus_rest.png'), dpi=300)
	
	def ev_out(self,phase='extinction_recall',start=-1):

		evdf = self.event_df.loc[phase]
		evdf = evdf.reorder_levels(['cond','sub','trial'])

		self.tavg = pd.DataFrame(index=pd.MultiIndex.from_product(
							[['CS+','CS-'],range(1,13),self.sub_args],
							names=['condition','trial','subject']),
							columns=['evidence'])
		for sub in self.sub_args:
			events = glm_timing(sub,phase).phase_events(con=True)
			csp = np.where(events=='CS+')[0] + 1
			csm = np.where(events=='CS-')[0] + 1

			for trial in range(1,25):
				if trial in csp: 
					_con = 'CS+'
					_pos = np.where(csp == trial)[0][0] + 1
				elif trial in csm:
					_con = 'CS-'
					_pos = np.where(csm == trial)[0][0] + 1

				# self.tavg['evidence'][_con][_pos][sub] = np.mean([evdf[-1]['scene'][sub][trial], evdf[0]['scene'][sub][trial], evdf[1]['scene'][sub][trial]])
				# self.tavg['evidence'][_con][_pos][sub] = np.mean([evdf[0]['scene'][sub][trial], evdf[1]['scene'][sub][trial]])
				self.tavg['evidence'][_con][_pos][sub] = evdf[0]['scene'][sub][trial]
				# self.tavg['evidence'][_con][_pos][sub] = np.mean([evdf[0]['scene'][sub][trial], evdf[1]['scene'][sub][trial]])
				# self.tavg['evidence'][_con][_pos][sub] = np.mean(( (evdf[start]['scene'][sub][trial] - evdf[start]['rest'][sub][trial]) ))
				# self.tavg['evidence'][_con][_pos][sub] = np.mean([ (evdf[0]['scene'][sub][trial])])# - evdf[0]['rest'][sub][trial])])
																	#(evdf[1]['scene'][sub][trial] - evdf[1]['rest'][sub][trial])])
																	#(evdf[-1]['scene'][sub][trial] - evdf[-1]['rest'][sub][trial]) ])
		cspD = []
		csmD = []
		for trial in range(1,13):
			cspD.append(self.tavg['evidence'].loc['CS+'].loc[trial].std())
			csmD.append(self.tavg['evidence'].loc['CS-'].loc[trial].std())
		self.evD = np.concatenate((cspD,csmD))
		print(self.evD.mean())	

		plt_tavg = self.tavg.reset_index()
		plt_tavg.evidence = plt_tavg.evidence.astype('float')
		plt_tavg.subject = plt_tavg.subject.astype('float')
		plt_tavg.trial = plt_tavg.trial.astype('float')
		
		fig, ax = plt.subplots()
		ax = sns.boxplot(x='trial',y='evidence',hue='condition',data=plt_tavg)
		plt.title('%s - mean std = %.4f; std = %.4f'%(start, self.evD.mean(),self.evD.std()))

		#now do it for the initial ctx reinstatement relative to rest
		self.bic = {}
		for sub in self.sub_args:
			self.bic[sub] = np.mean(self.group_df['scene'].loc['extinction_recall'].loc[sub][0:4])# - self.group_df['rest'].loc['extinction_recall'].loc[sub][0:4])


	def exp_bootstrap(self):
		
		#in-class bootstrap function
		def bootstrap(group=None, nboot=1000):

			#generate results structure that takes the bootstrap samples
			bs_results = pd.DataFrame(index=pd.MultiIndex.from_product(
						[range(nboot),['csplus','csminus','scene','rest'],[-2,-1,0,1,2,3]],
						names=['rep','condition','tr']),
						columns=['evidence'])

			#and structure that takes the summary stats
			res_out = pd.DataFrame(index=pd.MultiIndex.from_product(
						[['csplus','scene','rest'],[-2,-1,0,1,2,3]],
						names=['condition','tr']),
						columns=['evidence','CI_low','CI_high'])

			#find the unique subs for this group
			subs = np.unique(group['subject'])
			#and count them
			N = len(subs)

			#do some pandas things to make our life easier
			group.set_index(['subject'],append=True,inplace=True)
			group = group.reorder_levels(['subject','condition','tr'])

			#for each bootstrap replication
			for rep in range(nboot):

				#create a results structure
				bs_iter = pd.DataFrame(index=pd.MultiIndex.from_product(
							[range(N),['csplus','csminus','scene','rest'],[-2,-1,0,1,2,3]],
							names=['rep','condition','tr']),
							columns=['evidence'])

				#randomly sample the subjects using the same sample size
				bootstrap_subs = subs[np.random.randint(0,N,size=N)]

				#fill in the sample
				for i, sub in enumerate(bootstrap_subs):

					bs_iter.loc[i] = group.loc[sub].values

				#collect the mean for this iteration
				bs_results['evidence'].loc[rep] = bs_iter.unstack(level=[1,2]).mean()['evidence'].values
			
			#come out of the loop and collect the summary stats
			bs_results = bs_results.unstack(level=0)
			res_out.reset_index(inplace = True)
			for cond in ['csplus','scene','rest']:
				_mean = np.zeros(6)
				_low = np.zeros(6)
				_high = np.zeros(6)

				#compute mean and 95% confidence interval
				for i, tr in enumerate([-2,-1,0,1,2,3]):

					_vals = bs_results.loc[cond].loc[tr]
					_mean[i] = _vals.mean()
					_low[i], _high[i] = np.percentile(_vals,2.5), np.percentile(_vals,97.5)
				
				res_out['evidence'][np.where(res_out.condition == cond)[0]] = _mean
				res_out['CI_low'][np.where(res_out.condition == cond)[0]] = _low
				res_out['CI_high'][np.where(res_out.condition == cond)[0]] = _high
					
			res_out.set_index(['condition','tr'], inplace=True)

			#return the results structure
			return res_out


		#Now use the function
		#collect the results from the classification 
		self.bs_raw = self.stat_df.reorder_levels(['trial','response','condition','tr'])
		#split them by group
		self.no_raw = self.bs_raw.loc[1].loc['no']
		self.yes_raw = self.bs_raw.loc[1].loc['expect']

		#run the bootstrap
		self.no_bs = bootstrap(group=self.no_raw)
		self.yes_bs = bootstrap(group=self.yes_raw)

		#do something with the results so they can be graphed easily
		self.bs_res = pd.DataFrame(index=pd.MultiIndex.from_product(
			[['no','expect'],['csplus','scene','rest'],[-2,-1,0,1,2,3]],
			names=['response','condition','tr']),
			columns=['evidence','CI_low','CI_high'])

		self.bs_res.loc['no'] = self.no_bs.values
		self.bs_res.loc['expect'] = self.yes_bs.values

		self.bs_res = self.bs_res.astype(float)

		#GRAPH!
		sns.set_style('whitegrid')
		sns.set_style('ticks')
		sns.set_style(rc={'axes.linewidth':'2'})
		plt.rcParams['xtick.labelsize'] = 10 
		plt.rcParams['ytick.labelsize'] = 10
		plt.rcParams['axes.labelsize'] = 10
		plt.rcParams['axes.titlesize'] = 12
		plt.rcParams['legend.labelspacing'] = .25
		
		xaxis_tr = [-2,-1,0,1,2,3]
		
		fig, ax = plt.subplots(2,2, sharex='col', sharey='row')
		for cond, color in zip(['csplus','scene','rest'], [plt.cm.Set1.colors[0],plt.cm.Set1.colors[3],plt.cm.Set1.colors[-1],'green']):
			
			if cond == 'csplus':
				labelcond = 'CS+'
			else:
				labelcond = cond

			ax[1][0].plot(xaxis_tr, self.bs_res['evidence'].loc['no'].loc[cond], color=color, marker='o', markersize=5, label=labelcond)
			
			ax[1][0].fill_between(xaxis_tr, self.bs_res['CI_low'].loc['no'].loc[cond], self.bs_res['CI_high'].loc['no'].loc[cond], alpha=.5, color=color)

			ax[1][1].plot(xaxis_tr, self.bs_res['evidence'].loc['expect'].loc[cond], color=color, marker='o', markersize=5, label=labelcond)
			
			ax[1][1].fill_between(xaxis_tr, self.bs_res['CI_low'].loc['expect'].loc[cond], self.bs_res['CI_high'].loc['expect'].loc[cond], alpha=.5, color=color)

			trial = 0
			trials = [1]

			ax[0][0].plot(xaxis_tr, self.err_['ev'].loc['no'].loc[trials[trial]].loc[cond], color=color, marker='o', markersize=5, label=labelcond)
			
			ax[0][0].fill_between(xaxis_tr, self.err_['ev'].loc['no'].loc[trials[trial]].loc[cond] - self.err_['err'].loc['no'].loc[trials[trial]].loc[cond],
			 self.err_['ev'].loc['no'].loc[trials[trial]].loc[cond] + self.err_['err'].loc['no'].loc[trials[trial]].loc[cond],
			 alpha=.5, color=color)

			ax[0][1].plot(xaxis_tr, self.err_['ev'].loc['expect'].loc[trials[trial]].loc[cond], label='%s'%(labelcond), color=color, marker='o', markersize=5)
			
			ax[0][1].fill_between(xaxis_tr, self.err_['ev'].loc['expect'].loc[trials[trial]].loc[cond] - self.err_['err'].loc['expect'].loc[trials[trial]].loc[cond],
			 self.err_['ev'].loc['expect'].loc[trials[trial]].loc[cond] + self.err_['err'].loc['expect'].loc[trials[trial]].loc[cond],
			 alpha=.5, color=color)
		
		ax[trial][0].set_title('CS+ Trial %s; Did not expect a shock (N=%s)'%(trials[trial], self.nexp['no'][trials[trial]]))
		ax[trial][1].set_title('CS+ Trial %s; Expected a shock (N=%s)'%(trials[trial], self.nexp['expect'][trials[trial]]))
		
		ax[1][0].set_title('Bootstrap Estimate with 95% CI')
		ax[1][1].set_title('Bootstrap Estimate with 95% CI')

		ax[1][0].set_xlabel('TR (away from stimulus onset)')
		ax[1][1].set_xlabel('TR (away from stimulus onset)')
		
		ax[0][0].set_ylabel('Classifier Evidence')
		ax[1][0].set_ylabel('Classifier Evidence')

		if not self.ptsd: ax[0][0].plot(1,.2,marker='$*$',markersize=10, color='black')
		fig.set_size_inches(10, 6)

		ax[0][0].legend(loc='upper center', bbox_to_anchor=(0.5, 1), frameon=True, ncol=3,borderaxespad=1)

		if not self.ptsd: fig.savefig(os.path.join(data_dir,'graphing','controls.png'), dpi=300)
		if self.ptsd: fig.savefig(os.path.join(data_dir,'graphing','ptsd.png'), dpi=300)
	
	def exp_scr(self):

		raw_scr = scr_stats('extinction_recall')
		self.er_scr_raw = raw_scr.sub_dfs
		
		self.er_scr = {}
		for sub in self.sub_args:
			self.er_scr[sub] = {}
			for con in ['CS+','CS-']:
				self.er_scr[sub][con] = {}
				for i in range(1,13):
					self.er_scr[sub][con][i] = {}


		for sub in self.sub_args:
			sub_meta = meta(sub).meta

			#collect index for fear and ER, and use the paragraph for finding extinction
			phase4_loc = np.where(sub_meta.phase == 'extinctionRecall')[0]

			_phase4 = {'resp':sub_meta['stim.RESP'][phase4_loc], 'condition': sub_meta['cstype'][phase4_loc], 'cs_trial': sub_meta['cstypebytrial'][phase4_loc]}
			phase4 = pd.DataFrame(_phase4)
			phase4.index = range(1,25)
			phase4.fillna(0,inplace=True)

			sub_scr= self.er_scr_raw[sub]
			sub_scr.index = range(1,25)

			for i in self.er_scr_raw[sub].index:
				self.er_scr[sub][sub_scr.loc[i,'CStype']][int(phase4['cs_trial'][i][-2:])] = sub_scr.loc[i, 't2p']

		self.er_scr_df = pd.DataFrame.from_dict({(sub, con): self.er_scr[sub][con]
												for sub in self.er_scr.keys()
												for con in self.er_scr[sub].keys()},
												orient='index')

		self.er_scr_df.reset_index(inplace=True)
		self.er_scr_df.rename(columns={'level_0':'sub','level_1':'condition'},inplace=True)
		self.er_scr_df = self.er_scr_df.melt(id_vars=['sub','condition'])
		self.er_scr_df.rename(columns={'variable':'trial','value':'t2p'},inplace=True)

		# ax = sns.factorplot(data=self.er_scr_df, x='trial', y='t2p',
		# 					hue='condition',
		# 					kind='point',dodge=True)

		self.er_scr_stats = self.er_scr_df.set_index(['sub','condition','trial'])
		self.er_scr_stats = self.er_scr_stats.reorder_levels(['sub','trial','condition'])
		er_scr_1st_csplus = self.er_scr_stats.copy()

		self.er_stats_df = pd.DataFrame(index=pd.MultiIndex.from_product(
											[range(1,13),['CS+','CS-']],
											names=['trial','condition']),
											columns=['avg','err'])
		
		self.er_stats_df['avg'] = self.er_scr_stats.groupby(level=[1,2]).mean()
		self.er_stats_df['err'] = self.er_scr_stats.groupby(level=[1,2]).sem() 
		self.er_stats_df = self.er_stats_df.reorder_levels(['condition','trial'])

		plt.rcParams['xtick.labelsize'] = 14 
		plt.rcParams['ytick.labelsize'] = 14
		fig, ax = plt.subplots()
		
		ind1 = np.arange(.95,12.95,1)

		ind2 = np.arange(1.05,13.05,1)
		ax.plot(ind1, self.er_stats_df['avg'].loc['CS+'], marker='o', color=plt.cm.Set1.colors[0])
		ax.fill_between(ind1, self.er_stats_df['avg'].loc['CS+'] - self.er_stats_df['err'].loc['CS+'],
			self.er_stats_df['avg'].loc['CS+'] + self.er_stats_df['err'].loc['CS+'],
			label='%s'%('CS+'), color=plt.cm.Set1.colors[0], alpha=.5)
		
		ax.plot(ind2, self.er_stats_df['avg'].loc['CS-'], marker='o', color=plt.cm.Set1.colors[1])
		ax.fill_between(ind2, self.er_stats_df['avg'].loc['CS-'] - self.er_stats_df['err'].loc['CS-'],
			self.er_stats_df['avg'].loc['CS-'] + self.er_stats_df['err'].loc['CS-'],
			label='%s'%('CS+'), color=plt.cm.Set1.colors[1], alpha=.5)
		# ax.legend(loc='upper right', fontsize='larger')
		
		plt.xticks(np.arange(1,13,step=1))
		# ax.set_ylabel('Square Root SCR',size = 'x-large')
		# ax.set_xlabel('Trial',size = 'x-large')
		ax.set_xlim([.9,12.1])
		fig.set_size_inches(12, 6)
		plt.tight_layout()

		# fig.savefig(os.path.join(data_dir,'graphing','cns','SCR_ER.png'), dpi=300)



		self.scr_exp = {'expect':{}, 'no':{}}

		for sub in self.sub_args:

			#get the phase conditions and correct for pythonic indexing
			pc = glm_timing(sub, 'extinction_recall').phase_events(con=True)
			pc.index = range(1,pc.shape[0]+1)

			csplus_map = (np.where(pc=='CS+')[0] + 1)

			first_csp = csplus_map[0]

			if self.exp_bhv[sub]['extinction_recall']['CS+'][1]['exp'] == 1:

				self.scr_exp['expect'][sub] = self.er_scr_stats['t2p'].loc[sub].loc[first_csp].loc['CS+']

			elif self.exp_bhv[sub]['extinction_recall']['CS+'][1]['exp'] == 0:

				self.scr_exp['no'][sub] = self.er_scr_stats['t2p'].loc[sub].loc[first_csp].loc['CS+']

	def memory_events(self):
		memory_phases = ['memory_run_1','memory_run_2','memory_run_3']
		
		enc_ev = {}

		for phase in memory_phases:
			
			phase_df = self.event_df.loc[phase]
			phase_df = phase_df.reorder_levels(['sub','trial','cond'])
			# phase_df.reset_index(inplace=True) 
			# phase_df = phase_df.melt(id_vars=['sub','trial','cond'])                     
			# phase_df = phase_df.rename({'variable':'tr', 'value':'ev'},axis=1) 
			# phase_df = phase_df.set_index(['sub','trial','tr','cond'])

			# enc_ev = pd.DataFrame(index=pd.MultiIndex.from_product(
			# 			[phase,list(sub_args),list([-3,-2,-1,0,1,2,3]),self.group_decode_conds],
			# 			names=['phase','subject','tr','cond']),
			# 			columns=['ev'])
			
			# enc_ev[phase] = {'baseline':{}, 'fear_conditioning':{}, 'extinction':{}, 'foil':{}}
			enc_ev[phase] = {'extinction':{}, 'not':{}}
			
			for sub in self.sub_args:

				# enc_ev[phase]['baseline'][sub] = {}
				# enc_ev[phase]['fear_conditioning'][sub] = {}
				enc_ev[phase]['extinction'][sub] = {}
				# enc_ev[phase]['foil'][sub] = {}
				enc_ev[phase]['not'][sub] = {}
				

			
				#get the phase conditions and correct for pythonic indexing
				pc = glm_timing(sub, phase).mem_events()
				pc.index = range(1,pc.shape[0]+1)

				#have to correct index again
				base_map = (np.where(pc['encode'] == 'baseline')[0] + 1)
				fear_map = (np.where(pc['encode'] == 'fear_conditioning')[0] + 1)
				ext_map = (np.where(pc['encode'] == 'extinction')[0] + 1)
				foil_map = (np.where(pc['encode'] == 'foil')[0] + 1)

				not_map = np.concatenate((base_map,fear_map,foil_map))

				# for i, trial in enumerate(base_map):
				# 	enc_ev[phase]['baseline'][sub][i] = phase_df.loc[sub].loc[trial]
				# for i, trial in enumerate(fear_map):
				# 	enc_ev[phase]['fear_conditioning'][sub][i] = phase_df.loc[sub].loc[trial]
				for i, trial in enumerate(ext_map):
					enc_ev[phase]['extinction'][sub][i] = phase_df.loc[sub].loc[trial]
				# for i, trial in enumerate(foil_map):
				# 	enc_ev[phase]['foil'][sub][i] = phase_df.loc[sub].loc[trial]
				for i, trial in enumerate(not_map):
					enc_ev[phase]['not'][sub][i] = phase_df.loc[sub].loc[trial]
					

		self.enc_ev = enc_ev

		self.mem_ev_df = pd.DataFrame.from_dict({(phase, encode, sub, trial, tr): enc_ev[phase][encode][sub][trial][tr]
											for phase in enc_ev.keys()
											for encode in enc_ev[phase].keys()
											for sub in enc_ev[phase][encode].keys()
											for trial in enc_ev[phase][encode][sub].keys()
											for tr in enc_ev[phase][encode][sub][trial].keys()},
											orient='index')
		
		self.mem_ev_df.reset_index(inplace=True)
		# hold = self.mem_ev_df['index'].apply(pd.Series)
		# hold.rename(columns={0:'phase',1:'encode',2:'subject',3:'trial', 4:'tr'}, inplace=True)

		# self.mem_ev_df = self.mem_ev_df.drop(columns='index')
		# self.mem_ev_df = pd.concat([hold, self.mem_ev_df])
		# self.mem_ev_df.set_index(['phase','encode','subject','trial','tr'], inplace=True)
		

		self.mem_ev_df.rename(columns={'level_0':'phase', 'level_1':'encode', 'level_2':'subject', 'level_3':'trial', 'level_4':'tr'}, inplace=True)
		self.mem_ev_df = self.mem_ev_df.melt(id_vars=['phase','encode','subject','trial','tr'])
		self.mem_ev_df.rename(columns={'variable':'condition', 'value':'evidence'}, inplace=True)

		
		ax = sns.factorplot(data=self.mem_ev_df, x='tr', y='evidence',
							hue='condition', col='encode',
							kind='point',dodge=True)
		# plt.savefig('%s/NEW_BETA_ER_trial_evidence_with_expectancy'%(data_dir + 'graphing' + os.sep + 'mvpa_analysis'))

	def vis_cond_phase(self, phase=None, title=''):
		
		results=self.group_stats

		n_classes = len(self.group_decode_conds)
		print(n_classes)

		index = range(0, len(results[phase]['scene']['ev']))


		plt.figure()
		colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green','black'])
		for cond, color in zip(results[phase].keys(), colors):
			plt.plot(index, results[phase][cond]['ev'][index], color=color, lw=2,
				label='%s'%(cond))
		plt.legend()
		plt.title(phase + '; ' + title)
		plt.xlabel('TR')
		plt.ylabel('classifier evidence')
		plt.savefig('%s/%s_%s'%(data_dir + 'graphing' + os.sep + 'mvpa_analysis' + os.sep + 'cns', phase, title))

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
		n_classes = len(self.group_decode_conds)

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

		n_classes = len(self.group_decode_conds)

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
		plt.savefig('%s/%s_%s'%(data_dir + 'graphing' + os.sep + 'mvpa_analysis' + os.sep + 'cns' + os.sep + 'event', phase, title))

import os
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns

from fc_config import data_dir, mvpa_prepped, dataz, sub_args, hdr_shift, n_trials, working_subs
from fc_decoding import loc_decode
from preprocess_library import meta
from glm_timing import glm_timing

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold

from itertools import cycle
from scipy.stats import sem


class decode():

	decode_runs = ['baseline','fear_conditioning','extinction','extinction_recall']

	def __init__(self, subj=0, imgs='tr', k=0, save_dict=mvpa_prepped):

		self.subj = meta(subj)
		print('Decoding %s'%(self.subj.fsub))

		self.loc_dat, self.loc_labels = self.load_localizer(imgs=imgs)

		self.test_dat = self.load_test_dat(runs=decode.decode_runs, save_dict=save_dict)

		self.init_fit_clf(k=k, data=self.loc_dat, labels=self.loc_labels)

		self.clf_res = self.decode_test_data(test_data=self.test_dat)


	def load_localizer(self, imgs=None):

		loc_dat = loc_decode.load_localizer(self, imgs)

		loc_labels = loc_decode.better_tr_labels(self, data=loc_dat, imgs=imgs)
		
		#shift 3TRs (labels are already done in better_tr_labels)
		loc_dat = { phase: loc_dat[phase][hdr_shift['start']:, :] for phase in loc_dat}

		loc_dat = np.concatenate([loc_dat['localizer_1'], loc_dat['localizer_2']])

		loc_labels, loc_dat = loc_decode.snr_hack(self, labels=loc_labels, data=loc_dat)



		'''
		#hard coding in removing rest and collapsing scenes for training localizer
		#this provides the best decoding accuracy
		if 'tr' in imgs:
			loc_labels, loc_dat = loc_decode.remove_rest(self, labels=loc_labels, data=loc_dat)
		'''
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

		proba_res = { phase: self.clf.predict_proba(test_data[phase]) for phase in test_data }

		self.clf_lab = list(self.clf.classes_)

		_res = {phase:{} for phase in test_data}

		for phase in _res:
			
			for i, label in enumerate(self.clf_lab):

				_res[phase][label] = proba_res[phase][:,i]

		labels = list(self.clf.classes_)
		nlab = list(range(0,len(labels)))

		# self.base_res = pd.DataFrame([], columns=nlab)
		# self.fear_res = pd.DataFrame([], columns=nlab)
		# self.ext_res = pd.DataFrame([], columns=nlab)
		# self.er_res = pd.DataFrame([], columns=nlab)

		# for i in nlab:

		# 	self.base_res[i] = proba_res['baseline'][:,i]
		# 	self.fear_res[i] = proba_res['fear_conditioning'][:,i]
		# 	self.ext_res[i] = proba_res['extinction'][:,i]
		# 	self.er_res[i] = proba_res['extinction_recall'][:,i]

		# 	self.base_res = self.base_res.rename({i: labels[i]}, axis='columns')
		# 	self.fear_res = self.fear_res.rename({i: labels[i]}, axis='columns')
		# 	self.ext_res = self.ext_res.rename({i: labels[i]}, axis='columns')
		# 	self.er_res = self.er_res.rename({i: labels[i]}, axis='columns')
		return _res


class group_decode():

	conds = ['csplus','csminus','scene','scrambled','rest']

	def __init__(self, imgs='tr', k=1000, save_dict=mvpa_prepped):

		print('hi hello')
		self.sub_decode(imgs=imgs, k=k, save_dict=save_dict)
		self.event_res = self.event_results(self.group_results)

	def sub_decode(self, imgs='tr', k=1000, save_dict=mvpa_prepped):
		
		group_res = {}

		for sub in working_subs:

			sub_dec = decode(sub, imgs=imgs, k=k, save_dict=save_dict)
			sub_res = sub_dec.clf_res

			group_res[sub] = {}

			for phase in decode.decode_runs:

				group_res[sub][phase] = {}				
			
				group_res[sub][phase] = { label: sub_res[phase][label] for label in sub_dec.clf_lab }

		#group_results[sub][phase][cond]
		#make it a self here so it can be referenced later
		self.group_results = group_res

		#this nifty little line creates a dataframe from the nested dict group_res
		#the outer index is phase, and the inner index is subject number
		#the columns are the labels (scene,animal,etc.)		
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
	'''
	this is not done obv.
	event_res(self.group_results)
	'''
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

					event_res[sub][phase][trial] = {}

					if events.start_rem[trial] <= 1:
						
						start = events.start_tr[trial]

					elif events.start_rem[trial] >= 1:

						start = events.start_tr[trial] + 1				

					end = events.end_tr[trial]


					#right now we are looking 1 TR before the stim comes on through 1 TR after it ends
					#range() is not inclusive on the upper bound hence the +2
					window = range(int(events.start_tr[trial])-1, int(events.end_tr[trial])+2)
					#print(sub, phase, trial, window)
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

						event_res[sub][phase][trial][cond] = sub_res[sub][phase][stim_cond][window]

		event_df = pd.DataFrame.from_dict( {(phase, cond, trial, sub): event_res[sub][phase][trial][cond]
											for sub in event_res.keys()
											for phase in event_res[sub].keys()
											for trial in event_res[sub][phase].keys()
											for cond in event_res[sub][phase][trial].keys()}, orient='index')

		event_df.index = pd.MultiIndex.from_tuples(event_df.index, names=('phase','cond','trial','sub'))

		event_df.columns = [-1,0,1,2,3,4]

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

		return event_stats


def vis_cond_phase(results, phase=None, title=None):
	
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


def phase_bar_plot(results, title=None):
	
	stats = get_bar_stats(results)

	sns.barplot(x='phase',y='avg',hue='cond',data=stats)
	
	plt.title(title)


def get_bar_stats(results):

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



def vis_phase(results, cond='scene',index=range(0,50), title=None):

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













	# ax.errorbar(index, phase.csp_mean, yerr=phase.csp_sem, label='csplus')
	# ax.errorbar(index, phase.csm_mean, yerr=phase.csm_sem, label='csminus')
	# ax.errorbar(index, phase.sce_mean, yerr=phase.sce_sem, label='scene')
	# ax.errorbar(index, phase.scr_mean, yerr=phase.scr_sem, label='scrambled')
	
	# ax.plot(index, phase.csp_mean, label='csplus')
	# ax.plot(index, phase.csm_mean, label='csminus')
	# ax.plot(index, phase.sce_mean, label='scene')
	# ax.plot(index, phase.scr_mean, label='scrambled')
	# ax.plot(index, phase.rest_mean, label='rest')

	#first thing i need to do is take rest out again

	# ax.plot(index, np.mean(phase.baseline.sce_mean[:135]), label='baseline')
	# ax.plot(index, np.mean(phase.fear_conditioning.sce_mean[:135]), label='fear_conditioning')
	# ax.plot(index, np.mean(phase.extinction.sce_mean[:135]), label='extinction')
	# ax.plot(index, np.mean(phase.extinction_recall.sce_mean), label='extinction_recall')
	

	# ax.set_xlabel('TR')
	# ax.set_ylabel('Classifier Evidence')

	# legend = ax.legend()

	# frame = legend.get_frame()
	# frame.set_facecolor('0.90')

	# # Set the fontsize
	# for label in legend.get_texts():
	#     label.set_fontsize('large')

	# for label in legend.get_lines():
	#     label.set_linewidth(1.5)  # the legend line width


def vis_event_res(results,phase=None,title=None):

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
	plt.savefig('%s/%s_%s'%(data_dir + 'graphing' + os.sep + 'mvpa_analysis', phase, title))

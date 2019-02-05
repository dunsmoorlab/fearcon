import os
import numpy as np
import pandas as pd
import nibabel as nib
import sys
from glob import glob
from nilearn import signal
from nilearn.masking import apply_mask, unmask

from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import label_binarize
import seaborn as sns
import matplotlib.pyplot as plt

from fc_config import *
from preprocess_library import meta
from glm_timing import glm_timing



class loc_decode(object):
	#k must be an array of values to iter xval over
	def __init__(self, subj=0, imgs=None, name='no_name', del_rest=False,
				scene_collapse=False, scene_DS=False, rmv_indoor=False, binarize=False, rmv_scram=False, k=None, cf=False, save_dict=mvpa_masked_prepped, verbose=False):

		self.verbose=verbose
		self.subj = meta(subj)
		if self.verbose: print('Decoding %s'%(self.subj.fsub))

		self.loc_dat = self.load_localizer(imgs, save_dict)

		self.loc_lab = self.better_tr_labels(data=self.loc_dat, imgs=imgs)

		if 'tr' in imgs:
			self.loc_dat = self.tr_shift(data=self.loc_dat)
				
		#and concat
		if self.subj.num == 107: self.loc_dat = self.loc_dat['localizer_1']
		else: self.loc_dat = np.concatenate([self.loc_dat['localizer_1'], self.loc_dat['localizer_2']])

		if 'tr' in imgs:
			self.loc_lab, self.loc_dat = self.snr_hack(labels=self.loc_lab, data=self.loc_dat)


		if self.verbose: print(self.loc_dat.shape)

		if del_rest:

			self.loc_lab, self.loc_dat = self.remove_rest(labels=self.loc_lab, data=self.loc_dat)

		if scene_DS:
			if 'tr' in imgs: self.loc_lab, self.loc_dat = self.downsample_scenes(labels=self.loc_lab, data=self.loc_dat)

		if scene_collapse:

			self.loc_lab = self.collapse_scenes(labels=self.loc_lab)
		
		if rmv_indoor:

			self.loc_lab, self.loc_dat = self.remove_indoor(labels=self.loc_lab,data=self.loc_dat)


		if rmv_scram:

			self.loc_lab, self.loc_dat = self.remove_scrambled(labels=self.loc_lab, data=self.loc_dat)
	
		if binarize:

			self.loc_lab, self.loc_dat = self.binarize(labels=self.loc_lab, data=self.loc_dat)

		self.loc_groups = self.set_xval_groups(labels=self.loc_lab)

		self.init_clasif(alg='logreg')

		self.create_output(_name=name,_index=cv_iter)
		for k in k:

			if self.verbose: ('Running xval, k=%s'%(k))
			self.res.loc[k] = self.run_xval(data=self.loc_dat, labels=self.loc_lab, xval_groups=self.loc_groups, k=k)
			if self.verbose: print('2-fold acc = %s'%(self.res.loc[k].values))
			if cf:
				self.cf_mat, self.cf_rep = self.confusion_matrices(labels=self.loc_lab, data=self.loc_dat, groups=self.loc_groups)
		
		if not binarize:
			self.decode_ts(data=self.loc_dat,labels=self.loc_lab,groups=self.loc_groups)

		self.auc = self.create_roc(labels=self.loc_lab,data=self.loc_dat,groups=self.loc_groups)
	#takes in 'tr' or 'beta' for imgs
	def load_localizer(self, imgs, save_dict):

		#mvpa_masked_prepped
		#if you want to load in the tr by tr data (not beta)
		if imgs == 'tr':
			if self.verbose: print('loading TR localizer data')

			if self.subj.num == 107:
				phase = 'localizer_1'
				loc_dat = {}
				loc_dat[phase] = np.load('%s%s'%(self.subj.bold_dir, save_dict[phase]))[dataz]
			else:
				#load in the motion corrected niftis
				loc_dat = { phase: np.load('%s%s'%(self.subj.bold_dir, save_dict[phase]))[dataz] for phase in mvpa_masked_prepped if 'localizer' in phase }

			return loc_dat


		elif imgs == 'beta':
			#beta_masked_prepped
			if self.verbose: print('loading beta estimates localizer data')
			if save_dict == ppa_prepped: 
				beta_dict = beta_ppa_prepped
			elif save_dict == beta_ppa_prepped:
				beta_dict = beta_ppa_prepped
			elif save_dict == combined_prepped:
				beta_dict = beta_combined_prepped
			elif save_dict == mvpa_masked_prepped: 
				beta_dict = beta_masked_prepped
			else:
				beta_dict = save_dict
			#correct for Sub107
			if self.subj.num == 107:
				phase = 'localizer_1'
				loc_dat = {}
				loc_dat[phase] = np.load('%s%s'%(self.subj.bold_dir, beta_dict[phase]))[dataz]
			else:

				# loc_dat = { phase: nib.concat_images(glob('%s%snew_ls-s_betas/beta_***.nii.gz'%(self.subj.bold_dir, phase2rundir[phase]))) for phase in phase2rundir if 'localizer' in phase }
				loc_dat = { phase: np.load('%s%s'%(self.subj.bold_dir, beta_dict[phase]))[dataz] for phase in beta_masked_prepped if 'localizer' in phase }
				# loc_dat = loc_decode.prep_beta_vols(self, data=loc_dat)

			return loc_dat

	def better_tr_labels(self,data=None, imgs=None):

		if self.verbose: print('Making labels')
		out_labels = dict.fromkeys(list(data.keys()))

		if imgs == 'beta':
			out_labels = {phase: glm_timing(self.subj.num, phase).loc_blocks(con=True) for phase in data}

		elif 'tr' in imgs:
			#shift = hdr_shift['start']
			shift = 2
			for phase in data:
				#grab the onset information
				events = glm_timing(self.subj.num,phase).phase_events()
				#calculate the TR in which the stims are first presented, as well as the remainder
				events['start_tr'], events['start_rem'] = divmod(events.onset, 2)
				events['end_tr'], events['end_rem'] = divmod(events.onset+events.duration, 2)

				#create the label structure
				labels = pd.Series(index=np.arange(data[phase].shape[0]))

				#for every event in the run:
				#THIS IS WHERE THE TR SHIFT HAPPENS NOW
				#THE LABELS ARE PUT 3 TRs away from where they would be if we were labeling onsets
				for trial in events.index:
					#i did this just to avoid lots of typing, makes it look cleaner
					start_tr = events.start_tr[trial]
					start_rem = events.start_rem[trial]
					end_tr = events.end_tr[trial]
					end_rem = events.end_rem[trial]
					label = events.trial_type[trial]

					#if the stimulus starts and ends within the same TR, then label that TR!
					if start_tr == end_tr:
						labels[start_tr+shift] = label
					#otherwise if the stimulus starts pretty early in the starting TR,
					#but loses .3s of it in the next TR, go ahead and label the start TR
					elif start_rem <= 1.3:
						labels[start_tr+shift] = label
					#otherwise, if it ends within a second of the next TR, label the next one
					elif end_rem <= .5:
						labels[end_tr+shift] = label
					#and then also the label the next one for this condition
					elif start_rem >= 1.5 and end_rem >= .5:
						labels[end_tr+shift] = label						
				labels.fillna(value='rest', inplace=True)
				out_labels[phase] = labels

		#if tr then shift
		if 'tr' in imgs:
			out_labels = {phase: out_labels[phase][hdr_shift['start']:] for phase in out_labels}

		#concatenate the two localizer runs
		if self.subj.num == 107: out_labels = np.array(out_labels['localizer_1'])
		else: out_labels = np.concatenate([out_labels['localizer_1'], out_labels['localizer_2']])

		return out_labels


	def snr_hack(self, labels=None, data=None):

		tr_to_drop = np.array([])

		_labels = np.copy(labels)

		_labels[np.where(_labels != 'rest')] = 'stim'

		block_start_target = ['rest','stim','stim']

		#git rid of the rests on either side of the middle rest
		rest3_target = ['stim','rest','rest','rest','stim']

		#get rid of everything but the middle rest
		rest5_target = ['rest','rest','rest','rest','rest']
		
		#3 window loop
		window3 = np.array(range(3))

		for i in range(len(_labels)-3):

			#find the transition to rest, what we want to do here is git rid of the 
			#first 2 TRs of the block, so we want to find when the pattern comparison to rest
			#is ['rest','stim','stim']
			if np.array_equal(_labels[window3], block_start_target):

				tr_to_drop = np.append(tr_to_drop, window3)

			window3 += 1

		#5window loop
		window5 = np.array(range(5))

		#same thing, this time finding places to keep only rest in the middle of 5 TRs
		for i in range(len(_labels)-5):

			if np.array_equal(_labels[window5], rest3_target):

				tr_to_drop = np.append(tr_to_drop, [window5[1], window5[3]])

			if np.array_equal(_labels[window5], rest5_target):

				tr_to_drop = np.append(tr_to_drop, np.concatenate([window5[:2], window5[-2:]]))

			window5 += 1


		tr_to_drop = tr_to_drop.astype(int)

		if self.verbose: print('deleting %s in hopes that it increases decoding...'%(len(tr_to_drop)))

		#delete those conditions
		labels = np.delete(labels, tr_to_drop)
		
		data = np.delete(data, tr_to_drop, axis=0)

		#and make sure to update the working _labels
		_labels = np.delete(_labels, tr_to_drop)

		
		#fix anywhere that might be ['rest','stim','rest']
		#because those are just no good at all...
		label_fix_target = ['rest','stim','rest']
		
		#reinitialize these variables because the size of the array has changed
		window3 = np.array(range(3))
		tr_to_drop = np.array([])

		for i in range(len(_labels)-3):

			#find the transition to rest, what we want to do here is git rid of the 
			#first 2 TRs of the block, so we want to find when the pattern comparison to rest
			#is ['rest','stim','stim']
			if np.array_equal(_labels[window3], label_fix_target):

				tr_to_drop = np.append(tr_to_drop, window3)

			#if the comp window matches the patter i want but hard to do that

			window3 += 1

		#idk if this will help but lets give it a shot...
		# tr_to_drop = np.append(tr_to_drop, np.where(labels == 'indoor'))

		if self.verbose: print('deleting %s in hopes that it increases decoding...'%(len(tr_to_drop)))

		#delete those conditions
		labels = np.delete(labels, tr_to_drop)
		
		data = np.delete(data, tr_to_drop, axis=0)
		
		#do the count here
		############################
		# for label in np.unique(labels):
		# 	print('%s = %s'%(label,len(np.where(labels == label)[0])))
		
		return labels, data

	def downsample_scenes(self, labels=None, data=None):

		indoor = np.where(labels=='indoor')[0]
		outdoor = np.where(labels=='outdoor')[0]

		tr_to_drop = np.array([])

		tr_to_drop = np.append(tr_to_drop, indoor[0:-1:2])
		tr_to_drop = np.append(tr_to_drop, outdoor[0:-1:2])

		labels = np.delete(labels, tr_to_drop)
		
		data = np.delete(data, tr_to_drop, axis=0)

		return labels, data


	def moving_avg(self, labels=None, data=None):
		
		target = ''
		target_loc = 0
		
		av_dat = {}
		av_lab = {}

		j = 0

		for i, label in enumerate(labels):
			
			if target != label:
				
				block = np.array(range(target_loc, i))

				if len(block) == 6:
					#find the mid point of the block
					mid = int(len(block) / 2)

					#get the avg of the first half of the block
					av_dat[j] = np.mean(data[block[:mid], :],axis=0)
					av_lab[j] = target
					j += 1

					#and the second half
					av_dat[j] = np.mean(data[block[mid:], :],axis=0)
					av_lab[j] = target
					j += 1

				if len(block) == 5 or len(block) == 4:

					av_dat[j] = np.mean(data[block[-3:], :],axis=0)
					av_lab[j] = target
					j += 1

				if len(block) == 3:

					av_dat[j] = np.mean(data[block, :],axis=0)
					av_lab[j] = target
					j += 1

				
				if len(block) == 2 and label == 'rest':

					av_dat[j] = np.mean(data[block, :],axis=0)
					av_lab[j] = target
					j += 1

				if len(block) == 1 and label == 'rest':

					av_dat[j] = np.mean(data[block, :],axis=0)
					av_lab[j] = target
					j += 1


				target = label

				target_loc = i


		data = np.array([av_dat[tr] for tr in av_dat])
		labels = np.array([av_lab[tr] for tr in av_lab])

		return labels, data


	def remove_indoor(self, labels=None, data=None):
		if self.verbose: print('removing indoor scenes, animals, tools')
		tr_to_drop = np.array([])
		tr_to_drop = np.append(tr_to_drop, np.where(labels == 'indoor'))
		# tr_to_drop = np.append(tr_to_drop, np.where(labels == 'animal'))
		# tr_to_drop = np.append(tr_to_drop, np.where(labels == 'tool'))
		# tr_to_drop = np.append(tr_to_drop, np.where(labels == 'rest'))
		#delete those conditions

		labels = np.delete(labels, tr_to_drop)
		data = np.delete(data, tr_to_drop, axis=0)

		for i, label in enumerate(labels):
			if label == 'outdoor':
				labels[i] = 'scene'
	
		for i, label in enumerate(labels):
			if label == 'animal' or label == 'tool':
				labels[i] = 'stim'

		# for i, label in enumerate(labels):
		# 	if label == 'scrambled':
		# 		labels[i] = 'scene'

		return labels, data

	def remove_scrambled(self, labels=None, data=None):
		
		if self.verbose: print('removing scrambled scenes')

		tr_to_drop = np.array([])
		# tr_to_drop = np.append(tr_to_drop, np.where(labels == 'animal'))
		# tr_to_drop = np.append(tr_to_drop, np.where(labels == 'tool'))
		tr_to_drop = np.append(tr_to_drop, np.where(labels == 'scrambled'))
		# delete those conditions
		
		labels = np.delete(labels, tr_to_drop)
		data = np.delete(data, tr_to_drop, axis=0)

		# for i, label in enumerate(labels):
			# if label == 'scrambled':
				# labels[i] = 'scene'


		return labels, data

	def stim_binarize(self,labels=None,data=None):
		tr_to_drop = np.array([])
		tr_to_drop = np.append(tr_to_drop, np.where(labels == 'rest'))
		tr_to_drop = np.append(tr_to_drop, np.where(labels == 'indoor'))
		tr_to_drop = np.append(tr_to_drop, np.where(labels == 'scrambled'))
		tr_to_drop = np.append(tr_to_drop, np.where(labels == 'outdoor'))
		tr_to_drop = np.append(tr_to_drop, np.where(labels == 'scene'))

		labels = np.delete(labels, tr_to_drop)
		data = np.delete(data, tr_to_drop, axis=0)
		return labels, data

	def binarize(self,labels=None,data=None):

		tr_to_drop = np.array([])
		tr_to_drop = np.append(tr_to_drop, np.where(labels == 'animal'))
		tr_to_drop = np.append(tr_to_drop, np.where(labels == 'tool'))
		tr_to_drop = np.append(tr_to_drop, np.where(labels == 'rest'))
		tr_to_drop = np.append(tr_to_drop, np.where(labels == 'indoor'))
		labels = np.delete(labels, tr_to_drop)
		data = np.delete(data, tr_to_drop, axis=0)
		
		for i, label in enumerate(labels):
			if label == 'outdoor':
				labels[i] = 'scene'
		print(labels)
		return labels, data


	'''
	NO - we are not doing this
	'''
	# def load_old_labels(self):
	# 	out_labels = {phase: np.load('%s/%s/model/MVPA/labels/%s_labels.npy'%(data_dir,self.subj.fsub,phase)) for phase in ['localizer_1','localizer_2']}
	# 	out_labels = np.concatenate([out_labels['localizer_1'], out_labels['localizer_2']])
	# 	return out_labels
	
	def set_xval_groups(self, labels):
		

		xval_groups = np.zeros(len(labels))

		half = int(len(xval_groups)/2)

		xval_groups[0:half] = 1
		xval_groups[half:] = 2

		#something I need to check is how fixing the wholes in the label changes things
		return xval_groups


	def init_clasif(self, alg=None):

		if alg == 'logreg':
			alg = LogisticRegression()
		elif alg == 'svm':
			alg = LinearSVC()

		self.clf = Pipeline([ ('anova', SelectKBest(f_classif)), ('clf', alg) ])


	def run_xval(self, data=None, labels=None, xval_groups=None, k=0):

		self.clf.set_params(anova__k=k)
		xval_score = cross_val_score(self.clf, data, labels, groups=xval_groups, cv=2)
		# proba_res = { phase: self.clf.predict_proba(test_data[phase]) for phase in test_data }

		print(xval_score)
		return xval_score.mean()

	def create_output(self, _name=None, _index=None):

		self.res = pd.DataFrame([], columns=[_name], index=_index)


	#returns labels, data with rest TRs removed
	def remove_rest(self, labels=None, data=None):

		if self.verbose: print('Removing rest TRs')

		rest_ind = np.where(labels == 'rest')

		labels = np.delete(labels, rest_ind)

		data = np.delete(data, rest_ind, axis=0)

		return labels, data

	def collapse_scenes(self, labels=None):

		if self.verbose: print('Combining indoor and outdoor scenes')

		for i, label in enumerate(labels):

			if label == 'outdoor' or label == 'indoor':

				labels[i] = 'scene'

		return labels

	def confusion_matrices(self, labels=None, data=None, groups=None):

		cf_mats = dict.fromkeys(list(np.unique(groups)))
		cf_rep = {}
		for run in np.unique(groups):

			_res = self.clf.fit(data[groups == run], labels[groups == run]).predict(data[groups != run])

			# cf_mats[run] = confusion_matrix(labels[groups!=run], _res, list(np.unique(labels)))
			# cf_mats[run] = confusion_matrix(labels[groups!=run], _res, labels=['animal', 'tool', 'outdoor', 'indoor', 'scrambled', 'rest'])
			cf_mats[run] = confusion_matrix(labels[groups!=run], _res, labels=list(np.unique(labels)))
			
			cf_rep[run] = precision_recall_fscore_support(labels[groups != run], _res, labels=list(np.unique(labels)), average=None)[0]

		cf_mat_avg = np.mean( np.array( [cf_mats[1], cf_mats[2] ] ), axis=0 )
		cf_rep_avg = np.mean( (cf_rep[1],cf_rep[2]), axis=0)
		return cf_mat_avg, cf_rep_avg


	def prep_beta_vols(self, data=None):

			
			data = { phase: apply_mask(data[phase], self.subj.ctx_maskvol) for phase in data }

			#with new betas all preprocessing is already done
			#data = { phase: signal.clean(data[phase], detrend=True, standardize=True, t_r=2, ensure_finite=True) for phase in data }

			#beta estimates leave some constant features, remove those
			#reduce_var = VarianceThreshold()
			#data = { phase: reduce_var.fit_transform(data[phase]) for phase in data }
			
			return data

	def tr_shift(self, data=None):
		
		#shift over the beginning to make sure we are only decoding what we show them
		data = { phase: data[phase][hdr_shift['start']:, :] for phase in data}
		return data

	def decode_ts(self,data=None,labels=None,groups=None):
		self.ts_res = {}
		phases = ['localizer_1','localizer_2']
		proba_res = {}
		for i, run in enumerate(np.unique(groups)):
			phase = phases[i]
			self.clf.fit(data[groups == run],labels[groups == run])
			
			proba_res[phase] = self.clf.decision_function(data[groups != run])
			proba_res[phase] *= -1
			np.exp(proba_res[phase], proba_res[phase])
			proba_res[phase] += 1
			np.reciprocal(proba_res[phase], proba_res[phase])
			
			#multinomial
			# proba_res[phase] = self.clf.predict_proba(data[groups != run])

		for phase in proba_res:
			self.ts_res[phase] = {}
			for i, label in enumerate(self.clf.classes_):
				self.ts_res[phase][label] = proba_res[phase][:,i]

		# self.ts_res = np.mean(self.ts_res['localizer_1'],self.ts_res['localizer_2'])

	#roc curve a la signal theory
	def create_roc(self,labels=None,data=None,groups=None):
		
		# np.random.shuffle(bin_lab)
		res = {run: self.clf.fit(data[groups == run], labels[groups == run]).decision_function(data[groups != run]) for run in np.unique(groups)}

		bin_lab = label_binarize(labels,self.clf.classes_)

		res = np.concatenate([res[1],res[2]])

		auc = roc_auc_score(bin_lab, res, average=None)

		if len(self.clf.classes_) == 2:
			out = {'bin': auc}
		else:
			out = { label: score for label, score in zip(self.clf.classes_,auc) }

		return out


class eval_xval(object):
	
	def __init__(self, name=None, imgs=None, scene_DS=False, rmv_indoor=False, binarize=False, del_rest=False, scene_collapse=False, rmv_scram=False,save_dict=None, k=None, cf=False, p=False):

		if len(k) > 1 and cf:
			print('DONT COMPUTE MORE THAN ONE CONFUSION MATRIX AT A TIME DUMMY')
			sys.exit()


		self.res = pd.DataFrame([], columns=['acc','error','analysis'], index=k)

		self.cf_mat = None

		self.cf_rep = None

		self.cf_labels = None

		if p == True:
			self.sub_args = p_sub_args
			self.ptsd = True
		elif p == False:
			self.sub_args = sub_args
			self.ptsd = False
		if p == 'all':
			self.sub_args = all_sub_args
			self.ptsd = False


		self.ts = pd.DataFrame([],index=pd.MultiIndex.from_product([['localizer_1','localizer_2'],all_sub_args],names=['phase','sub']))

		self.sub_xval(name=name, imgs=imgs, scene_DS=scene_DS, rmv_indoor=rmv_indoor, binarize=binarize, del_rest=del_rest, scene_collapse=scene_collapse,save_dict=save_dict, rmv_scram=rmv_scram, k=k, cf=cf)



		self.res.analysis = name



	def sub_xval(self, name=None, imgs=None, del_rest=False, scene_DS=False, rmv_indoor=False, binarize=False, scene_collapse=False, save_dict=None,rmv_scram=False, k=None, cf=False):

		sub_res = pd.DataFrame([], columns=self.sub_args, index=k)
		group_res = {}
		group_auc = {}

		for sub in self.sub_args:

			sub_decode = loc_decode(subj=sub, name=name, imgs=imgs, rmv_indoor=rmv_indoor, scene_DS=scene_DS, binarize=binarize, del_rest=del_rest,save_dict=save_dict, scene_collapse=scene_collapse, rmv_scram=rmv_scram, k=k, cf=cf)
			sub_res[sub] = sub_decode.res

			if cf:
				if self.cf_mat is not None:

					self.cf_mat = self.cf_mat + sub_decode.cf_mat

				else:
				
					self.cf_mat = sub_decode.cf_mat


				if not self.cf_labels:

					self.cf_labels = list(np.unique(sub_decode.loc_lab))

				if self.cf_rep is None:

					self.cf_rep = pd.DataFrame([],index=self.sub_args, columns=self.cf_labels)

				self.cf_rep.loc[sub] = sub_decode.cf_rep

			if not binarize:
				group_res[sub] = {}
				for phase in ['localizer_1','localizer_2']:
					group_res[sub][phase] = {}				
					group_res[sub][phase] = { label: sub_decode.ts_res[phase][label] for label in sub_decode.ts_res[phase] }

			group_auc[sub] = sub_decode.auc

		self.group_df = pd.DataFrame.from_dict({(phase, sub): group_res[sub][phase]
												for sub in group_res.keys()
												for phase in group_res[sub].keys() }, orient='index')

		# for label in self.group_df.columns:

		# 	df = self.group_df[label]
		# 	df = df.reset_index()
		# 	df = df.set_index(['level_0','level_1'])[label].apply(pd.Series).stack().reset_index()
		# 	fig,ax = plt.subplots()
		# 	ax = sns.lineplot(x='level_2',y=0,hue='level_0',data=df)
		# 	plt.title(label)




		if cf:
			self.cf_mat = self.cf_mat / len(self.sub_args)

		self.res.acc = sub_res.mean(axis=1)
		self.res.error = sub_res.sem(axis=1)

		self.auc = pd.DataFrame.from_dict({(sub,label): group_auc[sub][label]
											for sub in group_auc.keys()
											for label in group_auc[sub].keys()}, orient='index')
		self.auc.reset_index(inplace=True)
		self.auc.rename({0:'auc'},axis=1,inplace=True)
		self.auc = pd.concat([self.auc['index'].apply(pd.Series), self.auc['auc']],axis=1)#columns=['subject','label','auc'])
		self.auc.rename({0:'subject',1:'label'},axis=1,inplace=True)


		fig, bx = plt.subplots()
		bx = sns.boxplot(x='label',y='auc',data=self.auc, palette='husl')
		sns.swarmplot(x='label',y='auc',data=self.auc,color='.25',alpha=.6)
		bx.axhline(y=.5,color='red',linestyle='--')
		bx.set_ylim(0,1)
		if save_dict == ppa_prepped: roi = 'PPA'
		else: roi = 'VTC'

		plt.title('%s; imgs = %s; ROI = %s'%(name,imgs,roi))
		# self.cf_out = pd.DataFrame([],index=self.cf_labels,columns=['avg','err'])
		# self.cf_out['avg'] = self.cf_rep.mean()
		# self.cf_out['err'] = self.cf_rep.sem()

		# self.cf_out.to_csv(os.path.join(data_dir,'graphing','classification_report.csv'))


















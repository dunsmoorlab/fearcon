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
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

from fc_config import data_dir, sub_args, init_dirs, nifti_paths, hdr_shift, mvpa_prepped, dataz, cv_iter, phase2rundir, PPA_fs_prepped
from preprocess_library import meta
from glm_timing import glm_timing


class loc_decode(object):
	#k must be an array of values to iter xval over
	def __init__(self, subj=0, imgs=None, name='no_name', del_rest=False, scene_collapse=False, k=None, cf=False):

		self.subj = meta(subj)
		print('Decoding %s'%(self.subj.fsub))

		self.loc_dat = self.load_localizer(imgs)

		self.loc_lab = self.better_tr_labels(data=self.loc_dat, imgs=imgs)

		#shift 3TRs
		self.loc_dat = { phase: self.loc_dat[phase][hdr_shift['start']:, :] for phase in self.loc_dat}
		#and concat
		self.loc_dat = np.concatenate([self.loc_dat['localizer_1'], self.loc_dat['localizer_2']])

		self.loc_lab, self.loc_dat = self.snr_hack(labels=self.loc_lab, data=self.loc_dat)


		print(self.loc_dat.shape)

		if del_rest and 'tr' in imgs:

			self.loc_lab, self.loc_dat = self.remove_rest(labels=self.loc_lab, data=self.loc_dat)

		if scene_collapse:

			self.loc_lab = self.collapse_scenes(labels=self.loc_lab)

		self.loc_groups = self.set_xval_groups(labels=self.loc_lab)

		self.init_clasif(alg='logreg')

		self.create_output(_name=name,_index=cv_iter)

		for k in k:

			print('Running xval, k=%s'%(k))
			self.res.loc[k] = self.run_xval(data=self.loc_dat, labels=self.loc_lab, xval_groups=self.loc_groups, k=k)

			if cf:

				self.cf_mat = self.confusion_matrices(labels=self.loc_lab, data=self.loc_dat, groups=self.loc_groups)

	#takes in 'tr' or 'beta' for imgs
	def load_localizer(self, imgs):


		#if you want to load in the tr by tr data (not beta)
		if imgs == 'tr':
			print('loading TR localizer data')

			#load in the motion corrected niftis
			loc_dat = { phase: np.load('%s%s'%(self.subj.bold_dir, mvpa_prepped[phase]))[dataz] for phase in mvpa_prepped if 'localizer' in phase }

			return loc_dat


		elif imgs == 'beta':

			print('loading beta estimates localizer data')

			loc_dat = { phase: nib.concat_images(glob('%s%sls-s_betas/beta_***.nii.gz'%(self.subj.bold_dir, phase2rundir[phase]))) for phase in phase2rundir if 'localizer' in phase }

			loc_dat = self.prep_beta_vols(data=loc_dat)

			return loc_dat

	def better_tr_labels(self,data=None, imgs=None):

		print('Making labels')
		out_labels = dict.fromkeys(list(data.keys()))

		if imgs == 'beta':
			out_labels = {phase: glm_timing(self.subj.num, phase).phase_events(con=True) for phase in data}

		elif 'tr' in imgs:
			shift = hdr_shift['start']

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

		out_labels = {phase: out_labels[phase][hdr_shift['start']:] for phase in out_labels}
		out_labels = np.concatenate([out_labels['localizer_1'], out_labels['localizer_2']])

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

			#if the comp window matches the patter i want but hard to do that

			window3 += 1

		#5window loop
		window5 = np.array(range(5))

		for i in range(len(_labels)-5):

			if np.array_equal(_labels[window5], rest3_target):

				tr_to_drop = np.append(tr_to_drop, [window5[1], window5[3]])

			if np.array_equal(_labels[window5], rest5_target):

				tr_to_drop = np.append(tr_to_drop, np.concatenate([window5[:2], window5[-2:]]))

			window5 += 1


		tr_to_drop = tr_to_drop.astype(int)

		print('deleting %s in hopes that it increases decoding...'%(len(tr_to_drop)))

		#delete those conditions
		labels = np.delete(labels, tr_to_drop)
		
		data = np.delete(data, tr_to_drop, axis=0)

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
		tr_to_drop = np.append(tr_to_drop, np.where(labels == 'indoor'))

		print('deleting %s in hopes that it increases decoding...'%(len(tr_to_drop)))

		#delete those conditions
		labels = np.delete(labels, tr_to_drop)
		
		data = np.delete(data, tr_to_drop, axis=0)

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

		return xval_score.mean()

	def create_output(self, _name=None, _index=None):

		self.res = pd.DataFrame([], columns=[_name], index=_index)


	#returns labels, data with rest TRs removed
	def remove_rest(self, labels=None, data=None):

		print('Removing rest TRs')

		rest_ind = np.where(labels == 'rest')

		labels = np.delete(labels, rest_ind)

		data = np.delete(data, rest_ind, axis=0)

		return labels, data

	def collapse_scenes(self, labels=None):

		print('Combining indoor and outdoor scenes')

		for i, label in enumerate(labels):

			if label == 'outdoor' or label == 'indoor':

				labels[i] = 'scene'

		return labels

	def confusion_matrices(self, labels=None, data=None, groups=None):

		cf_mats = dict.fromkeys(list(np.unique(groups)))

		for run in np.unique(groups):

			_res = self.clf.fit(data[groups == run], labels[groups == run]).predict(data[groups != run])

			cf_mats[run] = confusion_matrix(labels[groups!=run], _res, list(np.unique(labels)))

		return np.mean( np.array( [cf_mats[1], cf_mats[2] ] ), axis=0 )


	def prep_beta_vols(self, data=None):

			data = { phase: apply_mask(data[phase], self.subj.maskvol) for phase in data }

			data = { phase: signal.clean(data[phase], detrend=True, standardize=True, t_r=2, ensure_finite=True) for phase in data }

			#beta estimates leave some constant features, remove those
			#reduce_var = VarianceThreshold()
			#data = { phase: reduce_var.fit_transform(data[phase]) for phase in data }
			
			return data

	#roc curve a la signal theory
	def create_roc(self):
		pass

class eval_xval(object):
	
	def __init__(self, name=None, imgs=None, del_rest=False, scene_collapse=False, k=None, cf=False):

		if len(k) > 1 and cf:
			print('DONT COMPUTE MORE THAN ONE CONFUSION MATRIX AT A TIME DUMMY')
			sys.exit()


		self.res = pd.DataFrame([], columns=['acc','error','analysis'], index=k)

		self.cf_mat = None

		self.cf_labels = None

		self.sub_xval(name=name, imgs=imgs, del_rest=del_rest, scene_collapse=scene_collapse, k=k, cf=cf)

		self.res.analysis = name



	def sub_xval(self, name=None, imgs=None, del_rest=False, scene_collapse=False, k=None, cf=False):

		sub_res = pd.DataFrame([], columns=sub_args, index=k)

		for sub in sub_args:

			sub_decode = loc_decode(subj=sub, name=name, imgs=imgs, del_rest=del_rest, scene_collapse=scene_collapse, k=k, cf=cf)
			sub_res[sub] = sub_decode.res

			if cf:
				if self.cf_mat is not None:

					self.cf_mat = self.cf_mat + sub_decode.cf_mat

				else:
				
					self.cf_mat = sub_decode.cf_mat


				if not self.cf_labels:

					self.cf_labels = list(np.unique(sub_decode.loc_lab))
		if cf:
			self.cf_mat = self.cf_mat / len(sub_args)

		self.res.acc = sub_res.mean(axis=1)
		self.res.error = sub_res.sem(axis=1)






















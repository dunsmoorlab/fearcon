from mvpa_analysis import *
import numpy as np
import pandas as pd
import seaborn as sns
from fc_config import *
from wesanderson import wes_palettes
from preprocess_library import *
from glm_timing import *

class rsa():
	def __init__(self,sub=None,save_dict=None,fs=False):

		self.subj = meta(sub)
		self.verbose=False
		rsa_phases = ['baseline','fear_conditioning','extinction','extinction_recall']

		self.data = {phase: np.load(self.subj.bold_dir + save_dict[phase])[dataz] for phase in rsa_phases}
		self.labels = {phase: glm_timing(sub,phase).phase_events(con=True) for phase in rsa_phases} 

		self.loc_dat, self.loc_lab = decode.load_localizer(self, imgs='beta', save_dict=save_dict, binarize=False,
						SC=True, S_DS=True, rmv_rest=False, rmv_scram=False, rmv_ind=False)
		print(self.loc_dat.shape)
		if fs:
			self.feature_reduction(k=fs,train_dat=self.loc_dat,train_lab=self.loc_lab,reduce_dat=self.data)
		
		self.reorder_data(labels=self.labels,data=self.data)
		self.z = self.compute_rsa(data=self.data)

	def feature_reduction(self,k=None,train_dat=None,train_lab=None,reduce_dat=None):

		classif_alg = LogisticRegression()

		feature_selection = SelectKBest(f_classif,k)

		self.clf = Pipeline([('anova', feature_selection), ('alg', classif_alg)])

		self.clf.fit(train_dat, train_lab)

		#reshape the test_data to 1000 voxels using the localizer data
		for phase in reduce_dat: reduce_dat[phase] = feature_selection.fit(train_dat, train_lab).transform(reduce_dat[phase])

	def reorder_data(self,labels=None,data=None):

		self.csplus_index = { phase: [i for i, label in enumerate(labels[phase]) if 'CS+' in label] for phase in labels }
		#get index of csminus
		self.csmin_index = { phase: [i for i, label in enumerate(labels[phase]) if 'CS-' in label] for phase in labels }

		new_data = {phase: np.ndarray(shape=data[phase].shape) for phase in data}

		for phase in self.csplus_index:
			for i, trial in enumerate(self.csplus_index[phase]):
				new_data[phase][i] = data[phase][self.csplus_index[phase][i]]
			for i, trial in enumerate(self.csmin_index[phase]):
				new_data[phase][i+int(data[phase].shape[0]/2)] = data[phase][self.csmin_index[phase][i]]

		self.data = new_data

	def compute_rsa(self,data=None):

		bigmat = np.concatenate((self.data['baseline'],self.data['fear_conditioning'],self.data['extinction'],self.data['extinction_recall']))

		#change the middle to something thats not 1 or inf
		r = np.corrcoef(bigmat)
		r[np.where(np.eye(r.shape[0],dtype=bool))] = 0
		
		z = np.arctanh(r)

		# off = np.where(~np.eye(z.shape[0],dtype=bool))

		# out = np.zeros(shape=z.shape)
		


		# for side in off:
		# 	out[side] = np.mean(z[])

		return z


class group_rsa():
	def __init__(self,subs,save_dict,fs=False):

		self.group_rs = np.ndarray(shape=(len(subs),168,168))

		for i,sub in enumerate(subs):
			print(sub)
			
			self.group_rs[i] = rsa(sub=sub,save_dict=save_dict,fs=fs).z

		self.sub_stats()

	def sub_stats(self,f=False):
		if f: end = 149
		else: end = 156
		
		n_sub = self.group_rs.shape[0]

		self.ind_stats = {}
		for phase in ['baseline','fear_conditioning','extinction','extinction_recall']:self.ind_stats[phase] = np.zeros(n_sub)

		for i in range(n_sub):
			self.ind_stats['baseline'][i] = self.group_rs[i,144:end,0:24].mean()
			self.ind_stats['fear_conditioning'][i] = self.group_rs[i,144:end,48:72].mean()
			self.ind_stats['extinction'][i] = self.group_rs[i,144:end,96:120].mean()
			self.ind_stats['extinction_recall'][i] = self.group_rs[i,144:end,144:156].mean()






from mvpa_analysis import *
import numpy as np
import pandas as pd
import seaborn as sns
from fc_config import *
from wesanderson import wes_palettes
from preprocess_library import *
from glm_timing import *

class rsa():
	def __init__(self,sub=None,save_dict=None,fs=False,weights='copes',mask=None):

		self.subj = meta(sub)
		self.verbose=False
		self.rsa_phases = ['baseline','fear_conditioning','extinction','extinction_recall']
		self.rsa_labels = ['baseline','fear','ext','early_rnw','late_rnw']
		self.rsa_split_labels = ['baseline','early_fear','late_fear','early_ext','late_ext','early_rnw','late_rnw']

		self.weight_dir = os.path.join(data_dir,'rsa_%s'%(weights))

		self.mask = mask
		self.load_data()

		# self.loc_dat, self.loc_lab = decode.load_localizer(self, imgs='beta', save_dict=save_dict, stim_binarize=False, binarize=False,
		# 				SC=True, S_DS=True, rmv_rest=False, rmv_scram=True, rmv_ind=False)
		# print(self.loc_dat.shape)
		if fs:
			self.feature_reduction(k=fs,train_dat=self.loc_dat,train_lab=self.loc_lab,reduce_dat=self.data)

		# self.data = self.reorder_data(labels=self.labels,data=self.data)
		self.z = self.compute_rsa(data=self.data)
		self.sz = self.compute_rsa(data=self.split_data)

	def load_data(self):

		# self.data = {phase: np.load(self.subj.bold_dir + save_dict[phase])[dataz] for phase in rsa_phases}
		# self.labels = {phase: glm_timing(sub,phase).phase_events(con=True) for phase in self.rsa_phases} 
		mask_img = os.path.join(self.subj.roi,'%s_mask.nii.gz'%(self.mask))
		self.masker = NiftiMasker(mask_img=mask_img, standardize=False, memory='nilearn_cache',memory_level=2)
		self.W_masker = NiftiMasker(mask_img=mask_img)
		
		self.data = {}
		self.split_data = {}
		self.sub_dir = os.path.join(self.weight_dir,self.subj.fsub)
		for phase in self.rsa_phases:				
			print(phase)
			#load in data
			beta_img = os.path.join(self.subj.bold_dir, fsl_betas[phase])
			#fit the mask
			beta_img = self.masker.fit_transform(beta_img)
			#get trial type information
			run_labels = glm_timing(self.subj.num,phase).phase_events(con=True)
			#clean out CS- trials
			beta_img = np.delete(beta_img, np.where(run_labels == 'CS-')[0],axis=0)
			#save it to the train data
			#BE CAREFUL WITH YOUR COPYS FUCK0
			self.data[phase] = np.copy(beta_img)
			self.split_data[phase] = np.copy(beta_img)

		#load in the weights
		self.W = {}
		for label in self.rsa_labels:
			print(label)
			self.W[label] = self.W_masker.fit_transform(os.path.join(self.sub_dir,'%s_csp.nii.gz'%(label)))
		
		#apply the weights
		self.data['baseline']                 *= self.W['baseline']
		self.data['fear_conditioning']        *= self.W['fear'] 
		self.data['extinction']               *= self.W['ext'] 
		self.data['extinction_recall'][:4,:]  *= self.W['early_rnw']
		self.data['extinction_recall'][4:,:]  *= self.W['late_rnw']
		
		#do again for the split data
		self.split_W = {}
		for label in self.rsa_split_labels:
			print(label)
			self.split_W[label] = self.W_masker.fit_transform(os.path.join(self.sub_dir,'%s_csp.nii.gz'%(label)))
		#apply the weights
		self.split_data['baseline']                 *= self.split_W['baseline']
		self.split_data['fear_conditioning'][:12,:] *= self.split_W['early_fear'] 
		self.split_data['fear_conditioning'][12:,:] *= self.split_W['late_fear'] 
		self.split_data['extinction'][:12,:]        *= self.split_W['early_ext'] 
		self.split_data['extinction'][12:,:]        *= self.split_W['late_ext'] 
		self.split_data['extinction_recall'][:4,:]  *= self.split_W['early_rnw']
		self.split_data['extinction_recall'][4:,:]  *= self.split_W['late_rnw']


		self.sub_labels = np.concatenate((np.repeat('baseline',24),
					                 np.repeat('fear',24),np.repeat('ext',24),
									 np.repeat('early_rnw',4),np.repeat('late_rnw',8)))

		self.sub_split_labels = np.concatenate((np.repeat('baseline',24),
					                 np.repeat('early_fear',12),np.repeat('late_fear',12),
									 np.repeat('early_ext',12),np.repeat('late_ext',12),
									 np.repeat('early_rnw',4),np.repeat('late_rnw',8)))


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

		return new_data

	def compute_rsa(self,data=None):

		bigmat = np.concatenate((data['baseline'],data['fear_conditioning'],data['extinction'],data['extinction_recall']))
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
	def __init__(self,subs,mask,fs=False):
		
		self.sub_labels = np.concatenate((np.repeat('baseline',24),
					                 np.repeat('fear',24),np.repeat('ext',24),
									 np.repeat('early_rnw',4),np.repeat('late_rnw',8)))

		self.sub_split_labels = np.concatenate((np.repeat('baseline',24),
					                 np.repeat('early_fear',12),np.repeat('late_fear',12),
									 np.repeat('early_ext',12),np.repeat('late_ext',12),
									 np.repeat('early_rnw',4),np.repeat('late_rnw',8)))

		self.collect_sub(subs=subs,mask=mask,fs=fs)
		self.sub_stats()

	def collect_sub(self,subs=None,mask=None,fs=False):	

		self.group_rs = np.ndarray(shape=(len(subs),len(self.sub_labels),len(self.sub_labels)))
		self.group_split_rs = np.ndarray(shape=(len(subs),len(self.sub_split_labels),len(self.sub_split_labels)))
		
		for i,sub in enumerate(subs):
			print(sub)

			sub_rsa = rsa(sub=sub,mask=mask,fs=fs)
			self.group_rs[i] = sub_rsa.z
			self.group_split_rs[i] = sub_rsa.sz

	def sub_stats(self,f=False):
		# if f: end = 149
		# else: end = 156
		
		# b_p = slice(0,24)
		# f_p = slice(48,72)
		# e_p = slice(96,120)
		# r_p = slice(144,end)

		b = slice(0,24)
		f = slice(24,48)
		e = slice(48,72)
		e_f = slice(24,36)
		l_f = slice(36,48)
		e_e = slice(48,60)
		l_e = slice(60,72)
		e_r = slice(72,76)
		l_r = slice(76,84)

		n_sub = self.group_rs.shape[0]

		self.stats = {}
		for phase in ['b__b','f__f','e__e','e_r__e_r','l_r__l_r','b__f','b__e','b__e_r','b__l_r','f__e','f__e_r','f__l_r','e__e_r','e__l_r','e_r__l_r']:self.stats[phase] = np.zeros(n_sub)
		for i in range(n_sub):
			if len(np.where(self.group_rs[i] == 0)[0]) != len(self.sub_labels):print('WARNING: EXTRA ABSOLUTE 0 DETECTED')

			#within phase
			self.stats['b__b'][i] = self.group_rs[i,b,b][np.triu(self.group_rs[i,b,b]) != 0].mean()
			self.stats['f__f'][i] = self.group_rs[i,f,f][np.triu(self.group_rs[i,f,f]) != 0].mean()
			self.stats['e__e'][i] = self.group_rs[i,e,e][np.triu(self.group_rs[i,e,e]) != 0].mean()
			self.stats['e_r__e_r'][i] = self.group_rs[i,e_r,e_r][np.triu(self.group_rs[i,e_r,e_r]) != 0].mean()
			self.stats['l_r__l_r'][i] = self.group_rs[i,l_r,l_r][np.triu(self.group_rs[i,l_r,l_r]) != 0].mean()

			#across phase
			self.stats['b__f'][i]   = self.group_rs[i,b,f].mean()
			self.stats['b__e'][i]   = self.group_rs[i,b,e].mean()
			self.stats['b__e_r'][i]   = self.group_rs[i,b,e_r].mean()
			self.stats['b__l_r'][i]   = self.group_rs[i,b,l_r].mean()
			self.stats['f__e'][i]   = self.group_rs[i,f,e].mean()
			self.stats['f__e_r'][i]   = self.group_rs[i,f,e_r].mean()
			self.stats['f__l_r'][i]   = self.group_rs[i,f,l_r].mean()
			self.stats['e__e_r'][i]   = self.group_rs[i,e,e_r].mean()
			self.stats['e__l_r'][i]   = self.group_rs[i,e,l_r].mean()
			self.stats['e_r__l_r'][i]   = self.group_rs[i,e_r,l_r].mean()


		self.split_stats = {}
		# for phase in ['baseline','fear_conditioning','extinction','renewal','b_f','b_e','f_e','r_b','r_f','r_e']:self.split_stats[phase] = np.zeros(n_sub)
		for phase in ['b__b','e_f__e_f','l_f__l_f','e_e__e_e','l_e__l_e','e_r__e_r','l_r__l_r','b__e_f','b__l_f','b__e_e','b__l_e','b__e_r','b__l_r','e_f__l_f','e_f__e_e','e_f__l_e','e_f__e_r','e_f__l_r','l_f__e_e','l_f__l_e','l_f__e_r','l_f__l_r','e_e__l_e','e_e__e_r','e_e__l_r','l_e__e_r','l_e__l_r','e_r__l_r']: self.split_stats[phase] = np.zeros(n_sub)
		for i in range(n_sub):
			if len(np.where(self.group_rs[i] == 0)[0]) != len(self.sub_labels):print('WARNING: EXTRA ABSOLUTE 0 DETECTED')

			#within phase			
			self.split_stats['b__b'][i]     = self.group_split_rs[i,b,b][np.triu(self.group_split_rs[i,b,b]) != 0].mean()
			self.split_stats['e_f__e_f'][i] = self.group_split_rs[i,e_f,e_f][np.triu(self.group_split_rs[i,e_f,e_f]) != 0].mean()
			self.split_stats['l_f__l_f'][i] = self.group_split_rs[i,l_f,l_f][np.triu(self.group_split_rs[i,l_f,l_f]) != 0].mean()
			self.split_stats['e_e__e_e'][i] = self.group_split_rs[i,e_e,e_e][np.triu(self.group_split_rs[i,e_e,e_e]) != 0].mean()
			self.split_stats['l_e__l_e'][i] = self.group_split_rs[i,l_e,l_e][np.triu(self.group_split_rs[i,l_e,l_e]) != 0].mean()
			self.split_stats['e_r__e_r'][i] = self.group_split_rs[i,e_r,e_r][np.triu(self.group_split_rs[i,e_r,e_r]) != 0].mean()
			self.split_stats['l_r__l_r'][i] = self.group_split_rs[i,l_r,l_r][np.triu(self.group_split_rs[i,l_r,l_r]) != 0].mean()
			
			#across phase
			self.split_stats['b__e_f'][i]   = self.group_split_rs[i,b,e_f].mean()
			self.split_stats['b__l_f'][i]   = self.group_split_rs[i,b,l_f].mean()
			self.split_stats['b__e_e'][i]   = self.group_split_rs[i,b,e_e].mean()
			self.split_stats['b__l_e'][i]   = self.group_split_rs[i,b,l_e].mean()
			self.split_stats['b__e_r'][i]   = self.group_split_rs[i,b,e_r].mean()
			self.split_stats['b__l_r'][i]   = self.group_split_rs[i,b,l_r].mean()
			self.split_stats['e_f__l_f'][i] = self.group_split_rs[i,e_f,l_f].mean()
			self.split_stats['e_f__e_e'][i] = self.group_split_rs[i,e_f,e_e].mean()
			self.split_stats['e_f__l_e'][i] = self.group_split_rs[i,e_f,l_e].mean() 
			self.split_stats['e_f__e_r'][i] = self.group_split_rs[i,e_f,e_r].mean() 
			self.split_stats['e_f__l_r'][i] = self.group_split_rs[i,e_f,l_r].mean() 
			self.split_stats['l_f__e_e'][i] = self.group_split_rs[i,l_f,e_e].mean() 
			self.split_stats['l_f__l_e'][i] = self.group_split_rs[i,l_f,l_e].mean() 
			self.split_stats['l_f__e_r'][i] = self.group_split_rs[i,l_f,e_r].mean() 
			self.split_stats['l_f__l_r'][i] = self.group_split_rs[i,l_f,l_r].mean() 
			self.split_stats['e_e__l_e'][i] = self.group_split_rs[i,e_e,l_e].mean() 
			self.split_stats['e_e__e_r'][i] = self.group_split_rs[i,e_e,e_r].mean() 
			self.split_stats['e_e__l_r'][i] = self.group_split_rs[i,e_e,l_r].mean() 
			self.split_stats['l_e__e_r'][i] = self.group_split_rs[i,l_e,e_r].mean() 
			self.split_stats['l_e__l_r'][i] = self.group_split_rs[i,l_e,l_r].mean() 
			self.split_stats['e_r__l_r'][i] = self.group_split_rs[i,e_r,l_r].mean() 





			# self.stats['baseline'][i] = self.group_rs[i,b_p,b_p][np.triu(self.group_rs[i,b_p,b_p]) != 0].mean()
			# self.stats['fear_conditioning'][i] = self.group_rs[i,f_p,f_p][np.triu(self.group_rs[i,f_p,f_p]) != 0].mean()
			# self.stats['extinction'][i] = self.group_rs[i,e_p,e_p][np.triu(self.group_rs[i,e_p,e_p]) != 0].mean()
			# self.stats['renewal'][i] = self.group_rs[i,r_p,r_p][np.triu(self.group_rs[i,r_p,r_p]) != 0].mean()

			# #across phase 
			# self.stats['b_f'][i] = self.group_rs[i,b_p,f_p].mean()
			# self.stats['b_e'][i] = self.group_rs[i,b_p,e_p].mean()
			# self.stats['f_e'][i] = self.group_rs[i,e_p,f_p].mean()

			# self.stats['r_b'][i] = self.group_rs[i,r_p,b_p].mean()
			# self.stats['r_f'][i] = self.group_rs[i,r_p,f_p].mean()
			# self.stats['r_e'][i] = self.group_rs[i,r_p,e_p].mean()

def rsa_graph(mask):
	c = group_rsa(sub_args,mask);p = group_rsa(p_sub_args,mask)

	group_stats = {'control':c.stats,
				   'ptsd':   p.stats}

	df = pd.DataFrame.from_dict({(group,label,i): group_stats[group][label][i]
													for group in group_stats.keys()
													for label in group_stats[group].keys()
													for i,sub in enumerate(group_stats[group][label])}, orient='index')
	df.reset_index(inplace=True)
	df.rename(columns={0:'z'},inplace=True)
	df = pd.concat((df['index'].apply(pd.Series),df['z']),axis=1)
	df.rename(columns={0:'group',1:'label',2:'sub'},inplace=True)
	df['sub'].loc[df.group == 'ptsd'] += 100
	fig, pp = plt.subplots()
	pp = sns.pointplot(x='label',y='z',data=df,hue='group',capsize=.08,join=False,dodge=True)
	plt.title(mask)

	group_split_stats = {'control':c.split_stats,
				   'ptsd':   p.split_stats}

	df2 = pd.DataFrame.from_dict({(group,label,i): group_split_stats[group][label][i]
													for group in group_split_stats.keys()
													for label in group_split_stats[group].keys()
													for i,sub in enumerate(group_split_stats[group][label])}, orient='index')
	df2.reset_index(inplace=True)
	df2.rename(columns={0:'z'},inplace=True)
	df2 = pd.concat((df2['index'].apply(pd.Series),df2['z']),axis=1)
	df2.rename(columns={0:'group',1:'label',2:'sub'},inplace=True)
	df2['sub'].loc[df2.group == 'ptsd'] += 100
	fig, pp2 = plt.subplots()
	pp2 = sns.pointplot(x='label',y='z',data=df2,hue='group',capsize=.08,join=False,dodge=True)
	plt.title(mask)





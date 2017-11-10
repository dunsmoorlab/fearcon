#this script does the real MVPA classification of the extinction recall data

import os
import argparse
import numpy as np
import nibabel as nib
import scipy as sp
import pandas as pd
import sys
import matplotlib.pyplot as plt
from glob import glob
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectKBest, SelectPercentile, f_classif
from sklearn.pipeline import make_pipeline, Pipeline 
from matplotlib.backends.backend_pdf import PdfPages


#import the local config file
from fc_config import run_key
from fc_config import mvpa_prepped

#import function to plot confusion matrices
from cf_mat_plot import plot_confusion_matrix

#def dive(phase=''):

data_dir = '/Users/ach3377/GoogleDrive/FC_FMRI_DATA/'

sub_args = [int(subs[3:]) for subs in os.listdir(data_dir) if 'Sub' in subs and 'fs' not in subs]

#make some results structures
mean_15 = pd.DataFrame([],columns=['Subject','extinction','baseline','fear_conditioning','extinction_recall'])
mean_15.Subject = sub_args 

csplus_out = pd.DataFrame([],columns=['CS','Subject','Phase','-4','-2','0','2'])
csplus_out.Phase = ['extinction','baseline','fear_conditioning','extinction_recall'] * len(sub_args)
csplus_out.Subject = np.repeat(sub_args,4)
csplus_out['CS'] = 'CS+'
csplus_out = csplus_out.set_index(['CS','Subject','Phase'])

csmin_out = pd.DataFrame([],columns=['CS','Subject','Phase','-4','-2','0','2'])
csmin_out.Phase = ['extinction','baseline','fear_conditioning','extinction_recall'] * len(sub_args)
csmin_out.Subject = np.repeat(sub_args,4)
csmin_out['CS'] = 'CS-'
csmin_out = csmin_out.set_index(['CS','Subject','Phase'])

comb_out = pd.DataFrame([], columns = ['Subject','Phase','-4','-2','0','2'])
comb_out.Phase = ['extinction','baseline','fear_conditioning','extinction_recall'] * len(sub_args)
comb_out.Subject = np.repeat(sub_args, 4)
comb_out = comb_out.set_index(['Subject','Phase'])

for sub in sub_args:
	SUBJ = 'Sub{0:0=3d}'.format(sub)

	#start with the localizer runs
	loc_runs = ['localizer_1','localizer_2']
	#loading in data
	loc_data = { phase: np.load('%s/%s/bold/%s'%(data_dir,SUBJ,mvpa_prepped[phase])) for phase in loc_runs }
	#and labels
	loc_labels = { phase: np.load('%s/%s/model/MVPA/labels/%s_labels.npy'%(data_dir,SUBJ,phase)) for phase in loc_runs }

	#shift over 3 for HDR
	hdr_shift = 3
	end_shift = 6 - hdr_shift
	
	loc_data = {phase: loc_data[phase][hdr_shift:-end_shift] for phase in loc_data}

	#find the rest TRs
	loc_rest_index = {phase: np.where(loc_labels[phase]=='rest') for phase in loc_labels}

	#delete rest from the localizers
	for phase in loc_data.keys():
			loc_data[phase] = np.delete(loc_data[phase], loc_rest_index[phase], axis=0)

	#as well as the labels
	for phase in loc_labels.keys():
			loc_labels[phase] = np.delete(loc_labels[phase], loc_rest_index[phase], axis=0)

	#collapse the localizer runs to be scenes and not scenes
	for phase in loc_labels.keys():
			for i, label in enumerate(loc_labels[phase]):
				if label == 'indoor' or label == 'outdoor':
					loc_labels[phase][i] = 'scene'
				else:
					loc_labels[phase][i] = 'not'
	
	#combine localizer runs to fit the model
	loc_data = np.concatenate([loc_data['localizer_1'], loc_data['localizer_2']])

	#combine localizer labels
	loc_labels = np.concatenate([loc_labels['localizer_1'] ,loc_labels['localizer_2']])

	
	#set the classifier
	alg = LogisticRegression()

	#and the feature selection
	feature_selection = SelectKBest(f_classif, k=1000)
	#feature_selection = SelectPercentile(f_classif, percentile=15)

	#make the classifier pipeline
	clf = Pipeline([('anova', feature_selection), ('alg', alg)])

	#fit the classifier to the localizer data
	clf.fit(loc_data, loc_labels)


	#now load in the data to test on
	test_runs = ['extinction','baseline','fear_conditioning','extinction_recall']
	#read numpy arrays
	test_data = { phase: np.load('%s/%s/bold/%s'%(data_dir,SUBJ,mvpa_prepped[phase])) for phase in test_runs }

	#shift test data over for HDR
	test_data = { phase: test_data[phase][hdr_shift:-end_shift] for phase in test_data }


	#predict_proba returns classifier evidence in the order of ['not','scene']
	clf_res = {phase: clf.predict_proba(test_data[phase])[:,1] for phase in test_runs}

	#collect the mean scene evidence for the first 15 TRs as a general measure of mental context at the beginning of each phase
	for phase in clf_res:
		mean_15.loc[mean_15.Subject==sub, phase] = clf_res[phase][0:15].mean()


	#load in the TR labels
	test_labels = { phase: np.load('%s/%s/model/MVPA/labels/%s_labels.npy'%(data_dir,SUBJ,phase)) for phase in test_data }

	#using the TR labels, create a temporal mask with 1s in a 4TR window around the start of a CS+ stim, 2 for CS-
	for phase in test_labels:
		if phase == 'extinction':
			for i, label in enumerate(test_labels[phase]):
				if label == 'CS+':
					if test_labels[phase][i-1] == 'scene':
						test_labels[phase][i-2:i+2] = 1
				if label == 'CS-':
					if test_labels[phase][i-1] == 'scene':
						test_labels[phase][i-2:i+2] = 2
		else:
			for i, label in enumerate(test_labels[phase]):
				if label == 'CS+':
					if test_labels[phase][i-1] == 'rest':
						test_labels[phase][i-2:i+2] = 1
				if label == 'CS-':
					if test_labels[phase][i-1] == 'rest':
						test_labels[phase][i-2:i+2] = 2

	#go back and set everything else to 0
	for phase in test_labels:
		for i, label in enumerate(test_labels[phase]):
			if label != '1' and label != '2':
				test_labels[phase][i] = 0


	#collect just the csplus and csmin results
	csplus_res = { phase: clf_res[phase][np.where(test_labels[phase] == '1')[0]] for phase in test_labels }
	csmin_res = { phase: clf_res[phase][np.where(test_labels[phase] == '2')[0]] for phase in test_labels }

	#and transform them to be one trial to a row
	csplus_res = {phase: np.reshape(csplus_res[phase], (-1,4)) for phase in csplus_res}
	csmin_res = {phase: np.reshape(csmin_res[phase], (-1,4)) for phase in csmin_res}

	#then combine them for all trails in a phase
	all_res = { phase: np.concatenate([csplus_res[phase], csmin_res[phase]]) for phase in test_labels }

	for phase in test_labels:
		for trial in csplus_res[phase]:
			csplus_out['-4']['CS+'][sub][phase] = trial[0]
			csplus_out['-2']['CS+'][sub][phase] = trial[1]
			csplus_out['0']['CS+'][sub][phase] = trial[2]
			csplus_out['2']['CS+'][sub][phase] = trial[3]

		for trial in csmin_res[phase]:
			csmin_out['-4']['CS-'][sub][phase] = trial[0]
			csmin_out['-2']['CS-'][sub][phase] = trial[1]
			csmin_out['0']['CS-'][sub][phase] = trial[2]
			csmin_out['2']['CS-'][sub][phase] = trial[3]

		for trial in all_res[phase]:
			comb_out['-4'][sub][phase] = trial[0]
			comb_out['-2'][sub][phase] = trial[1]
			comb_out['0'][sub][phase] = trial[2]
			comb_out['2'][sub][phase] = trial[3]


event_out = pd.DataFrame([], columns = ['Phase','CS','TR','Mean','SEM'])
event_out.Phase = np.repeat(['extinction','baseline','fear_conditioning','extinction_recall'],8) 
event_out.CS = np.tile(np.concatenate([np.repeat('CS+',4),np.repeat('CS-',4)]),4)
event_out.TR = np.tile(['-4','-2','0','2'],8)
event_out = event_out.set_index(['CS','Phase','TR'])

for phase in test_runs:
	event_out.loc[('CS+',phase),'Mean'][:] = csplus_out.xs(phase, level='Phase').mean()
	event_out.loc[('CS+',phase),'SEM'][:] = csplus_out.xs(phase, level='Phase').sem()

	event_out.loc[('CS-',phase), 'Mean'][:] = csmin_out.xs(phase, level='Phase').mean()
	event_out.loc[('CS-',phase), 'SEM'][:] = csmin_out.xs(phase, level='Phase').sem()

event_out.to_csv('%s/graphing/sklearn_dive/event_cs_scene.csv'%(data_dir), sep=',')

tr = ['-4','-2','0','2']
all_out = pd.DataFrame([], columns= ['Phase','TR','Mean','SEM'])
all_out.Phase = np.repeat(['extinction','baseline','fear_conditioning','extinction_recall'],4) 
all_out.TR = np.tile(tr,4)
all_out = all_out.set_index(['Phase','TR'])

for phase in test_runs:
	for time in tr:
		all_out.loc[(phase,time),'Mean'] = comb_out.xs(phase, level='Phase').mean()[time]
		all_out.loc[(phase,time),'SEM'] = comb_out.xs(phase, level='Phase').sem()[time]

all_out.to_csv('%s/graphing/sklearn_dive/event_scene.csv'%(data_dir), sep=',')


Rgraph = pd.DataFrame([], columns = ['Phase','Mean','SEM'])
Rgraph.Phase = ['extinction','baseline','fear_conditioning','extinction_recall']
#THIS RIGHT HERE IS WHAT PREVENTS ALL THE COPY WARNINGS
#ITS ALL ABOUT THE INDICES BABY
Rgraph = Rgraph.set_index('Phase')

Rgraph.Mean = mean_15.mean()[1:5]
Rgraph.SEM = mean_15.sem()[1:5]

Rgraph.to_csv('%s/graphing/sklearn_dive/evidence_15tr.csv'%(data_dir), sep=',')


mean_15.to_csv('%s/graphing/sklearn_dive/subject_evidence_15tr.csv'%(data_dir), sep=',')









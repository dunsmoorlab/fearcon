#the goal of this script is to run MVPA classification via scikit learn

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

#build the python arg_parser
parser = argparse.ArgumentParser(description='Function arguments')

#add arguments
parser.add_argument('-s', '--subj', nargs='+', help='Subject Number', default=0, type=str)
#parser.add_argument('-r','--runs', nargs='+', help='Load prepped data and labels for these runs', default='', type=str)
parser.add_argument('-fs', '--feature_selection', help='Feature Selection', default='', type=str)
parser.add_argument('-c', '--classifier', help='Machine Learning Algorithm Used', default='', type=str)
#parser.add_argument('-cat', '--categories', help='4 or 5 categories, collapsing across scenes', default='', type=str)
#parse them
args=parser.parse_args()

if sys.platform == 'linux':
	#point bash to the folder with the subjects in
	data_dir = '/mnt/c/Users/ACH/Google Drive/FC_FMRI_DATA/'
elif sys.platform == 'win32':
	data_dir = 'C:\\Users\\ACH\\Google Drive\\FC_FMRI_DATA'
#but mostly it runs on a school mac
else:
	data_dir = '/Users/ach3377/GoogleDrive/FC_FMRI_DATA/'


#change this when you want to work with other runs
loc_runs = ['localizer_1','localizer_2']


mean_out = pd.DataFrame([])

if args.subj == ['all']:
	sub_args = [int(subs[3:]) for subs in os.listdir(data_dir) if 'Sub' in subs and 'fs' not in subs]
else:
	sub_args = args.subj


for sub in sub_args:
	sub_out = []


	sub = int(sub)
	SUBJ = 'Sub{0:0=3d}'.format(sub)

	#load in the prepped data for each run
	data = {phase: np.load('%s/%s/bold/%s'%(data_dir,SUBJ,mvpa_prepped[phase])) for phase in loc_runs}

	#as well as the labels
	labels = {phase: np.load('%s/%s/model/MVPA/labels/%s_labels.npy'%(data_dir,SUBJ,phase)) for phase in loc_runs}

	#shift over 2 for HDR
	hdr_shift = 3
	end_shift = 6 - hdr_shift

	data = {phase: data[phase][hdr_shift:-end_shift] for phase in data.keys()}

	#qualitiy control step to make sure the size of the labels matches the data for each phase
	{ (print('ERROR: data/label size mismatch for %s'%(phase)), sys.exit()) for phase in data.keys() if data[phase].shape[0] != labels[phase].shape[0]  }

	#take out rest
	rest_index = {phase: np.where(labels[phase]=='rest') for phase in labels.keys()}


	#better to collect the data for each analysis
	nr4_data = {phase: data[phase] for phase in data.keys()}
	nr4_labels = {phase: labels[phase] for phase in labels.keys()}

	#delete rest from the data
	for phase in nr4_data.keys():
		if 'localizer' in phase:
			nr4_data[phase] = np.delete(nr4_data[phase], rest_index[phase], axis=0)

	#as well as the labels
	for phase in nr4_labels.keys():
		if 'localizer' in phase:
			nr4_labels[phase] = np.delete(nr4_labels[phase], rest_index[phase], axis=0)

	#collapse across indoor and outdoor scenes
	for phase in nr4_labels.keys():
		if 'localizer' in phase:
			for i, label in enumerate(nr4_labels[phase]):
				if label == 'indoor' or label == 'outdoor':
					nr4_labels[phase][i] = 'scene'
				else:
					nr4_labels[phase][i] = 'not'
	
	#manually set the labels
	#nr4_names = ['animal','tool','scene','scrambled']
	
	#combine runs for cross validation
	nr4_data_reduced = np.concatenate([nr4_data['localizer_1'],nr4_data['localizer_2']])
	
	#combine labels
	nr4_labels_cv = np.concatenate([nr4_labels['localizer_1'],nr4_labels['localizer_2']])

	#create the run index for the localizer xval
	run_index = np.zeros(len(nr4_labels_cv))
	run_index[0:160] = 1
	run_index[160:] = 2


	#determine machine learning algorithm for classifier
	if args.classifier == 'svm':
		clf = LinearSVC()
	elif args.classifier == 'logreg':
		clf = LogisticRegression()

	#determine type of feature selection
	if args.feature_selection == 'k':
		feature_selection = SelectKBest(f_classif)
		cv_iter = [50,100,250,500,1000,1500,2000,2500,3000,3500,4000]

	if args.feature_selection == '%':
		feature_selection = SelectPercentile(f_classif)
		cv_iter = [1,3,6,10,15,20,30,40,60,80,100]
	
	#make the pipeline
	anova_clf = Pipeline([('anova', feature_selection), ('clf', clf)])

	#anova_clf.fit(nr4_data_reduced, nr4_labels_cv)
	for i, thresh in enumerate(cv_iter):
		
		if args.feature_selection == 'k':
			anova_clf.set_params(anova__k=thresh)
		
		if args.feature_selection == '%':
			anova_clf.set_params(anova__percentile=thresh)
		
		nr4_fs_cv = cross_val_score(anova_clf, nr4_data_reduced, nr4_labels_cv, groups=run_index, cv=2)
		
		nr4_fs_cv_res = nr4_fs_cv.mean()

		sub_out.append(nr4_fs_cv_res)

	mean_out[sub] = sub_out

mean_out = np.transpose(mean_out)

out = pd.DataFrame([], columns=['Mean','SEM'])
out.Mean = mean_out.mean()
out.SEM = mean_out.sem()

out.to_csv('%s/graphing/sklearn_xval/%s_%s_iter2.csv'%(data_dir,args.classifier,args.feature_selection), sep=',')


#print('%s nr4 cv score = %s'%(SUBJ,nr4_fs_cv_res))

		#nr4_res1_fs = clf.fit( nr4_data_reduced['localizer_1'], nr4_labels['localizer_1'] ).predict( nr4_data_reduced['localizer_2'] )
		#nr4_cfmat1_fs = confusion_matrix( nr4_labels['localizer_2'], nr4_res1_fs, labels=nr4_names )

		#nr4_res2_fs = clf.fit( nr4_data_reduced['localizer_2'], nr4_labels['localizer_2'] ).predict( nr4_data_reduced['localizer_1'] )
		#nr4_cfmat2_fs = confusion_matrix( nr4_labels['localizer_1'], nr4_res2_fs, labels=nr4_names )

		#nr4_cfmat_fs = np.mean( np.array( [ nr4_cfmat1_fs, nr4_cfmat2_fs ] ), axis=0 )


		#nr4_cfmat_normal_fs = nr4_cfmat_fs.astype('float') / nr4_cfmat_fs.sum(axis=1)[:, np.newaxis]
		#nr4_accuracy_fs = np.mean(np.diagonal(nr4_cfmat_normal_fs))









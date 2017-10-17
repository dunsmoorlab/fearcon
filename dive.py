#the goal of this script is to run MVPA classification via scikit learn

import os
import argparse
import numpy as np
import nibabel as nib
import scipy as sp
import sys
import matplotlib.pyplot as plt
from glob import glob
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import make_pipeline

#import the local config file
from fc_config import run_key
from fc_config import mvpa_prepped

#import function to plot confusion matrices
from cf_mat_plot import plot_confusion_matrix

#build the python arg_parser
parser = argparse.ArgumentParser(description='Function arguments')

#add arguments
parser.add_argument('-s', '--subj', nargs='+', help='Subject Number', default=0, type=int)
#parser.add_argument('-r','--runs', nargs='+', help='Load prepped data and labels for these runs', default='', type=str)

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


out = '%s/graphing/confusion_matrices/'%(data_dir)
if not os.path.isdir(out):
	os.mkdir(out)

for subject in args.subj:
	#identify the subject
	SUBJ = 'Sub00%s'%(subject)


	#load in the prepped data for each run
	data = {phase: np.load('%s/%s/bold/%s'%(data_dir,SUBJ,mvpa_prepped[phase])) for phase in loc_runs}

	#as well as the labels
	labels = {phase: np.load('%s/%s/model/MVPA/labels/%s_labels.npy'%(data_dir,SUBJ,phase)) for phase in loc_runs}# if phase == 'localizer_2' }


	#labels = {'localizer_1': labels['localizer_2'],
	#	'localizer_2': labels['localizer_2']}


	#shift over 2 for HDR
	hdr_shift = 2
	end_shift = 6 - hdr_shift

	data = {phase: data[phase][hdr_shift:-end_shift] for phase in data.keys()}

	#qualitiy control step to make sure the size of the labels matches the data for each phase
	{ (print('ERROR: data/label size mismatch for %s'%(phase)), sys.exit()) for phase in data.keys() if data[phase].shape[0] != labels[phase].shape[0]  }


	#take out rest
	rest_index = {phase: np.where(labels[phase]=='rest') for phase in labels.keys()}

	#copy the data for each analyses
	#no rest, 5 cat
	nr5_data = data
	nr5_labels = labels

	for phase in nr5_data.keys():
		if phase == 'extinction':
			nr5_data[phase] = nr5_data[phase]
		else:
			nr5_data[phase] = np.delete(nr5_data[phase], rest_index[phase], axis=0)

	for phase in nr5_labels.keys():
		if phase == 'extinction':
			nr5_labels[phase] = nr5_labels[phase]
		else:
			nr5_labels[phase] = np.delete(nr5_labels[phase], rest_index[phase], axis=0)

	#iteratively run classification on localizer runs, swapping what is trained and tested
	clf = svm.LinearSVC()

	#ok, I need to figure out the difference between f_regression and f_classif,
	#and need to correctly implement feature selection as to avoid permanently changing data
	#might have to break up single line of code classifier
	
	anova_filter = SelectKBest(f_regression, k=1000)

	anova_clf = make_pipeline(anova_filter, clf)

	#nr5_clf1.fit(nr5_data['localizer_1'],labels['localizer_1'])
	nr5_names = ['animal','tool','outdoor','indoor','scrambled']

	nr5_res1 = anova_clf.fit( nr5_data['localizer_1'], nr5_labels['localizer_1'] ).predict( nr5_data['localizer_2'] )
	nr5_cfmat1 = confusion_matrix( nr5_labels['localizer_2'], nr5_res1, labels=nr5_names )

	nr5_res2 = anova_clf.fit( nr5_data['localizer_2'], nr5_labels['localizer_2'] ).predict( nr5_data['localizer_1'] )
	nr5_cfmat2 = confusion_matrix( nr5_labels['localizer_1'], nr5_res2, labels=nr5_names )

	nr5_cfmat = np.mean( np.array( [ nr5_cfmat1, nr5_cfmat2 ] ), axis=0 )


	nr5_cfmat_normal = nr5_cfmat.astype('float') / nr5_cfmat.sum(axis=1)[:, np.newaxis]
	accuracy = np.mean(np.diagonal(nr5_cfmat_normal))


	plot_confusion_matrix( nr5_cfmat, classes=nr5_names, normalize=True, title='%s No rest, 5 categories, ANOVA; Mean acc = %s'%(SUBJ,accuracy), save='%s/nr5/%s_nr5.jpg'%(out,SUBJ) )
	nr5_plt = plt.show()

	comp1 = len(np.where((nr5_res1==nr5_labels['localizer_2']) == True)[0])
	comp2 = len(np.where((nr5_res2==nr5_labels['localizer_1']) == True)[0])

	qa_acc = np.mean([comp1,comp2]) / 160

	if qa_acc != accuracy:
		print('QA accuracy error, %s != %s'%(accuracy,qa_acc))
		sys.exit()


sys.exit()


























train = data['localizer_1']
train_labels = labels['localizer_1']

test = data['localizer_2']
test_labels = labels['localizer_2']

for i,label in enumerate(train_labels):
	if label == 'outdoor' or label == 'indoor':
		train_labels[i] = 'scene'
#	else:
#		train_labels[i] = 'not'

for i,label in enumerate(test_labels):
	if label == 'outdoor' or label == 'indoor':
		test_labels[i] = 'scene'
#	else:
#		test_labels[i] = 'not'

clf = svm.LinearSVC()
clf.fit(train, train_labels)


clfres = clf.predict(test)

#results = clfres==labels['localizer_2']
results = clfres==test_labels


out1 = [len(np.where(results==True)[0]), len(np.where(results==True)[0]) / len(results)]



train2 = data['localizer_2']
train_labels2 = labels['localizer_2']

test2 = data['localizer_1']
test_labels2 = labels['localizer_1']

for i,label in enumerate(train_labels2):
	if label == 'outdoor' or label == 'indoor':
		train_labels2[i] = 'scene'
#	else:
#		train_labels2[i] = 'not'

for i,label in enumerate(test_labels2):
	if label == 'outdoor' or label == 'indoor':
		test_labels2[i] = 'scene'
#	else:
#		test_labels2[i] = 'not'

clf2 = svm.LinearSVC()
clf2.fit(train2, train_labels2)


clfres2 = clf2.predict(test2)

#results = clfres==labels['localizer_2']
results2 = clfres2==test_labels2


out2 = [len(np.where(results2==True)[0]), len(np.where(results2==True)[0]) / len(results2)]















#please go to dropbox
#print(out)

#whats next is 
#data = np.concatenate([run for i,run in sorted(data.items())],axis=0)

#labels = np.concatenate([run for i,run in sorted(labels.items())],axis=0)

#run_index = np.empty([320])
#run_index[0:160] = 1
#run_index[160:] = 2

#out = cross_val_score(clf,data,y=labels,groups=run_index, n_jobs=1)

#idk need to change something here but idk what im going home

#out = cross_val_score(clf,train,test, n_jobs=1)




#for extinction...
#data = {phase: np.delete(data[phase], rest_index[phase],axis=0) for phase in data.keys() if phase != 'extinction'}
#labels = {phase: np.delete(labels[phase], rest_index[phase],axis=0) for phase in labels.keys() if phase != 'extinction'}
#for phase in labels.keys():
#	for i,label in enumerate(labels[phase]):
#		if phase == 'extinction':
#			if label == 'CS+':
#				labels[phase][i] = 'animal'
#			if label == 'CS-':
#				labels[phase][i] = 'tool'






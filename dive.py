#the goal of this script is to run MVPA classification via scikit learn

import os
import argparse
import numpy as np
import nibabel as nib
import scipy as sp
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
parser.add_argument('-s', '--subj', nargs='+', help='Subject Number', default=0, type=int)
#parser.add_argument('-r','--runs', nargs='+', help='Load prepped data and labels for these runs', default='', type=str)
parser.add_argument('-fs', '--feature_selection', help='Feature Selection', default=False, type=bool)
parser.add_argument('-c', '--classifier', help='Machine Learning Algorithm Used', default='', type=str)
parser.add_argument('-cat', '--categories', help='4 or 5 categories, collapsing across scenes', default='', type=str)
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


nr5_pdf = PdfPages('%s/nr5/%s_nr5.pdf'%(out,args.classifier))
nr5_fs_pdf = PdfPages('%s/nr5/%s_nr5_ANOVA.pdf'%(out,args.classifier))

nr4_pdf = PdfPages('%s/nr4/%s_nr4.pdf'%(out,args.classifier))
nr4_fs_pdf = PdfPages('%s/nr4/%s_nr4_ANOVA.pdf'%(out,args.classifier))

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
	hdr_shift = 3
	end_shift = 6 - hdr_shift

	data = {phase: data[phase][hdr_shift:-end_shift] for phase in data.keys()}

	#qualitiy control step to make sure the size of the labels matches the data for each phase
	{ (print('ERROR: data/label size mismatch for %s'%(phase)), sys.exit()) for phase in data.keys() if data[phase].shape[0] != labels[phase].shape[0]  }


	#take out rest
	rest_index = {phase: np.where(labels[phase]=='rest') for phase in labels.keys()}

	if args.categories == '5':
			#copy the data for each analyses
			#no rest, 5 cat
		nr5_data = {phase: data[phase] for phase in data.keys()}
		nr5_labels = {phase: labels[phase] for phase in labels.keys()}

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
		if args.classifier == 'svm':
			clf = LinearSVC()
		elif args.classifier == 'logreg':
			clf = LogisticRegression()
		
		nr5_names = ['animal','tool','outdoor','indoor','scrambled']

		
		if args.feature_selection == False:
		

			nr5_res1 = clf.fit( nr5_data['localizer_1'], nr5_labels['localizer_1'] ).predict( nr5_data['localizer_2'] )
			nr5_cfmat1 = confusion_matrix( nr5_labels['localizer_2'], nr5_res1, labels=nr5_names )

			nr5_res2 = clf.fit( nr5_data['localizer_2'], nr5_labels['localizer_2'] ).predict( nr5_data['localizer_1'] )
			nr5_cfmat2 = confusion_matrix( nr5_labels['localizer_1'], nr5_res2, labels=nr5_names )

			nr5_cfmat = np.mean( np.array( [ nr5_cfmat1, nr5_cfmat2 ] ), axis=0 )


			nr5_cfmat_normal = nr5_cfmat.astype('float') / nr5_cfmat.sum(axis=1)[:, np.newaxis]
			nr5_accuracy = np.mean(np.diagonal(nr5_cfmat_normal))


			plot_confusion_matrix( nr5_cfmat, classes=nr5_names ,normalize=True, title='%s No rest, 5 categories, Shift %sTRs; Mean acc = %s'%(SUBJ,hdr_shift,nr5_accuracy), save='pdf', pdf=nr5_pdf )

			comp1 = len(np.where((nr5_res1==nr5_labels['localizer_2']) == True)[0])
			comp2 = len(np.where((nr5_res2==nr5_labels['localizer_1']) == True)[0])

			qa_acc = np.mean([comp1,comp2]) / 160

			if qa_acc != nr5_accuracy:
				print('QA accuracy error, %s != %s'%(nr5_accuracy,qa_acc))
				sys.exit()

		if args.feature_selection == True:
			
			feature_selection = SelectKBest(f_classif, k=1000)
			
			anova_clf = Pipeline([('anova', feature_selection), ('clf', clf)])
			#anova_filter = SelectPercentile(f_classif, percentile=75)
			
			nr5_data_reduced = np.concatenate([nr5_data['localizer_1'],nr5_data['localizer_2']])
			
			nr5_labels_cv = np.concatenate([nr5_labels['localizer_1'],nr5_labels['localizer_2']])

			
			run_index = np.zeros(len(nr5_labels_cv))
			run_index[0:160] = 1
			run_index[160:] = 2

			anova_clf.fit(nr5_data_reduced, nr5_labels_cv)

			nr5_fs_cv = cross_val_score(anova_clf, nr5_data_reduced, nr5_labels_cv, groups=run_index, cv=2)
			nr5_fs_cv_res = nr5_fs_cv.mean()

			print('%s nr5 cv score = %s'%(SUBJ,nr5_fs_cv_res))
			#nr5_res1_fs = clf.fit( nr5_data_reduced['localizer_1'], nr5_labels['localizer_1'] ).predict( nr5_data_reduced['localizer_2'] )
			#nr5_cfmat1_fs = confusion_matrix( nr5_labels['localizer_2'], nr5_res1_fs, labels=nr5_names )

			#nr5_res2_fs = clf.fit( nr5_data_reduced['localizer_2'], nr5_labels['localizer_2'] ).predict( nr5_data_reduced['localizer_1'] )
			#nr5_cfmat2_fs = confusion_matrix( nr5_labels['localizer_1'], nr5_res2_fs, labels=nr5_names )

			#nr5_cfmat_fs = np.mean( np.array( [ nr5_cfmat1_fs, nr5_cfmat2_fs ] ), axis=0 )


			#nr5_cfmat_normal_fs = nr5_cfmat_fs.astype('float') / nr5_cfmat_fs.sum(axis=1)[:, np.newaxis]
			#nr5_accuracy_fs = np.mean(np.diagonal(nr5_cfmat_normal_fs))


			#plot_confusion_matrix( nr5_cfmat_fs, classes=nr5_names ,normalize=True, title='%s No rest, 5 categories, ANOVA, Shift %sTRs; Mean acc = %s'%(SUBJ,hdr_shift,nr5_accuracy_fs), save='pdf', pdf=nr5_fs_pdf )
			
	####################################################################################################
	if args.categories == '4':

		nr4_data = {phase: data[phase] for phase in data.keys()}
		nr4_labels = {phase: labels[phase] for phase in labels.keys()}

		for phase in nr4_data.keys():
			if phase == 'extinction':
				nr4_data[phase] = nr4_data[phase]
			else:
				nr4_data[phase] = np.delete(nr4_data[phase], rest_index[phase], axis=0)

		for phase in nr4_labels.keys():
			if phase == 'extinction':
				nr4_labels[phase] = nr4_labels[phase]
			else:
				nr4_labels[phase] = np.delete(nr4_labels[phase], rest_index[phase], axis=0)

		for phase in nr4_labels.keys():
			if phase == 'extinction':
				nr4_labels[phase] = nr4_labels[phase]
			else:
				for i, label in enumerate(nr4_labels[phase]):
					if label == 'indoor' or label == 'outdoor':
						nr4_labels[phase][i] = 'scene'

		#iteratively run classification on localizer runs, swapping what is trained and tested
		if args.classifier == 'svm':
			clf = LinearSVC()
		elif args.classifier == 'logreg':
			clf = LogisticRegression()
		


		
		nr4_names = ['animal','tool','scene','scrambled']
		
		if args.feature_selection == False:

			nr4_res1 = clf.fit( nr4_data['localizer_1'], nr4_labels['localizer_1'] ).predict( nr4_data['localizer_2'] )
			nr4_cfmat1 = confusion_matrix( nr4_labels['localizer_2'], nr4_res1, labels=nr4_names )

			nr4_res2 = clf.fit( nr4_data['localizer_2'], nr4_labels['localizer_2'] ).predict( nr4_data['localizer_1'] )
			nr4_cfmat2 = confusion_matrix( nr4_labels['localizer_1'], nr4_res2, labels=nr4_names )

			nr4_cfmat = np.mean( np.array( [ nr4_cfmat1, nr4_cfmat2 ] ), axis=0 )


			nr4_cfmat_normal = nr4_cfmat.astype('float') / nr4_cfmat.sum(axis=1)[:, np.newaxis]
			nr4_accuracy = np.mean(np.diagonal(nr4_cfmat_normal))


			plot_confusion_matrix( nr4_cfmat, classes=nr4_names ,normalize=True, title='%s No rest, 4 categories, Shift %sTRs; Mean acc = %s'%(SUBJ,hdr_shift,nr4_accuracy), save='pdf', pdf=nr4_pdf )

			comp1 = len(np.where((nr4_res1==nr4_labels['localizer_2']) == True)[0])
			comp2 = len(np.where((nr4_res2==nr4_labels['localizer_1']) == True)[0])

			qa_acc = np.mean([comp1,comp2]) / 160

			#if qa_acc != nr4_accuracy:
			#	print('QA accuracy error, %s != %s'%(nr4_accuracy,qa_acc))
			#	sys.exit()

		if args.feature_selection == True:
			
			feature_selection = SelectKBest(f_classif, k=1000)
			
			anova_clf = Pipeline([('anova', feature_selection), ('clf', clf)])
			#anova_filter = SelectPercentile(f_classif, percentile=75)
			
			nr4_data_reduced = np.concatenate([nr4_data['localizer_1'],nr4_data['localizer_2']])
			
			nr4_labels_cv = np.concatenate([nr4_labels['localizer_1'],nr4_labels['localizer_2']])

			
			run_index = np.zeros(len(nr4_labels_cv))
			run_index[0:160] = 1
			run_index[160:] = 2

			anova_clf.fit(nr4_data_reduced, nr4_labels_cv)

			nr4_fs_cv = cross_val_score(anova_clf, nr4_data_reduced, nr4_labels_cv, groups=run_index, cv=2)
			nr4_fs_cv_res = nr4_fs_cv.mean()

			print('%s nr4 cv score = %s'%(SUBJ,nr4_fs_cv_res))
			#nr4_res1_fs = clf.fit( nr4_data_reduced['localizer_1'], nr4_labels['localizer_1'] ).predict( nr4_data_reduced['localizer_2'] )
			#nr4_cfmat1_fs = confusion_matrix( nr4_labels['localizer_2'], nr4_res1_fs, labels=nr4_names )

			#nr4_res2_fs = clf.fit( nr4_data_reduced['localizer_2'], nr4_labels['localizer_2'] ).predict( nr4_data_reduced['localizer_1'] )
			#nr4_cfmat2_fs = confusion_matrix( nr4_labels['localizer_1'], nr4_res2_fs, labels=nr4_names )

			#nr4_cfmat_fs = np.mean( np.array( [ nr4_cfmat1_fs, nr4_cfmat2_fs ] ), axis=0 )


			#nr4_cfmat_normal_fs = nr4_cfmat_fs.astype('float') / nr4_cfmat_fs.sum(axis=1)[:, np.newaxis]
			#nr4_accuracy_fs = np.mean(np.diagonal(nr4_cfmat_normal_fs))


			#plot_confusion_matrix( nr4_cfmat_fs, classes=nr4_names ,normalize=True, title='%s No rest, 5 categories, ANOVA, Shift %sTRs; Mean acc = %s'%(SUBJ,hdr_shift,nr4_accuracy_fs), save='pdf', pdf=nr4_fs_pdf )

nr5_pdf.close()
nr5_fs_pdf.close()

nr4_pdf.close()
nr4_fs_pdf.close()

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






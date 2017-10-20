import os
import argparse
import numpy as np
import pandas as pd


#to do a GLM in FSL, need 3 column matrix for each condition in each run, in the format of:
#onset, duration, parametric modulation


#set arguments
parser = argparse.ArgumentParser(description='Function arguments')
#set the subject argument as a string so that it can take arguments like '1' or 'all'
parser.add_argument('-s','--subj', help='Subject number', default=00, type=str)
args = parser.parse_args()

#point to the data direct
data_dir = '/Users/ach3377/GoogleDrive/FC_FMRI_DATA/'

#set numpy print options
np.set_printoptions(precision=8)

#handle the subjects arg (all or one)
if args.subj == 'all':
	subj = [int(subs[3:]) for subs in os.listdir(data_dir) if 'Sub' in subs and 'fs' not in subs]
	iterations = len(subj)
#if its not 'all' then we need to convert it into an int
else:
	subj = [int(args.subj)]
	iterations = 1

for iteration in range(0,iterations):
	#have to do all experimental runs first
	meta = pd.read_csv(data_dir + os.sep + "Sub{0:0=3d}".format(subj[iteration]) + os.sep + 'behavior' + os.sep + "Sub{0:0=3d}_elog.csv".format(subj[iteration]))


	SUBJ = 'Sub{0:0=3d}'.format(subj[iteration])
	onsets = '/%s/%s/model/GLM/onsets'%(data_dir,SUBJ)
	#make the directories to store the files
	run_folders = [
		onsets,
		'%s/run001'%(onsets),
		'%s/run002'%(onsets),
		'%s/run003'%(onsets)
		]

	#if not os.path.isdir(onsets):
	#	for i, folder in enumerate(run_folders):
	#		os.mkdir(run_folders[i])

	{ os.mkdir(run_folders[i]) for i, folder in enumerate(run_folders) if not os.path.isdir(run_folders[i]) }

	#then load in localizer meta
	############################

	base_meta = meta[meta.phase == 'baseline']
	base_start = base_meta['InitialITI.OnsetTime'][0]
	
	base_CSplus = pd.DataFrame([],columns=['Onset','Duration','PM'])
	base_CSmin = pd.DataFrame([],columns=['Onset','Duration','PM'])

	#Grab the onset times for everything
	base_CSplus.Onset = base_meta['stim.OnsetTime'][base_meta.cstype == 'CS+'] - base_start
	base_CSmin.Onset = base_meta['stim.OnsetTime'][base_meta.cstype == 'CS-'] - base_start

	#then get the durations
	base_CSplus.Duration = base_meta['stim.Duration'][base_meta.cstype == 'CS+']
	base_CSmin.Duration = base_meta['stim.Duration'][base_meta.cstype == 'CS-']

	#set the parametric modulation, which for now is just going to be 1 for everything until I learn what that means
	base_CSplus.PM = [1] * len(base_CSplus)
	base_CSmin.PM = [1] * len(base_CSmin)

	#convert onset and duration to seconds from miliseconds
	base_CSplus.Onset = base_CSplus.Onset / 1000
	base_CSmin.Onset = base_CSmin.Onset / 1000

	base_CSplus.Duration = base_CSplus.Duration / 1000
	base_CSmin.Duration = base_CSmin.Duration / 1000

	#make them into numpy arrarys that do wonderful scientific notation
	base_CSplus = np.array(base_CSplus)
	base_CSmin = np.array(base_CSmin)

	#save them
	np.savetxt('%s/%s/model/GLM/onsets/run001/csplus.txt'%(data_dir,SUBJ), base_CSplus, fmt='%.8e', delimiter='\t')
	np.savetxt('%s/%s/model/GLM/onsets/run001/csmin.txt'%(data_dir,SUBJ), base_CSmin, fmt='%.8e', delimiter='\t')
	

	#do it all again for fear conditioning

	fear_meta = meta[meta.phase == 'fearconditioning']
	fear_start = fear_meta['InitialITI.OnsetTime'][48]
	
	fear_CSplus = pd.DataFrame([],columns=['Onset','Duration','PM'])
	fear_CSmin = pd.DataFrame([],columns=['Onset','Duration','PM'])

	#Grab the onset times for everything
	fear_CSplus.Onset = fear_meta['stim.OnsetTime'][fear_meta.cstype == 'CS+'] - fear_start
	fear_CSmin.Onset = fear_meta['stim.OnsetTime'][fear_meta.cstype == 'CS-'] - fear_start

	#then get the durations
	fear_CSplus.Duration = fear_meta['stim.Duration'][fear_meta.cstype == 'CS+']
	fear_CSmin.Duration = fear_meta['stim.Duration'][fear_meta.cstype == 'CS-']

	#set the parametric modulation, which for now is just going to be 1 for everything until I learn what that means
	fear_CSplus.PM = [1] * len(fear_CSplus)
	fear_CSmin.PM = [1] * len(fear_CSmin)
	
	#convert onset and duration to seconds from miliseconds
	fear_CSplus.Onset = fear_CSplus.Onset / 1000
	fear_CSmin.Onset = fear_CSmin.Onset / 1000

	fear_CSplus.Duration = fear_CSplus.Duration / 1000
	fear_CSmin.Duration = fear_CSmin.Duration / 1000


	#make them into numpy arrarys that do wonderful scientific notation
	fear_CSplus = np.array(fear_CSplus)
	fear_CSmin = np.array(fear_CSmin)

	#save them
	np.savetxt('%s/%s/model/GLM/onsets/run002/csplus.txt'%(data_dir,SUBJ), fear_CSplus, fmt='%.8e', delimiter='\t')
	np.savetxt('%s/%s/model/GLM/onsets/run002/csmin.txt'%(data_dir,SUBJ), fear_CSmin, fmt='%.8e', delimiter='\t')


	#do the same for extinction, but also undo the dumb row expansion that e-prime does with the scene ITIs
	phase3_stims = pd.Series(0)
	phase3_unique_loc = pd.Series(0)
	q = 0
	#this paragraph goes through all the stims and extinciton
	#and collects unique stim names in their experimental order
	for loc, unique in enumerate(meta.stims[meta.phase == 'extinction']):
		if not any(stim == unique for stim in phase3_stims):
			phase3_stims[q] = unique
			phase3_unique_loc[q] = loc + 96
			q = q + 1

	ext_meta = meta.loc[phase3_unique_loc,]
	ext_meta.index = list(range(0,ext_meta.shape[0]))

	ext_start = ext_meta['InitialITI.OnsetTime'][0]
	
	ext_CSplus = pd.DataFrame([],columns=['Onset','Duration','PM'])
	ext_CSmin = pd.DataFrame([],columns=['Onset','Duration','PM'])

	#Grab the onset times for everything
	ext_CSplus.Onset = ext_meta['stim.OnsetTime'][ext_meta.cstype == 'CS+'] - ext_start
	ext_CSmin.Onset = ext_meta['stim.OnsetTime'][ext_meta.cstype == 'CS-'] - ext_start

	#then get the durations
	ext_CSplus.Duration = ext_meta['stim.Duration'][ext_meta.cstype == 'CS+']
	ext_CSmin.Duration = ext_meta['stim.Duration'][ext_meta.cstype == 'CS-']

	#set the parametric modulation, which for now is just going to be 1 for everything until I learn what that means
	ext_CSplus.PM = [1] * len(ext_CSplus)
	ext_CSmin.PM = [1] * len(ext_CSmin)

	#convert onset and duration to seconds from miliseconds
	ext_CSplus.Onset = ext_CSplus.Onset / 1000
	ext_CSmin.Onset = ext_CSmin.Onset / 1000

	ext_CSplus.Duration = ext_CSplus.Duration / 1000
	ext_CSmin.Duration = ext_CSmin.Duration / 1000

	#make them into numpy arrarys that do wonderful scientific notation
	ext_CSplus = np.array(ext_CSplus)
	ext_CSmin = np.array(ext_CSmin)

	#save them
	np.savetxt('%s/%s/model/GLM/onsets/run003/csplus.txt'%(data_dir,SUBJ), ext_CSplus, fmt='%.8e', delimiter='\t')
	np.savetxt('%s/%s/model/GLM/onsets/run003/csmin.txt'%(data_dir,SUBJ), ext_CSmin, fmt='%.8e', delimiter='\t')


	print('%s GLM timing files created'%(SUBJ))





























	#base_CSplus.to_csv('cond001.txt', header=None, index=None, sep=' ', mode='a')
	#fucking figure out how to do 8 decimal points with scientific notation idfk this is really frustrating
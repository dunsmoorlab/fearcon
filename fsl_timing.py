import os
import argparse
import numpy as np
import pandas as pd
from toolz import interleave

#to do a GLM in FSL, need 3 column matrix for each condition in each run, in the format of:
#onset, duration, parametric modulation


#set arguments
parser = argparse.ArgumentParser(description='Function arguments')
#set the subject argument as a string so that it can take arguments like '1' or 'all'
parser.add_argument('-s','--subj', nargs = '+', help='Subject number', default=00, type=str)
args = parser.parse_args()

#point to the data direct
data_dir = '/Users/ach3377/GoogleDrive/FC_FMRI_DATA/'

#set numpy print options
np.set_printoptions(precision=8)

if args.subj == ['all']:
	sub_args = [int(subs[3:]) for subs in os.listdir(data_dir) if 'Sub' in subs and 'fs' not in subs]
else:
	sub_args = args.subj

for sub in sub_args:
	#have to do all experimental runs first
	SUBJ = 'Sub00%s'%(sub)

	meta = pd.read_csv(data_dir + os.sep + SUBJ + os.sep + 'behavior' + os.sep + '%s_elog.csv'%(SUBJ))

	onsets = '/%s/%s/model/GLM/onsets'%(data_dir,SUBJ)

	#########################
	#baseline
	#########################

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
	

	#########################
	#fear conditioning
	#########################
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

	#########################
	#extinciton
	#########################
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

	
	#now code the scene events
	ext_objects = pd.DataFrame(np.concatenate([ext_CSplus, ext_CSmin])).sort_values(0)
	ext_objects.index = list(range(0,48))

	#set up the data frame
	ext_scenes = pd.DataFrame([],columns=['Onset','Duration','PM'])

	#calculate the onsets
	ext_scenes.Onset = [(onset + ext_objects[1][i]) for i, onset in enumerate(ext_objects[0])] 

	#and durations
	scene_durations = [(ext_objects[0][i+1] - onset) for i, onset in enumerate(ext_scenes.Onset) if i < 47]

	#just using the mean ITI duration for the last one
	scene_durations.append(np.mean(scene_durations))

	#save the durations
	ext_scenes.Duration = scene_durations

	#set PM to 1
	ext_scenes.PM = [1] * len(ext_scenes)

	#convert back to numpy array
	ext_scenes = np.array(ext_scenes)

	#save it
	np.savetxt('%s/%s/model/GLM/onsets/run003/context_tag.txt'%(data_dir,SUBJ), ext_scenes, fmt='%.8e', delimiter='\t')

	
	#########################
	#extinciton recall
	#########################
	er_meta = meta[meta.phase == 'extinctionRecall']
	er_start = er_meta['InitialITI.OnsetTime'][er_meta['InitialITI.OnsetTime'].index[1]]
	
	er_CSplus = pd.DataFrame([],columns=['Onset','Duration','PM'])
	er_CSmin = pd.DataFrame([],columns=['Onset','Duration','PM'])
	er = pd.DataFrame([],columns=['Onset','Duration','PM'])


	#Grab the onset times for everything
	er_CSplus.Onset = er_meta['stim.OnsetTime'][er_meta.cstype == 'CS+'] - er_start
	er_CSmin.Onset = er_meta['stim.OnsetTime'][er_meta.cstype == 'CS-'] - er_start
	er.Onset = er_meta['stim.OnsetTime'] - er_start

	#then get the durations
	er_CSplus.Duration = er_meta['stim.Duration'][er_meta.cstype == 'CS+']
	er_CSmin.Duration = er_meta['stim.Duration'][er_meta.cstype == 'CS-']
	er.Duration = er_meta['stim.Duration']

	#set the parametric modulation, which for now is just going to be 1 for everything until I learn what that means
	er_CSplus.PM = [1] * len(er_CSplus)
	er_CSmin.PM = [1] * len(er_CSmin)
	er.PM = [1] * len(er)

	#convert onset and duration to seconds from miliseconds
	er_CSplus.Onset = er_CSplus.Onset / 1000
	er_CSmin.Onset = er_CSmin.Onset / 1000
	er.Onset = er.Onset / 1000

	er_CSplus.Duration = er_CSplus.Duration / 1000
	er_CSmin.Duration = er_CSmin.Duration / 1000
	er.Duration = er.Duration / 1000


	#make them into numpy arrarys that do wonderful scientific notation
	er_CSplus = np.array(er_CSplus)
	er_CSmin = np.array(er_CSmin)
	er = np.array(er)

	early_er_CSplus = er_CSplus[0:4, :]
	early_er_CSmin = er_CSmin[0:4, :]

	late_er_CSplus = er_CSplus[4:, :]
	late_er_CSmin = er_CSmin[4:, :]

	#save them
	np.savetxt('%s/%s/model/GLM/onsets/run004/csplus.txt'%(data_dir,SUBJ), er_CSplus, fmt='%.8e', delimiter='\t')
	np.savetxt('%s/%s/model/GLM/onsets/run004/csmin.txt'%(data_dir,SUBJ), er_CSmin, fmt='%.8e', delimiter='\t')
	np.savetxt('%s/%s/model/GLM/onsets/run004/all_stim.txt'%(data_dir,SUBJ), er, fmt='%.8e', delimiter='\t')

	np.savetxt('%s/%s/model/GLM/onsets/run004/csplus_early.txt'%(data_dir,SUBJ), early_er_CSplus, fmt='%.8e', delimiter='\t')
	np.savetxt('%s/%s/model/GLM/onsets/run004/csmin_early.txt'%(data_dir,SUBJ), early_er_CSmin, fmt='%.8e', delimiter='\t')

	np.savetxt('%s/%s/model/GLM/onsets/run004/csplus_late.txt'%(data_dir,SUBJ), late_er_CSplus, fmt='%.8e', delimiter='\t')
	np.savetxt('%s/%s/model/GLM/onsets/run004/csmin_late.txt'%(data_dir,SUBJ), late_er_CSmin, fmt='%.8e', delimiter='\t')



	print('%s GLM timing files created'%(SUBJ))
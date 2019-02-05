#def make_labels(phase=str)
#phase = 'localizer_1'

import pandas as pd
import numpy as np
import sys
import os
import itertools as it
import argparse
from sys import platform
from fc_config import day2phase
from fc_config import time2add
from fc_config import LocStims

parser = argparse.ArgumentParser(description='Function arguments')

#add arguments
parser.add_argument('-s', '--subj', nargs='+', help='Subject Number', default=0, type=str)

#parser.add_argument('-ph', '--label', help='Phases to generate labels for, as in, "label these!"', default='', type=str)
parser.add_argument('-l', '--label', nargs='+', help='Phases to generate labels for, as in, "label these!"', default='', type=str)
#parse them
args=parser.parse_args()

#point to data directory
if platform == 'linux':
	#point bash to the folder with the subjects in
	data_dir = '/mnt/c/Users/ACH/Google Drive/FC_FMRI_DATA/'
#but mostly it runs on a school mac
else:
	data_dir = '/Users/ach3377/GoogleDrive/FC_FMRI_DATA'

if args.subj == ['all']:
	sub_args = [int(subs[3:]) for subs in os.listdir(data_dir) if 'Sub' in subs and 'fs' not in subs]
else:
	sub_args = args.subj

for sub in sub_args:
	sub = int(sub)
	SUBJ = 'Sub{0:0=3d}'.format(sub)

	labels = '%s/%s/model/MVPA/labels'%(data_dir, SUBJ)
	if not os.path.isdir(labels):
		os.mkdir(labels)

	#need to set a phase == all case
	if args.label == ['all']:
		label_args = ['baseline','fear_conditioning','extinction','extinction_recall','memory_run_1','memory_run_2','memory_run_3','localizer_1','localizer_2']
	else:
		label_args = args.label

	for phase in label_args:

		runtype = day2phase[phase]

		#read in the meta data for this phase
		meta = pd.read_csv('%s/%s/behavior/run_logs/'%(data_dir,SUBJ) + phase + '_meta.csv')

		#with the scenes, eprime spits out a row for every single stim
		#So, we need to collapse the metadata
		if runtype == 'day1_tag':
			#gather the unique trials
			unique_trials = { trial: meta.overalltrial[meta.overalltrial == trial] for trial in np.unique(meta.overalltrial) }
			
			#need these now
			iti_dur = []	
			stim_dur = []

			#and will need this for later
			cs_tag = []

			#assign the relevant variables based on their first occurance in the meta file
			{ ( iti_dur.extend([meta.ITIWeight.values[unique_trials[trial].index[0]]]),
			stim_dur.extend([meta.stimdur.values[unique_trials[trial].index[0]]]),
			cs_tag.extend([meta.cstype.values[unique_trials[trial].index[0]]]) 
			) for trial in unique_trials.keys() }

			#convert them all to arrays, and the iti_dur to miliseconds
			iti_dur = np.array(iti_dur) * 1000
			stim_dur = np.array(stim_dur)

		#the localizer run has speacial and constant stim & iti dur
		elif runtype == 'day2_loc':
			#stim duration
			stim_dur = time2add[runtype][1]
			#ACTUALLY, what were doing for now, is that were going to model the localizer as just 1 TR per stim.
			#stim is 1.5s and iti is .5s, so lets just make it 1TR
			iti_dur = time2add[runtype][2]
			#inter mini-block duration
			imbi_dur = time2add[runtype][3]
			#inter block duration
			ibi_dur = time2add[runtype][4]

			#also initialize raw_time
			temp_time = []

		#if its not the tag or localizer run, then get the stim and iti durations
		else:
			#memory runs dont have a stim dur variable, but all are set at 3000ms
			if runtype == 'day2_mem':
				stim_dur = pd.Series([3000] * len(meta.ITIdur))
			else:
				stim_dur = meta.stimdur
			iti_dur = meta.ITIdur


		#the localizer run is speacial, need to reconstruct the block design
		if runtype == 'day2_loc':
			for run in meta.Procedure:
				if run == 'LocStim':
					temp_time.extend([stim_dur + iti_dur])
				elif run == 'IMBI':
					temp_time.extend([imbi_dur])
			temp_time = np.array(temp_time)

			blk1 = temp_time[0:44]
			blk2 = temp_time[44:88]
			blk3 = temp_time[88:132]
			blk4 = temp_time[132:]

			raw_time = np.concatenate(([time2add[runtype][0]],blk1,[ibi_dur],blk2,[ibi_dur],blk3,[ibi_dur],blk4,[ibi_dur]))

		#if its not the localizer, then:
		#interleavingly stack the stim_dur and iti_dur, starting with stim
		else:
			raw_time = np.vstack((stim_dur,iti_dur)).reshape((-1,),order='F')

			#add the initial and final ITI based on eprime files (this has to be set manually)
			#I think we said that it was supposed to be 8000 ITI but it ended up being 4...
			raw_time = np.insert(raw_time,0,time2add[runtype][0])
			raw_time = np.append(raw_time,time2add[runtype][1])

		#get the total number of TRs
		tr_sum = sum(raw_time / 1000 / 2)

		#And the total time in .25s
		total_time = range(1,int(tr_sum * 4 + 1))

		#this is so messy but it works
		#basically wanted to blow out the trs to take up 4 spaces
		p = 1
		tr_count = []
		while p <= tr_sum:
			for i, tr in enumerate(it.repeat(p,4)):
				tr_count = np.append(tr_count,tr)
			p += 1



		#next step is to break out the labels for 4 TRs,
		#with the goal that whichever label is the majority of the TR wins, with rest winning ties

		#add the appropriate number of 'rest' labels based on # of ITIs
		#but make it 'scene' if it is the tag run
		if runtype == 'day1_tag':
			iti_list = ['scene'] * time2add[runtype][2]
		elif runtype != 'day2_loc':
			iti_list = ['rest'] * time2add[runtype][2]


		#get the CS conditions, which varies depending on which run youre on...
		if runtype == 'day1_tag':
			cs_type = np.array(cs_tag)
		elif runtype == 'day2_mem':
			cs_type = meta.CStype
		elif runtype != 'day2_loc':
			cs_type = meta.cstype

		if runtype == 'day2_loc':
			temp_label = []
			for cond in meta.stims:
				if type(cond) == float:
					temp_label.extend(['rest'])
				else:
					temp_label.extend([LocStims[cond[7:11]]])

			temp_label = np.array(temp_label)

			#looks like ibi should go every 45th,
			#so with the initial ITI it should be [0,45,90,135]
			#but we have to account for the fact that when we add something we have to shift the index
			short_label = np.insert(temp_label,[0,44,88,132,176],'rest')



		#interleavingly stack the cs conditions with the ITI conditions, starting with cs
		else: 
			short_label = np.vstack((cs_type,iti_list)).reshape((-1,),order='F')

			#add in initialITI and finalITI labels
			short_label = np.insert(short_label,0,'rest')
			short_label = np.append(short_label,'rest')


		#compile the raw time, compact labels, and TRs
		raw_time2label = pd.DataFrame({'time':raw_time,'label':short_label, 'TRs':(raw_time/2000) * 4})


		#stretch out the labels
		label_stretch = []
		{label_stretch.extend([raw_time2label['label'][q]] * int(raw_time2label['TRs'][q])) for q in range(0,raw_time2label.shape[0])}

		#compile the total time, TR count, and stretched labels
		timing_master = pd.DataFrame({'time':total_time, 'TR':tr_count, 'label':label_stretch})

		#create a dicitonary where each tr contains the 4 labels that occupy it in time
		tr2label = { tr: timing_master.label[timing_master.TR == tr] for tr in timing_master.TR }

		#{ print(tr2label[tr].values) for tr in tr2label.keys() }

		#As FearCon is now, I dont think its possible for TRs to have more than 2 unique labels in them
		#That being said, we should throw an error in care there is 3 or 4
		{ sys.exit('TR with 3 different values present!!!!') for tr in tr2label.keys() if len(np.unique(tr2label[tr].values)) == 3 }
		{ sys.exit('TR with 4 different values present!!!!') for tr in tr2label.keys() if len(np.unique(tr2label[tr].values)) == 4 }

		#do the TR math!
		out_labels = []
		#for each tr,
		for tr in tr2label.keys():
			#if all the labels for each 1/4 second are the same,
			if tr2label[tr].values[0] == tr2label[tr].values[1] == tr2label[tr].values[2] == tr2label[tr].values[3]:
				#then make that label the TR label
				out_labels.extend([tr2label[tr].values[0]])
			#otherwise, if there are 2 different labels in a TR
			elif len(np.unique(tr2label[tr].values)) == 2:
				#see if one is in the TR more than the other
				if len(tr2label[tr][tr2label[tr].values == np.unique(tr2label[tr].values[0])]) > len(tr2label[tr][tr2label[tr].values == np.unique(tr2label[tr].values[1])]):
					#and make that the TR label
					out_labels.extend([tr2label[tr].values[0]])
				#or the other way around
				elif len(tr2label[tr][tr2label[tr].values == np.unique(tr2label[tr].values[0])]) < len(tr2label[tr][tr2label[tr].values == np.unique(tr2label[tr].values[1])]):
					#and make that the TR label
					out_labels.extend([tr2label[tr].values[1]])
				#if they are the same length then set it what came first/CS because i think np.unique returns in alphabetical or something
				elif len(tr2label[tr][tr2label[tr].values == np.unique(tr2label[tr].values[0])]) == len(tr2label[tr][tr2label[tr].values == np.unique(tr2label[tr].values[1])]):
					out_labels.extend([tr2label[tr].values[0]])

		#save a csv so we can work with them in R for the MVPA toolbox			
		np.savetxt('%s/%s/model/MVPA/toolbox/%s_labels_RAW.csv'%(data_dir,SUBJ,phase), out_labels, fmt='%s', delimiter = ',')
			#function returns an array of the labels, with same length as TRs in run
		np.save('%s/%s/model/MVPA/labels/%s_labels'%(data_dir,SUBJ,phase), np.array(out_labels))


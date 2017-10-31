import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys
from scipy import stats
#script needs to be run from the 'fearcon pilot analysis' folder or it will not w0rk
# After i get this to work once, go back and figure out how to add an optional argument to evaluate 
# only high conffccridence hits.
# current goal is to simply recreate the fearconCR.m script

#set arguments
parser = argparse.ArgumentParser(description='Function arguments')
#set the subject argument as a string so that it can take arguments like '1' or 'all'
parser.add_argument('-s','--subj', help='Subject number', default=00, type=str)
args = parser.parse_args()

#point to the data direct
if sys.platform == 'linux':
	#point bash to the folder with the subjects in
	data_dir = '/mnt/c/Users/ACH/Google Drive/FC_FMRI_DATA/'
elif sys.platform == 'win32':
	data_dir = 'C:\\Users\\ACH\\Google Drive\\FC_FMRI_DATA'
#but mostly it runs on a school mac
else:
	data_dir = '/Users/ach3377/GoogleDrive/FC_FMRI_DATA/'

#handle the subjects arg (all or one)
if args.subj == 'all':
	subj = [int(subs[3:]) for subs in os.listdir(data_dir) if 'Sub' in subs and 'fs' not in subs]
	iterations = len(subj)
#if its not 'all' then we need to convert it into an int
else:
	subj = [int(args.subj)]
	iterations = 1


#Initialize some variables that will be the same for every subject in this experiment
#this is number of rows in the merged csv file that correspond to day1, including the multiple rows of extinction
day1_index = list(range(0,384))
#this is the number of rows corresponding to just the recognition memroy runs
phase5 = list(range(408,648))

#before we get to the heavy lifting, we need to initialzie some variables to accept the results for each subject

#If you're going to export to R or something, use these
phase_cr = pd.DataFrame([],columns=['Phase','Condition','CR','SEM'])
phase_cr['Phase'] = ['Baseline','Baseline','Fear_Conditioning','Fear_Conditioning','Extinction','Extinction','False_Alarm','False_Alarm']
phase_cr['Condition'] = ['CS+','CS-','CS+','CS-','CS+','CS-','CS+','CS-',]
#need this in the middle
r_phase_cr = pd.DataFrame([])

#Use this one if graphing in python
phase_cr_long = pd.DataFrame([],columns=['Subj','Phase','Condition','CR'])

#intialize variables for looking at CR by block
block_cr = pd.DataFrame([],columns=['Phase','Block','Condition','CR','SEM'])
block_cr['Phase'] = ['Baseline'] * 12 + ['Fear_Conditioning'] * 12 + ['Extinction'] * 12
block_cr['Condition'] = ['CS+', 'CS-'] * 18
block_cr['Block'] = ['Block_1']*2 + ['Block_2']*2 + ['Block_3']*2 + ['Block_4']*2 + ['Block_5']*2 + ['Block_6']*2 + ['Block_7']*2 + ['Block_8']*2 + ['Block_9']*2 + ['Block_10']*2 + ['Block_11']*2 + ['Block_12']*2 + ['Block_13']*2 + ['Block_14']*2 + ['Block_15']*2 + ['Block_16']*2 + ['Block_17']*2 + ['Block_18']*2 
#need this in the middle
r_block_cr = pd.DataFrame([])


#and make one for graphing in python
block_cr_long = pd.DataFrame([],columns=['Subj','Phase','Block','Condition','CR'])

#initialize variables for shock expectancy
#shock_expectancy = pd.DataFrame([],columns=['Phase','Trial','Response','Sum'])
#shock_expectancy.Trial = list(range(1,121))
#shock_expectancy.Phase[0:48] = ['Fear_Conditioning']*48
#shock_expectancy.Phase[48:96] = ['Extinction']*48
#shock_expectancy.Phase[96:120] = ['Extinction_Recall']*24


#do the analysis for each subject given
for iteration in range(0,iterations):
	#load in the merged csv file for each subect
	meta = pd.read_csv(data_dir + os.sep + "Sub{0:0=3d}".format(subj[iteration]) + os.sep + 'behavior' + os.sep + "Sub{0:0=3d}_elog.csv".format(subj[iteration]))
	#say which subject is loading
	print('loading data from subject {0:0=3d}'.format(subj[iteration]))

	#collect the raw responses from the recognition test
	respconv = meta['oldnew.RESP'][phase5]
	#fill in any non-responses with 0
	respconv = respconv.fillna(0)
	#collect the old/new attribute for each stim
	memcond = meta.MemCond[phase5]
	#collect the CS type for each stim
	condition = meta.cstype

	#collect stims from baseline and fear conditioning
	phase1_2_stims = meta.stims[0:96]

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
	#give it the correct index numbering
	phase3_stims.index = list(range(96,144))
	
	#concatenate all stims from day1
	day1_stims = pd.Series(np.zeros(144))
	day1_stims[0:96] = phase1_2_stims
	day1_stims[96:144] = phase3_stims

	#collect the stims from day2, phase5 (much easier)
	day2_stims = meta.stims[phase5]

	#get rid of 'stims/' and 'stims2/' in both so they can be compared
	day1_stims = day1_stims.str.replace('stims/','')
	day2_stims = day2_stims.str.replace('stims2/','')

	#lastely, make a variable for all of the day1 conditions, counter-acting how eprime fucks up extinciton
	day1_condition = pd.Series(np.zeros(144))
	day1_condition[0:96] = condition[0:96]
	day1_condition[96:144] = condition[phase3_unique_loc]

	#now lets look at their responses
	#set up some variables to translate raw responses into meaning
	correct_rejection = np.zeros(0)
	false_alarm = np.zeros(0)
	miss = np.zeros(0)
	hit = np.zeros(0)

	#convert raw response into meaning
	#right now this isn't built to seperate out confidence, this is where that would have to happen
	for i in day2_stims.index:
		#if its new, has to be either CR or FA
		if memcond[i] == 'New':
			#non-responses get counted as correct rejection if its new
			if respconv[i] == 1 or respconv[i] == 2 or respconv[i] == 0:
				correct_rejection = np.append(correct_rejection, respconv.index[i-408])
			
			elif respconv[i] == 3 or respconv[i] == 4:
				false_alarm = np.append(false_alarm, respconv.index[i-408])
		
		elif memcond[i] == 'Old':
			#non-responses get counted as misses if they're old
			if respconv[i] == 1 or respconv[i] == 2 or respconv[i] == 0:
				miss = np.append(miss, respconv.index[i-408])
			
			elif respconv[i] == 3 or respconv[i] == 4:
				hit = np.append(hit, respconv.index[i-408])

	#count up old and new 
	#(I could do this manually since these shouldn't change, but I'm paranoid and this is my check_sum)
	old = len(memcond[memcond == 'Old'])
	new = len(memcond[memcond == 'New'])

	#calculate overall false alarm rate by condition
	CSplus_false_alarm_rate = len(condition[false_alarm][condition == 'CS+']) / (new /2)
	CSmin_false_alarm_rate = len(condition[false_alarm][condition == 'CS-']) / (new /2)

	#calculate overall hit rate by condition
	CSplus_hit_rate = len(condition[hit][condition == 'CS+']) / (old/2)
	CSmin_hit_rate = len(condition[hit][condition == 'CS-']) / (old/2)

	#calculate overall corrected recognition by condition
	CSplus_corrected_recognition = CSplus_hit_rate - CSplus_false_alarm_rate
	CSmin_corrected_recognition = CSmin_hit_rate - CSmin_false_alarm_rate


	#lets break it up into phase
	#since false alarm rate isn't calculate phase by phase, all we need are the location of the hits on day1
	hit_index = []
	[[hit_index.append(i) for i, stim in enumerate(day1_stims) if stim == day2_stims[target]] for target in hit]

	#the next section goes through each block in each phase and totals up the hits
	
	#these variables store the hit count for each block,
	#they are in the format [CS+ hits, CS- hits]
	#first phase
	blk1_1 = [0,0]
	blk1_2 = [0,0]
	blk1_3 = [0,0]
	blk1_4 = [0,0]
	blk1_5 = [0,0]
	blk1_6 = [0,0]

	#second phase
	blk2_1 = [0,0]
	blk2_2 = [0,0]
	blk2_3 = [0,0]
	blk2_4 = [0,0]
	blk2_5 = [0,0]
	blk2_6 = [0,0]

	#third phase
	blk3_1 = [0,0]
	blk3_2 = [0,0]
	blk3_3 = [0,0]
	blk3_4 = [0,0]
	blk3_5 = [0,0]
	blk3_6 = [0,0]

	#for each stim, take its location from day1 and find its correct block
	#then sum the hits by condition
	for stim in hit_index:
		#phase1, baseline
		if stim < 8:
			if day1_condition[stim] == 'CS+':
				blk1_1[0] += 1
			elif day1_condition[stim] == 'CS-':
				blk1_1[1] += 1
		elif stim >= 8 and stim < 16:
			if day1_condition[stim] == 'CS+':
				blk1_2[0] += 1
			elif day1_condition[stim] == 'CS-':
				blk1_2[1] += 1
		elif stim >= 16 and stim < 24:
			if day1_condition[stim] == 'CS+':
				blk1_3[0] += 1
			elif day1_condition[stim] == 'CS-':
				blk1_3[1] += 1
		elif stim >= 24 and stim < 32:
			if day1_condition[stim] == 'CS+':
				blk1_4[0] += 1
			elif day1_condition[stim] == 'CS-':
				blk1_4[1] += 1
		elif stim >= 32 and stim < 40:
			if day1_condition[stim] == 'CS+':
				blk1_5[0] += 1
			elif day1_condition[stim] == 'CS-':
				blk1_5[1] += 1
		elif stim >= 40 and stim < 48:
			if day1_condition[stim] == 'CS+':
				blk1_6[0] += 1
			elif day1_condition[stim] == 'CS-':
				blk1_6[1] += 1
		
		#phase2, fear day1_conditioning
		elif stim >= 48 and stim < 56:
			if day1_condition[stim] == 'CS+':
				blk2_1[0] += 1
			elif day1_condition[stim] == 'CS-':
				blk2_1[1] += 1
		elif stim >= 56 and stim < 64:
			if day1_condition[stim] == 'CS+':
				blk2_2[0] += 1
			elif day1_condition[stim] == 'CS-':
				blk2_2[1] += 1
		elif stim >= 64 and stim < 72:
			if day1_condition[stim] == 'CS+':
				blk2_3[0] += 1
			elif day1_condition[stim] == 'CS-':
				blk2_3[1] += 1
		elif stim >= 72 and stim < 80:
			if day1_condition[stim] == 'CS+':
				blk2_4[0] += 1
			elif day1_condition[stim] == 'CS-':
				blk2_4[1] += 1
		elif stim >= 80 and stim < 88:
			if day1_condition[stim] == 'CS+':
				blk2_5[0] += 1
			elif day1_condition[stim] == 'CS-':
				blk2_5[1] += 1
		elif stim >= 88 and stim < 96:
			if day1_condition[stim] == 'CS+':
				blk2_6[0] += 1
			elif day1_condition[stim] == 'CS-':
				blk2_6[1] += 1
		
		#phase3, extinction
		elif stim >= 96 and stim < 104:
			if day1_condition[stim] == 'CS+':
				blk3_1[0] += 1
			elif day1_condition[stim] == 'CS-':
				blk3_1[1] += 1
		elif stim >= 104 and stim < 112:
			if day1_condition[stim] == 'CS+':
				blk3_2[0] += 1
			elif day1_condition[stim] == 'CS-':
				blk3_2[1] += 1
		elif stim >= 112 and stim < 120:
			if day1_condition[stim] == 'CS+':
				blk3_3[0] += 1
			elif day1_condition[stim] == 'CS-':
				blk3_3[1] += 1
		elif stim >= 120 and stim < 128:
			if day1_condition[stim] == 'CS+':
				blk3_4[0] += 1
			elif day1_condition[stim] == 'CS-':
				blk3_4[1] += 1
		elif stim >= 128 and stim < 136:
			if day1_condition[stim] == 'CS+':
				blk3_5[0] += 1
			elif day1_condition[stim] == 'CS-':
				blk3_5[1] += 1
		elif stim >= 136 and stim < 144:
			if day1_condition[stim] == 'CS+':
				blk3_6[0] += 1
			elif day1_condition[stim] == 'CS-':
				blk3_6[1] += 1


	#now we need to convert the [block hit counts] into corrected recognition rates! 
	#so were going to dived each hit count by 4, and then subtract the appropriate CS false_alarm rate for each condition
	#phase1
	blk1_1[0] = (blk1_1[0] / 4) - CSplus_false_alarm_rate
	blk1_1[1] = (blk1_1[1] / 4) - CSmin_false_alarm_rate
	blk1_2[0] = (blk1_2[0] / 4) - CSplus_false_alarm_rate
	blk1_2[1] = (blk1_2[1] / 4) - CSmin_false_alarm_rate
	blk1_3[0] = (blk1_3[0] / 4) - CSplus_false_alarm_rate
	blk1_3[1] = (blk1_3[1] / 4) - CSmin_false_alarm_rate
	blk1_4[0] = (blk1_4[0] / 4) - CSplus_false_alarm_rate
	blk1_4[1] = (blk1_4[1] / 4) - CSmin_false_alarm_rate
	blk1_5[0] = (blk1_5[0] / 4) - CSplus_false_alarm_rate
	blk1_5[1] = (blk1_5[1] / 4) - CSmin_false_alarm_rate
	blk1_6[0] = (blk1_6[0] / 4) - CSplus_false_alarm_rate
	blk1_6[1] = (blk1_6[1] / 4) - CSmin_false_alarm_rate

	#phase2
	blk2_1[0] = (blk2_1[0] / 4) - CSplus_false_alarm_rate
	blk2_1[1] = (blk2_1[1] / 4) - CSmin_false_alarm_rate
	blk2_2[0] = (blk2_2[0] / 4) - CSplus_false_alarm_rate
	blk2_2[1] = (blk2_2[1] / 4) - CSmin_false_alarm_rate
	blk2_3[0] = (blk2_3[0] / 4) - CSplus_false_alarm_rate
	blk2_3[1] = (blk2_3[1] / 4) - CSmin_false_alarm_rate
	blk2_4[0] = (blk2_4[0] / 4) - CSplus_false_alarm_rate
	blk2_4[1] = (blk2_4[1] / 4) - CSmin_false_alarm_rate
	blk2_5[0] = (blk2_5[0] / 4) - CSplus_false_alarm_rate
	blk2_5[1] = (blk2_5[1] / 4) - CSmin_false_alarm_rate
	blk2_6[0] = (blk2_6[0] / 4) - CSplus_false_alarm_rate
	blk2_6[1] = (blk2_6[1] / 4) - CSmin_false_alarm_rate

	#phase3
	blk3_1[0] = (blk3_1[0] / 4) - CSplus_false_alarm_rate
	blk3_1[1] = (blk3_1[1] / 4) - CSmin_false_alarm_rate
	blk3_2[0] = (blk3_2[0] / 4) - CSplus_false_alarm_rate
	blk3_2[1] = (blk3_2[1] / 4) - CSmin_false_alarm_rate
	blk3_3[0] = (blk3_3[0] / 4) - CSplus_false_alarm_rate
	blk3_3[1] = (blk3_3[1] / 4) - CSmin_false_alarm_rate
	blk3_4[0] = (blk3_4[0] / 4) - CSplus_false_alarm_rate
	blk3_4[1] = (blk3_4[1] / 4) - CSmin_false_alarm_rate
	blk3_5[0] = (blk3_5[0] / 4) - CSplus_false_alarm_rate
	blk3_5[1] = (blk3_5[1] / 4) - CSmin_false_alarm_rate
	blk3_6[0] = (blk3_6[0] / 4) - CSplus_false_alarm_rate
	blk3_6[1] = (blk3_6[1] / 4) - CSmin_false_alarm_rate

	
	#now create some results structures to send out

	#first average corrected recognition by phase and condition
	baseline_csplus = np.mean([blk1_1[0],blk1_2[0],blk1_3[0],blk1_4[0],blk1_5[0],blk1_6[0]])
	baseline_csmin = np.mean([blk1_1[1],blk1_2[1],blk1_3[1],blk1_4[1],blk1_5[1],blk1_6[1]])

	fearcond_csplus = np.mean([blk2_1[0],blk2_2[0],blk2_3[0],blk2_4[0],blk2_5[0],blk2_6[0]])
	fearcond_csmin = np.mean([blk2_1[1],blk2_2[1],blk2_3[1],blk2_4[1],blk2_5[1],blk2_6[1]])

	extinction_csplus = np.mean([blk3_1[0],blk3_2[0],blk3_3[0],blk3_4[0],blk3_5[0],blk3_6[0]])
	extinction_csmin = np.mean([blk3_1[1],blk3_2[1],blk3_3[1],blk3_4[1],blk3_5[1],blk3_6[1]])


	#left it off here, basically im trying to recreate the variabe 'scg' in my R graphing script
		#dont forget to include false alarm rate in your ourput with phase
	
	data_byphase = pd.Series([baseline_csplus,baseline_csmin,fearcond_csplus,fearcond_csmin,extinction_csplus,extinction_csmin,CSplus_false_alarm_rate,CSmin_false_alarm_rate])

	#R-graphing
	r_phase_cr[iteration] = data_byphase


	#python-graphing
	ind_phase_cr = pd.DataFrame([],columns=['Subj','Phase','Condition','CR'])
	ind_phase_cr['Phase'] = ['Baseline','Baseline','Fear_Conditioning','Fear_Conditioning','Extinction','Extinction','False_Alarm','False_Alarm']
	ind_phase_cr['Condition'] = ['CS+','CS-','CS+','CS-','CS+','CS-','CS+','CS-',]
	ind_phase_cr['CR'] = data_byphase
	ind_phase_cr['Subj'] =  iteration + 1 
	
	phase_cr_long = pd.concat([phase_cr_long,ind_phase_cr])

	#do the same thing by block

	data_byblock = pd.Series([blk1_1[0] , blk1_1[1] , blk1_2[0] , blk1_2[1] , blk1_3[0], blk1_3[1],blk1_4[0] ,blk1_4[1] ,blk1_5[0] , blk1_5[1] , blk1_6[0] , blk1_6[1] ,blk2_1[0] , blk2_1[1] , blk2_2[0], blk2_2[1] ,blk2_3[0], blk2_3[1] , blk2_4[0] ,blk2_4[1] , blk2_5[0] , blk2_5[1], blk2_6[0] ,blk2_6[1] , blk3_1[0] ,blk3_1[1] , blk3_2[0] , blk3_2[1] , blk3_3[0], blk3_3[1] , blk3_4[0] ,blk3_4[1] , blk3_5[0] , blk3_5[1] , blk3_6[0] , blk3_6[1]])

	r_block_cr[iteration] = data_byblock


	ind_block_cr = pd.DataFrame([],columns=['Subj','Phase','Block','Condition','CR'])
	ind_block_cr['Phase'] = ['Baseline'] * 12 + ['Fear_Conditioning'] * 12 + ['Extinction'] * 12
	ind_block_cr['Condition'] = ['CS+', 'CS-'] * 18
	ind_block_cr['Block'] = ['Block_1']*2 + ['Block_2']*2 + ['Block_3']*2 + ['Block_4']*2 + ['Block_5']*2 + ['Block_6']*2 + ['Block_7']*2 + ['Block_8']*2 + ['Block_9']*2 + ['Block_10']*2 + ['Block_11']*2 + ['Block_12']*2 + ['Block_13']*2 + ['Block_14']*2 + ['Block_15']*2 + ['Block_16']*2 + ['Block_17']*2 + ['Block_18']*2 
	ind_block_cr['Subj'] = [subj[iteration]] * 36
	ind_block_cr['CR'] = blk1_1[0] , blk1_1[1] , blk1_2[0] , blk1_2[1] , blk1_3[0], blk1_3[1],blk1_4[0] ,blk1_4[1] ,blk1_5[0] , blk1_5[1] , blk1_6[0] , blk1_6[1] ,blk2_1[0] , blk2_1[1] , blk2_2[0], blk2_2[1] ,blk2_3[0], blk2_3[1] , blk2_4[0] ,blk2_4[1] , blk2_5[0] , blk2_5[1], blk2_6[0] ,blk2_6[1] , blk3_1[0] ,blk3_1[1] , blk3_2[0] , blk3_2[1] , blk3_3[0], blk3_3[1] , blk3_4[0] ,blk3_4[1] , blk3_5[0] , blk3_5[1] , blk3_6[0] , blk3_6[1]

	block_cr_long = pd.concat([block_cr_long, ind_block_cr])


	#now that we've done memory, lets look at shock expectancy
	#this will involve their responses (1 or 2) during fear conditioning, extinction, and extinction recall
	#fear conditioning is easy
	#fear_shock = meta['stim.RESP'][meta.phase == 'fearconditioning'][meta.cstype == 'CS+']
	#extinction, we need to use the location of the first line of each trial from the eprime log
	#ext_shock = meta['stim.RESP'][phase3_unique_loc][meta.cstype == 'CS+']
	#extinction recall
	#exre_shock = meta['stim.RESP'][meta.phase == 'extinctionRecall'][meta.cstype == 'CS+']

	#shock_expectancy = pd.Series(np.zeros(60))
	#shock_expectancy[0:24] = fear_shock
	#shock_expectancy[24:48] = ext_shock
	#shock_expectancy[48:60] = exre_shock

	#expect_sum = []
	#run_sum = 0
	#for expect in shock_expectancy:
	#	if expect == 1:
	#		run_sum += 1
	#	elif expect == 2:
	#		run_sum -= 1
	#	expect_sum.append(run_sum)

	#shock_expectancy['Response'] = pd.concat(fear_shock, ext_shock, exre_shock)
	#shock_expectancy.Response[0:48] = fear_shock
	#shock_expectancy.Response[48:96] = ext_shock
	#shock_expectancy.Response[96:120] = exre_shock







#now we need to come out of the loop, and average across each subject for phase and block
phase_cr['CR'] = [np.mean(r_phase_cr.loc[0,:]),np.mean(r_phase_cr.loc[1,:]),np.mean(r_phase_cr.loc[2,:]),np.mean(r_phase_cr.loc[3,:]),np.mean(r_phase_cr.loc[4,:]),np.mean(r_phase_cr.loc[5,:]),np.mean(r_phase_cr.loc[6,:]),np.mean(r_phase_cr.loc[7,:])]
phase_cr['SEM'] = [stats.sem(r_phase_cr.loc[0,:]),stats.sem(r_phase_cr.loc[1,:]),stats.sem(r_phase_cr.loc[2,:]),stats.sem(r_phase_cr.loc[3,:]),stats.sem(r_phase_cr.loc[4,:]),stats.sem(r_phase_cr.loc[5,:]),stats.sem(r_phase_cr.loc[6,:]),stats.sem(r_phase_cr.loc[7,:])]


block_cr['CR'] = [np.mean(r_block_cr.loc[0,:]),np.mean(r_block_cr.loc[1,:]),np.mean(r_block_cr.loc[2,:]),np.mean(r_block_cr.loc[3,:]),np.mean(r_block_cr.loc[4,:]),np.mean(r_block_cr.loc[5,:]),np.mean(r_block_cr.loc[6,:]),np.mean(r_block_cr.loc[7,:]),np.mean(r_block_cr.loc[8,:]),np.mean(r_block_cr.loc[9,:]),np.mean(r_block_cr.loc[10,:]),np.mean(r_block_cr.loc[11,:]),np.mean(r_block_cr.loc[12,:]),np.mean(r_block_cr.loc[13,:]),np.mean(r_block_cr.loc[14,:]),np.mean(r_block_cr.loc[15,:]),np.mean(r_block_cr.loc[16,:]),np.mean(r_block_cr.loc[17,:]),np.mean(r_block_cr.loc[18,:]),np.mean(r_block_cr.loc[19,:]),np.mean(r_block_cr.loc[20,:]),np.mean(r_block_cr.loc[21,:]),np.mean(r_block_cr.loc[22,:]),np.mean(r_block_cr.loc[23,:]),np.mean(r_block_cr.loc[24,:]),np.mean(r_block_cr.loc[25,:]),np.mean(r_block_cr.loc[26,:]),np.mean(r_block_cr.loc[27,:]),np.mean(r_block_cr.loc[28,:]),np.mean(r_block_cr.loc[29,:]),np.mean(r_block_cr.loc[30,:]),np.mean(r_block_cr.loc[31,:]),np.mean(r_block_cr.loc[32,:]),np.mean(r_block_cr.loc[33,:]),np.mean(r_block_cr.loc[34,:]),np.mean(r_block_cr.loc[35,:])]
block_cr['SEM'] = [stats.sem(r_block_cr.loc[0,:]),stats.sem(r_block_cr.loc[1,:]),stats.sem(r_block_cr.loc[2,:]),stats.sem(r_block_cr.loc[3,:]),stats.sem(r_block_cr.loc[4,:]),stats.sem(r_block_cr.loc[5,:]),stats.sem(r_block_cr.loc[6,:]),stats.sem(r_block_cr.loc[7,:]),stats.sem(r_block_cr.loc[8,:]),stats.sem(r_block_cr.loc[9,:]),stats.sem(r_block_cr.loc[10,:]),stats.sem(r_block_cr.loc[11,:]),stats.sem(r_block_cr.loc[12,:]),stats.sem(r_block_cr.loc[13,:]),stats.sem(r_block_cr.loc[14,:]),stats.sem(r_block_cr.loc[15,:]),stats.sem(r_block_cr.loc[16,:]),stats.sem(r_block_cr.loc[17,:]),stats.sem(r_block_cr.loc[18,:]),stats.sem(r_block_cr.loc[19,:]),stats.sem(r_block_cr.loc[20,:]),stats.sem(r_block_cr.loc[21,:]),stats.sem(r_block_cr.loc[22,:]),stats.sem(r_block_cr.loc[23,:]),stats.sem(r_block_cr.loc[24,:]),stats.sem(r_block_cr.loc[25,:]),stats.sem(r_block_cr.loc[26,:]),stats.sem(r_block_cr.loc[27,:]),stats.sem(r_block_cr.loc[28,:]),stats.sem(r_block_cr.loc[29,:]),stats.sem(r_block_cr.loc[30,:]),stats.sem(r_block_cr.loc[31,:]),stats.sem(r_block_cr.loc[32,:]),stats.sem(r_block_cr.loc[33,:]),stats.sem(r_block_cr.loc[34,:]),stats.sem(r_block_cr.loc[35,:])]

#only save the output if its a complete analysis
if args.subj == 'all':


	phase_cr.to_csv('%s/graphing/ggbp.csv'%(data_dir),sep=',')
	block_cr.to_csv('%s/graphing/ggbb.csv'%(data_dir),sep=',')
	#if its not working, then check non-responses
	phase_cr_long.to_csv('%s/graphing/ggbp_long.csv'%(data_dir),sep=',')
	block_cr_long.to_csv('%s/graphing/ggbb_long.csv'%(data_dir),sep=',')



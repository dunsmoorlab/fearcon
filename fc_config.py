import pandas as pd
import numpy as np
import itertools as it


run_key = {
	'BASELINE': 'run001',
	'FEAR_CONDITIONING': 'run002',
	'EXTINCTION': 'run003',
	'EXTINCTION_RECALL': 'run004',
	'MEMORY_RUN_1': 'run005',
	'MEMORY_RUN_2': 'run006',
	'MEMORY_RUN_3': 'run007',
	'LOCALIZER_1': 'run008',
	'LOCALIZER_2': 'run009',
	'MPRAGE': 'struct',
}



#cond2 = {
#	0: 'rest',
#	1: 'CS-',
#	2: 'CS+',
#	3: 'scene',
#}

#these dictionaries are for FC_timing
#this dictionary lets us be more precise in our code, as experimentally some runs have the same structure
#IMPORTANT: need to manually set which phase is currently the 'tag phase' for day 1
day2phase = {
	'baseline': 'day1',
	'fear_conditioning': 'day1',
	'extinction': 'day1_tag',
	'extinction_recall': 'day2_er' ,
	'memory_run_1': 'day2_mem',
	'memory_run_2': 'day2_mem',
	'memory_run_3': 'day2_mem',
	'localizer_1': 'day2_loc',
	'localizer_2': 'day2_loc',
}

#this is a dictionary in format of phase:[initialITI, finalITI, number of ITIs(not counting first and last)]
#all time in miliseconds
time2add = {
	'day1': [4000,2000,48],
	'day1_tag': [4000,2000,48],
	'day2_er': [8000,2000,24],
	'day2_mem': [4000,2000,80],
	#localizer is a little different [initialITI, stimdur, ITI, IMBI, IBI]
	'day2_loc': [4000, 1500, 500, 6000, 12000]
}

#the localizer stim categories
LocStims = {
	'anim': 'animal',
	'tool': 'tool',
	'indo': 'indoor',
	'loca': 'outdoor',
	'scra': 'scrambled',
}


phase2location = {
	'baseline': 'day1/run001/zd_mc_run001.npy',
	'fear_conditioning': 'day1/run002/zd_mc_run002.npy',
	'extinction': 'day1/run003/zd_mc_run003.npy',
	'extinction_recall': 'day2/run004/zd_mc_run004.npy' ,
	'memory_run_1': 'day2/run005/zd_mc_run005.npy',
	'memory_run_2': 'day2/run006/zd_mc_run006.npy',
	'memory_run_3': 'day2/run007/zd_mc_run007.npy',
	'localizer_1': 'day2/run008/zd_mc_run008.npy',
	'localizer_2': 'day2/run009/zd_mc_run009.npy',
}


mvpa_prepped = {
	'baseline': 'day1/run001/prepped_run001.npy',
	'fear_conditioning': 'day1/run002/prepped_run002.npy',
	'extinction': 'day1/run003/prepped_run003.npy',
	'extinction_recall': 'day2/run004/prepped_run004.npy' ,
	'memory_run_1': 'day2/run005/prepped_run005.npy',
	'memory_run_2': 'day2/run006/prepped_run006.npy',
	'memory_run_3': 'day2/run007/prepped_run007.npy',
	'localizer_1': 'day2/run008/prepped_run008.npy',
	'localizer_2': 'day2/run009/prepped_run009.npy',
}






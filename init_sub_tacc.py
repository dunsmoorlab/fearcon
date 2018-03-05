import os
from glob import glob
import sys
import argparse

parser = argparse.ArgumentParser(description='Function arguments')

parser.add_argument('-s', '--subj',nargs='+', help='Subject Number', default=0, type=int)
args=parser.parse_args()

data_dir = '$WORK/FearCon/'
for sub in args.subj:


	SUBJ = 'Sub{0:0=3d}'.format(sub)

	subj_dir = data_dir + SUBJ + os.sep

	os.mkdir(subj_dir)

	anatomy = subj_dir + 'anatomy'
	behavior = subj_dir + 'behavior'
	bold = subj_dir + 'bold'
	mask = subj_dir + 'mask'
	model = subj_dir + 'model'
	SCR = subj_dir + 'SCR'
	raw = subj_dir + 'raw'

	os.mkdir(anatomy)
	os.mkdir(behavior)
	os.mkdir(bold)
	os.mkdir(mask)
	os.mkdir(model)
	os.mkdir(SCR)
	os.mkdir(raw)

	converted = behavior + '/converted'
	raw_log_files = behavior + '/raw_log_files'
	run_logs = behavior + '/run_logs'

	os.mkdir(converted)
	os.mkdir(raw_log_files)
	os.mkdir(run_logs)

	day1 = bold + '/day1'
	day2 = bold + '/day2'

	os.mkdir(day1)
	os.mkdir(day2)

	run1 = day1 + '/run001'
	run2 = day1 + '/run002'
	run3 = day1 + '/run003'
	run4 = day2 + '/run004'
	run5 = day2 + '/run005'
	run6 = day2 + '/run006'
	run7 = day2 + '/run007'
	run8 = day2 + '/run008'
	run9 = day2 + '/run009'

	os.mkdir(run1)
	os.mkdir(run2)
	os.mkdir(run3)
	os.mkdir(run4)
	os.mkdir(run5)
	os.mkdir(run6)
	os.mkdir(run7)
	os.mkdir(run8)
	os.mkdir(run9)

	GLM = model + '/GLM'
	MVPA = model + '/MVPA'

	os.mkdir(GLM)
	os.mkdir(MVPA)

	onsets = GLM + '/onsets'
	labels = MVPA + '/labels'
	toolbox = MVPA + '/toolbox'

	os.mkdir(onsets)
	os.mkdir(labels)
	os.mkdir(toolbox)

	run1o = onsets + '/run001'
	run2o = onsets + '/run002'
	run3o = onsets + '/run003'
	run4o = onsets + '/run004'
	run5o = onsets + '/run005'
	run6o = onsets + '/run006'
	run7o = onsets + '/run007'
	run8o = onsets + '/run008'
	run9o = onsets + '/run009'

	os.mkdir(run1o)
	os.mkdir(run2o)
	os.mkdir(run3o)
	os.mkdir(run4o)
	os.mkdir(run5o)
	os.mkdir(run6o)
	os.mkdir(run7o)
	os.mkdir(run8o)
	os.mkdir(run9o)

	BASELINE = raw + '/BASELINE'
	EXTINCTION = raw + '/EXTINCTION'
	EXTINCTION_RECALL = raw + '/EXTINCTION_RECALL'
	FEAR_CONDITIONING = raw + '/FEAR_CONDITIONING'
	LOCALIZER_1 = raw + '/LOCALIZER_1'
	LOCALIZER_2 = raw + '/LOCALIZER_2'
	MEMORY_RUN_1 = raw + '/MEMORY_RUN_1'
	MEMORY_RUN_2 = raw + '/MEMORY_RUN_2'
	MEMORY_RUN_3 = raw + '/MEMORY_RUN_3'
	MPRAGE = raw + '/MPRAGE'
	scan_folders = raw + '/scan_folders'

	os.mkdir(BASELINE)
	os.mkdir(EXTINCTION)
	os.mkdir(EXTINCTION_RECALL)
	os.mkdir(FEAR_CONDITIONING)
	os.mkdir(LOCALIZER_1)
	os.mkdir(LOCALIZER_2)
	os.mkdir(MEMORY_RUN_1)
	os.mkdir(MEMORY_RUN_2)
	os.mkdir(MEMORY_RUN_3)
	os.mkdir(MPRAGE)
	os.mkdir(scan_folders)

	day1scr = SCR + '/Day1'
	day2scr = SCR + '/Day2'

	os.mkdir(day1scr)
	os.mkdir(day2scr)

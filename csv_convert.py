import pandas as pd
import os
import sys
import numpy as np
import argparse
from sys import platform



parser = argparse.ArgumentParser(description='Function arguments')

#add arguments
parser.add_argument('-s', '--subj', nargs='+', help='Subject Number', default=0, type=str)

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

	txt_dir = '%s/%s/behavior/converted'%(data_dir,SUBJ)

	csv_dir = '%s/%s/behavior/run_logs'%(data_dir,SUBJ)

	for txt in os.listdir(txt_dir):
		if '.txt' in txt:
			os.system('ssconvert %s/%s %s/%s.csv'%(txt_dir,txt,csv_dir,txt[:-4]))

	for csv in os.listdir(csv_dir):
		if '.csv' in csv:
			del1 = pd.read_csv('%s/%s'%(csv_dir,csv),skiprows=1)

			del1.to_csv('%s/%s'%(csv_dir,csv), sep=',', index = False)

	elog = '%s_elog.csv'%(SUBJ)

	if elog in os.listdir(csv_dir):
		os.system('mv %s/%s %s/%s/behavior'%(csv_dir,elog,data_dir,SUBJ))
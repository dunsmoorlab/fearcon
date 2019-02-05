from fc_config import *
from preprocess_library import meta
from shutil import copytree
import os
from glm_timing import *


def copy_files(subs=None):

	# dest = '/Volumes/DunsmoorLab/Second_Submit_FC/dicom/'
	# dest = 'F:\\FearCon_Box\\dicom'
	dest = '/Volumes/DunsmoorRed/dicom/'
	runs = ['BASELINE','FEAR_CONDITIONING','EXTINCTION','EXTINCTION_RECALL','MPRAGE',
			'MEMORY_RUN_1','MEMORY_RUN_2','MEMORY_RUN_3','LOCALIZER_1','LOCALIZER_2']
	# runs = ['BASELINE','FEAR_CONDITIONING','EXTINCTION','EXTINCTION_RECALL','MPRAGE']


	for subj in subs:
		print(subj)
		sub = meta(subj)
		
		dcm = os.path.join(dest,'fc{0:03d}'.format(subj), 'ses-1')
		
		os.makedirs(dcm)

		for run in runs:
			if subj == 117 and run == 'LOCALIZER_2': pass
			else: copytree(os.path.join(sub.raw,run)+ os.sep, os.path.join(dcm,run) + os.sep)

def write_events(subs=None):

	wrk_dir = '/Volumes/DunsmoorLab/Second_Submit_FC/bids/'
	# wrk_dir = 'F:\\FearCon_Box\\dicom'
	for sub in subs:

		bids = 'sub-fcp%s_ses-1'%(str(sub)[-2:])

		sub_dir = os.path.join(wrk_dir, 'sub-fcp%s'%(str(sub)[-2:]), 'ses-1','func')

		for run in ['baseline','fear_conditioning','extinction','extinction_recall']:

			events = glm_timing(sub,run).phase_events()

			events = events[['onset','duration','trial_type']]

			if run == 'fear_conditioning':
				save_name = 'fearconditioning'
			elif run == 'extinction_recall':
				save_name = 'extinctionrecall'
			else:
				save_name = run

			path = sub_dir + os.sep + bids + '_task-%s_run-01_events.tsv'%(save_name)

			events.to_csv(path,sep='\t',index=False)




















# import re
# import json
# s = 'FCP001--Baseline--EP--7.json'
# w = 0
# while w < 1:
# 	try:
# 		fd = open(s, 'r')
# 		json_dict = json.load(fd)
# 		fd.close()
# 		break                    # parsing worked -> exit loop
# 	except Exception as e:
# 		# "Expecting , delimiter: line 34 column 54 (char 1158)"
# 		# position of unexpected character after '"'
# 		print(len(str(e)))
# 		print(str(e))
# 		unexp = int(re.findall(r'\(char (\d+)\)', str(e))[0])
# 		print(unexp)
# 		# position of unescaped '"' before that
# 		# unesc = s.rfind(r'"', 0, unexp)
# 		s = s[:unexp] + r'\"' + s[unexp+1:]
# 		# position of correspondig closing '"' (+2 for inserted '\')
# 		# closg = s.find(r'"', unesc + 2)
# 		# s = s[:closg] + r'\"' + s[closg+1:]
# 	w += 1
# print(result)
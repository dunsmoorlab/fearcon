"""Custom dicom to bids conversion script for NDA submission

assumes that you have installed dcm2niix and dcm2bids
	"conda install -c conda-forge dcm2niix"
	"pip install dcm2bids"

also assumes that you have all of the config files in the dicom folder
	i.e. CCX_dcm/ccx_ses-1.json
	     CCX_dcm/ccx_ses-2.json
		 CCX_dcm/ccx_ses-3.json
		 CCX_dcm/ccx_ses-3-pre-10.json
		 CCX_dcm/ccx_nda.json

to run:
	1. navigate terminal to the directory one level above the dicom folder
		i.e. current_dir/CCX_dcm
	2. initialize output folder by running:
		"python custom_bids.py -i True -o CCX-bids"
	3. convert a single subject by replacing CCX000 with the desired subject and running: 
		"python custom_bids.py -s CCX000 -d CCX_dcm -o CCX-bids"
	4. to convert just a single session, use the -e flag:
		"python custom_bids.py -s CCX000 -d CCX_dcm -o CCX-bids -e 3"

display this message - python custom_bids.py --help

written by gus hennings
"""

#imports
import os
import sys
import argparse
import json
from shutil import copyfile
from warnings import warn
from fc_config import *

#set up argparse
parser = argparse.ArgumentParser(description=__doc__,
	formatter_class=argparse.RawDescriptionHelpFormatter)

parser.add_argument('-s', '--subj',  default='s666',type=str, help='subject #')
parser.add_argument('-d', '--dcm',   default=None,  type=str, help='dicom directory')
parser.add_argument('-o', '--out',   default=None,  type=str, help='bids outputdir')
parser.add_argument('-n', '--nda', 	 default=False, type=bool,help='NDA conversion for CCX')
parser.add_argument('-i', '--init',  default=False, type=bool,help='initialize the output')
args = parser.parse_args()


output = os.path.join(os.getcwd(),args.out) #find the output dir

#check to see if we are just initializing the output
if args.init:
	print('initializing output')
	if not os.path.exists(output): os.mkdir(output) #create output if doesn't exists
	for file_name in ['README','participants.tsv','CHANGES','dataset_description.json']: #create bids necessary files 
		if file_name == 'README': 
			with open(os.path.join(output,file_name),'w+') as file:
				file.write('FC bids'); file.close() #readme can't be empty
		else: open(os.path.join(output,file_name),'w+') #just make empty files with 'w+'

	deriv = os.path.join(output,'derivatives') #we'll probably need this
	if not os.path.exists(deriv): os.mkdir(deriv)

	sys.exit()

# data_dir = os.path.join(os.getcwd(),args.dcm) #find the dicom dir

sub_arg = 'FC%s'%(args.subj[-3:]) #this is format the dicom folders are in
dcm = os.path.join(data_dir,'Sub'+args.subj[-3:],'raw')
#make sure we can find all three dicom folders for this subject
# day1dcm = os.path.join(data_dir,sub_arg + '_01')
# day2dcm = os.path.join(data_dir,sub_arg + '_02')
# day3dcm = os.path.join(data_dir,sub_arg + '_03')
# dir_list = [day1dcm,day2dcm,day3dcm]
# if args.sess is not 0: dir_list = [dir_list[int(args.sess - 1)]]
# #and raise an error if we can't
# for _dir in dir_list:
# 	try: assert os.path.exists(_dir)
# 	except: print('Dicom directory not found: %s'%(_dir)); sys.exit()

# print('Found 3 dicom folders for %s'%(args.subj)) #we found all the dirs

#############################################################
#if not running conversion for NDA, convert each session one at a time
# dcm_list = [day1dcm,day2dcm,day3dcm]
# if args.sess is not 0: dcm_list = [dcm_list[int(args.sess - 1)]]
# for i, dcm in enumerate(dcm_list):
	# if args.sess is 0: ses = i+1 #fix pythonic indexing (no ses-0)
	# else: ses = args.sess	

config = 'fc-bids-config.json' #load the config file
# dcm = args.dcm	
try: assert os.path.exists(config) #and make sure it exists
except: print('config file does not exist: %s'%(config)); sys.exit()

dcm2bids_call = 'dcm2bids -d %s -c %s -p %s -o %s -s 1'%(dcm,config,args.subj,output) #set up the command
	
try: os.system(dcm2bids_call) #and run it
except: print('bids conversion failed with call: %s'%(dcm2bids_call)); sys.exit()
	
os.system('rm -R %s'%(os.path.join(output,'tmp_dcm2bids'))) #delete the tmp folder that dcm2bids makes

#write in the task name to the json - doesn't happen in the conversion for some reason
#need this for bids validation
#very lazy loop - scans all files in the subject directory
for _dir in os.walk(output):
	for file in _dir[2]:
		if 'task' in file and '.json' in file:
			with open(os.path.join(_dir[0],file)) as json_file:
				json_decoded = json.load(json_file)

			name_start = file.find('task-') + 5 #file is a string
			try: 
				name_end = file.find('_run')
				assert name_end != -1
			except: 
				name_end = file.find('_bold')

			json_decoded['TaskName'] = file[name_start:name_end]

			with open(os.path.join(_dir[0],file), 'w') as json_file:
				json.dump(json_decoded, json_file)

sub_dir = os.path.join(output,'sub-'+sub_arg)
ses1 = os.path.join(sub_dir,'ses-1','func')
ses2 = os.path.join(sub_dir,'ses-2','func')
os.makedirs(ses2)
for file in os.listdir(ses1):
	if 'renewal' in file or 'memory' in file or 'localizer' in file:
		dest = file.replace('ses-1','ses-2')
		os.system('mv %s %s'%(os.path.join(ses1,file),os.path.join(ses2,dest)))
print('Bids conversion for sub-%s complete!'%(args.subj))
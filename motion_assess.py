#my take on J.M.'s script to assess motion for GLM
#update this as we go, right now its just built for 2 subjects and just for baseline and fear conditioning

import os
import sys
import subprocess
from glob import glob
import argparse

#point to experiment folder
data_dir = '/Users/ach3377/GoogleDrive/FC_FMRI_DATA'

parser = argparse.ArgumentParser(description='Function arguments')

parser.add_argument('-s', '--subj', nargs='+', help='Subject Number', default=0, type=int)
parser.add_argument('-o', '--overwrite', help='Overwrite output html', default=False, type=bool)

args=parser.parse_args()

#name output file and bad bold list
outhtml = '/Users/ach3377/GoogleDrive/FC_FMRI_DATA/motion_assess/bold_motion_QA.html'
out_bad_bold_list = '/Users/ach3377/GoogleDrive/FC_FMRI_DATA/motion_assess/bad_subj_no_donut.txt'

if args.overwrite == True:
	#remove previous versions of the output files, since html only appends and doesn't overwrite
	os.system('rm %s'%(outhtml))
	os.system('rm %s'%(out_bad_bold_list))

for subj in args.subj:
	
	SUBJ = 'Sub00%s'%(subj)
	
	#collect all the runs
	bold_files = glob('%s/%s/bold/day1/run00[1-9]/run00[1-9].nii.gz'%(data_dir,SUBJ))



		

	#create the html again
	#with open(outhtml, 'w'):
	#	pass

	#run the motion assessment
	for file in list(bold_files):
		print(file)
		#store directory name
		file_dir = os.path.dirname(file)

		#strip off .nii.gz
		#cur_bold_no_nii = cur_bold[:-7]

		#run motion assessment
		if os.path.isdir('%s/motion_assess/'%(file_dir)) == False:
			os.system('mkdir %s/motion_assess'%(file_dir))
		
		os.system(
			'fsl_motion_outliers' + ' '
			'-i' + ' ' + file + ' '
			'-o' + ' ' + file_dir + '/motion_assess/confound.txt' + ' '
			'--fd' + ' ' + '--thresh=0.9' + ' '
			'-p' + ' ' + file_dir + '/motion_assess/fd_plot' + ' '
			'-v >' + ' ' + file_dir + '/motion_assess/outlier_output.txt')

		#create empty confound.txt files if they're aren't any for a subject,
		#so that we can model them out later (higher levels)
		if os.path.isfile("%s/motion_assess/confound.txt"%(file_dir))==False:
			os.system("touch %s/motion_assess/confound.txt"%(file_dir))

		#put them all in a html
		os.system("cat %s/motion_assess/outlier_output.txt >> %s"%(file_dir, outhtml))
		os.system("echo '<p>=============<p>FD plot %s <br><IMG BORDER=0 SRC=%s/motion_assess/fd_plot.png WIDTH=100%s></BODY></HTML>' >> %s"%(file_dir, file_dir,'%', outhtml))


		#create list of subjects who are bad and should feel bad for moving so much
		#this is based on a percentage of scrubbed volumes, were going with 75% as an example (64 vols)
		output = subprocess.check_output('grep -o 1 %s/motion_assess/confound.txt | wc -l'%(file_dir), shell=True)
		num_scrub = [int(s) for s in output.split() if s.isdigit()]
		if num_scrub[0]>64:
			with open(out_bad_bold_list, "a") as myfile:
				myfile.write("%s\n"%(file_bold))

from glm_timing import glm_timing
from fc_config import *
from preprocess_library import meta
from shutil import copytree
import sys
import subprocess

if sys.platform.startswith('win'):
	dest = os.path.join('D:'+os.sep,'rotation_project')
else:
	dest = os.path.join('/Volumes','DunsmoorRed','rotation_project')#path to DunsmoorRed
mkdir(dest)

for sub in sub_args[:10]:
	#make the output directories
	subj = meta(sub)
	sub_dir  = os.path.join(dest, subj.fsub)
	behavior = os.path.join(sub_dir, 'behavior')
	dicom    = os.path.join(sub_dir, 'dicom')

	for d in [sub_dir, behavior, dicom]: mkdir(d)

	#create the behavior file
	events = glm_timing(sub,'fear_conditioning')
	events.df = events.phase_events(stims=True)
	events.df['US'] = (events.proc == 'CSUS').astype(bool).astype(int)
	events.df['phase'] = 'fear_conditioning'
	events.df['sub'] = subj.num
	events.df['response'] = events.phase_meta['stim.RESP'].values.astype(int)
	events.df.to_csv(os.path.join(behavior,'run002.csv'),index=False)

	#move over the dicom folders
	folders = {'run002':'FEAR_CONDITIONING',
			   'T1'    :'MPRAGE'}
	for f in folders:
		src = os.path.join(subj.raw,folders[f])
		dst = os.path.join(dicom,f)
		# fastcopy(src,dst)
		copytree(src,dst)



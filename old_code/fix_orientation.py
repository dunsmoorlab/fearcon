from preprocess_library import *
from fc_config import *


for sub in [105,106]:
	subj = meta(sub)

	for phase in ['baseline','fear_conditioning','extinction']:

		run = os.path.join(subj.bold_dir, raw_paths[phase])
		cmd = 'fslreorient2std %s %s'%(run, run)
		print(cmd)
		os.system(cmd)
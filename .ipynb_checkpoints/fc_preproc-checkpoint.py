from preprocess_library import preproc, meta
from fc_config import sub_args



#assumes that you want to run everything on every subject
#sub is the counter, and subj is the subject object
# for sub in sub_args:

# 	preproc(sub).fsl_mcflirt()



for sub in sub_args:
	preproc(sub).mvpa_prep()















	# def set_subjects(self, subject):
		
	# 	if subject == 'all':

	# 		self.subs = sub_args

	# 	else:

	# 		self.subs = subject
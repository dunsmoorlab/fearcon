from fc_config import *
from glm_timing import *
from preprocess_library import meta
from shutil import copytree, move


def pop_beta_timing():

	for sub in sub_args:

		for phase in ['memory_run_1','memory_run_2','memory_run_3']:

			glm_timing(sub,phase).betaseries()

def clean_old_betas():

	for sub in sub_args:

		subj = meta(sub)

		for phase in ['localizer_1','localizer_2']:

			rundir = subj.bold_dir + phase2rundir[phase]

			target = os.path.join(rundir,'old_betas')

			os.mkdir(target)

			move(os.path.join(rundir,'ls-s_betas'),target)

			move(os.path.join(rundir,'new_ls-s_betas'),target)



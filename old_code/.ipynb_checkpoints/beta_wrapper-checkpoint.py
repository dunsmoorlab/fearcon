from nistats_betas import generate_lss_betas

from fc_config import sub_args


def group_beta(phase=None,overwrite=False):

	for sub in sub_args:

		generate_lss_betas(sub=sub, phase=phase, display=False, overwrite=overwrite)


for phase in ['localizer_1','localizer_2']:

	group_beta(phase=phase)
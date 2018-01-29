from nistats_betas import generate_lss_betas

from fc_config import sub_args


def group_beta(phase=None,overwrite=False):

	for sub in sub_args:

		generate_lss_betas(subj=sub, phase=phase, tr=2, hrf_model='glover', display=False, overwrite=overwrite)

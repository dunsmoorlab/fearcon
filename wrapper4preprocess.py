from preprocess_library import preproc
from fc_config import *



#python motion_assess.py -s 101 102 103 104 108 109


for sub in [23,24,25,26]:
# for sub in [122,123,124,125]:
	do = preproc(sub)
	# do.convert_dicom()
	# do.recon_all()
	# do.convert_freesurfer()
	# do.register_freesurfer()
	# do.mc_first_pass()
	# do.epi_reg()
	# do.create_refvol()
	# do.prep_final_epi_part1()
	do.prep_final_epi_part2(get_files=True)
	do.new_mask()
	do.mvpa_prep()
	do.fsl_reg()
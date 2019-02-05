from preprocess_library import preproc
from fc_config import *

'''
Put the subejct numbers in [] in the for loop,
of if you want to do something to all of the control subjects put sub_args,
and p_sub_args for PTSD subjects (not in brackets)

I would caution you NOT to run a lot of things at once, as its good practice to do some 
quality control on each output. ESPECIALLY on the dicoms once they are made.

Otherwise this is the order of things.
'''

'''
remember that you have to run this command to create the folders:

python init_sub.py -s 110 111
'''


for sub in [114,115]:
	#this creates the preproc object, all steps are method of the preproc class
	do = preproc(sub)
	
	do.convert_dicom()
	# do.recon_all()
	# do.convert_freesurfer()
	# do.register_freesurfer()
	# do.mc_first_pass()
	# do.epi_reg()
	# do.create_refvol()
	# do.prep_final_epi_part1()
	# do.prep_final_epi_part2(get_files=True)
	# do.new_mask()
	# do.mvpa_prep()
	# do.fsl_reg()

from preprocess_library import meta
from fc_config import *

sub_args = [1]
phase = 'fear_conditioning'
for sub in sub_args:

	subj = meta(sub)

	bold_in = os.path.join(subj.bold_dir, nifti_paths[phase])
	bold_out = os.path.join(subj.bold_dir, phase2rundir[phase], 'MNI_' + nifti_paths[phase][-16:])

	if not os.path.exists(bold_out):
		print('warping to MNI space')
		
		struct_warp_img = os.path.join(subj.reg_dir, 'struct2std_warp.nii.gz')
		func2struct = os.path.join(subj.reg_dir, 'func2struct.mat')
		
		os.system('applywarp --in=%s --ref=$FSL_DIR/data/standard/MNI152_T1_1mm_brain.nii.gz --out=%s --premat=%s --warp=%s'%(
					bold_in, bold_out, func2struct, struct_warp_img))
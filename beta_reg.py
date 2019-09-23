from fc_config import *
from preprocess_library import meta


for sub in all_sub_args:
    subj = meta(sub)
    omat = os.path.join(subj.reg_dir, 'ref2std.mat')
    os.system('flirt -in %s -ref /usr/local/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz -dof 12 -omat %s' % (subj.refvol_be, omat))

    for phase in fsl_betas:
        new_img = os.path.join(subj.bold_dir, fsl_betas_std[phase])
        os.system('flirt -in %s -ref /usr/local/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz -out %s -init %s -applyxfm'%(os.path.join(subj.bold_dir, fsl_betas[phase]), new_img, omat))
        os.system('fslmaths %s -s 3.3973 %s'%(new_img, new_img))

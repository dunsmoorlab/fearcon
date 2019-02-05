from fc_config import *
import nibabel as nib


for run in ['run002']:

	CSp_hit = nib.load('%s/group_pyGLM/%s'%(data_dir,run))
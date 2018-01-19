# from glm_timing import glm_timing

# phase = 'fearconditioning'

# a = glm_timing(1)

# a.phase_events(phase)

# from pygmy import first_level
from nilearn.plotting import plot_stat_map, plot_anat, plot_img, plot_glass_brain
from nilearn import image
from nistats.reporting import plot_design_matrix
import matplotlib.pyplot as plt
from pyGLM_lev1 import level1_univariate
from L2_pygmy import second_level
from pygmy import first_level
import nibabel as nib


from fc_config import data_dir

plt.ion()

# a = first_level(1,'fear_conditioning')

# a.glm_stats(a.CSplus_minus_CSmin)

# plot_stat_map(a.z_map, bg_img=a.mean_func, threshold=3.0, display_mode='z', cut_coords=3, black_bg=True, title='zoobop')

# # a.load_bold('fear_conditioning')

# # vol1 = image.index_img(a.func,0)
# plot_img(vol1)

# level1_univariate('baseline',display=False)
# second_level('baseline',display=False)


# level1_univariate('fear_conditioning',display=False)
# second_level('fear_conditioning',display=False)

#what i need to figure out next for level 2/3 is how the transformations are actually taking place
#(i.e. fix the (:,:,:,1) problem)
#remember that you commented out a paragraph in niimg_conversions.py within nistats


#a = first_level(4,'fear_conditioning')
q = '%s/Sub004/model/pyGLM/run002/CSplus___CSmin_z_map.nii.gz'%(data_dir)


z = '%s/group_pyGLM/run002/CSplus___CSmin_level2_z_map.nii.gz'%(data_dir)

x = nib.load(z) 
plot_stat_map(z,threshold=2.3, display_mode='z', colorbar=True)
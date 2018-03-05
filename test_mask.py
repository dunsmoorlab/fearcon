from fc_config import data_dir, sub_args, nifti_paths
from pygmy import first_level
from glm_timing import glm_timing
from preprocess_library import meta
from L2_pygmy import second_level

import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import os

from nilearn.plotting import plot_stat_map, plot_glass_brain
from glob import glob

from nilearn.input_data import NiftiMasker
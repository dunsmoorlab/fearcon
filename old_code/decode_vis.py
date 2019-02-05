from fc_decoding import cv_iter
from fc_config import data_dir, sub_args

import matplotlib.pyplot as plt
import pandas as pd
import os


ni = pd.read_csv('%s%sgood_tr.csv'%(data_dir, os.sep + 'graphing' + os.sep + 'quals' + os.sep))
# ni_SC = pd.read_csv('%s%sscene_collapse.csv'%(data_dir, os.sep + 'graphing' + os.sep + 'localizer_decoding' + os.sep))
# ni_NR = pd.read_csv('%s%sno_rest.csv'%(data_dir, os.sep + 'graphing' + os.sep + 'localizer_decoding' + os.sep))
# ni_NR_SC = pd.read_csv('%s%sno_rest_scene_collapse.csv'%(data_dir, os.sep + 'graphing' + os.sep + 'localizer_decoding' + os.sep))

# b = pd.read_csv('%s%sbeta.csv'%(data_dir, os.sep + 'graphing' + os.sep + 'localizer_decoding' + os.sep))
# b_SC = pd.read_csv('%s%sDS_beta_scene_collapse.csv'%(data_dir, os.sep + 'graphing' + os.sep + 'localizer_decoding' + os.sep))
# b_NR = pd.read_csv('%s%sbeta_no_rest.csv'%(data_dir, os.sep + 'graphing' + os.sep + 'localizer_decoding' + os.sep))
# b_NR_SC = pd.read_csv('%s%sbeta_no_rest_scene_collapse.csv'%(data_dir, os.sep + 'graphing' + os.sep + 'localizer_decoding' + os.sep))
# sc = pd.read_csv('%s%sDS_scene_collapse.csv'%(data_dir, os.sep + 'graphing' + os.sep + 'localizer_decoding' + os.sep))

# new_b = pd.read_csv('%s%snew_beta_DS_scene_collapse.csv'%(data_dir, os.sep + 'graphing' + os.sep + 'localizer_decoding' + os.sep))


fig, ax = plt.subplots()

# ax.errorbar(cv_iter, ni.acc, yerr=ni.error, label='as_is')
# ax.errorbar(cv_iter, ni_NR.acc, yerr=ni_NR.error, label='no_rest')
# ax.errorbar(cv_iter, ni_SC.acc, yerr=ni_SC.error, label='SC')
# ax.errorbar(cv_iter, ni_NR_SC.acc, yerr=ni_NR_SC.error, label='no_rest_SC')

# ax.errorbar(cv_iter, b.acc, yerr=b.error, label='beta')
# ax.errorbar(cv_iter, b_SC.acc, yerr=b_SC.error, label='beta_SC')

# ax.errorbar(cv_iter, b.acc, yerr=b.error, label='beta 6_way')
ax.errorbar(cv_iter, ni.acc, yerr=ni.error, label='tr scene_collapse')
# ax.errorbar(cv_iter, b_SC.acc, yerr=b_SC.error, label='beta scene_collapse')
# ax.errorbar(cv_iter, new_b.acc, yerr=new_b.error, label='new beta scene_collapse')

# ax.errorbar(cv_iter, b_NR_SC.acc, yerr=b_NR_SC.error, label='beta no_rest_scene_collapse')
# ax.errorbar(cv_iter, b_NR.acc, yerr=b_NR.error, label='beta no_rest')



ax.set_xlabel('N voxels')
ax.set_ylabel('Mean Xval Acc (2-fold)')


legend = ax.legend()

frame = legend.get_frame()
frame.set_facecolor('0.90')

# Set the fontsize
for label in legend.get_texts():
    label.set_fontsize('large')

for label in legend.get_lines():
    label.set_linewidth(1.5)  # the legend line width

plt.savefig('%s/good_tr'%(data_dir + 'graphing' + os.sep + 'quals'))

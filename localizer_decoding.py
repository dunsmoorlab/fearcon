from fc_decoding import loc_decode, eval_xval
from fc_config import *
from cf_mat_plot import plot_confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns

# sns.set_style(rc={'axes.linewidth':'1.5'})
# plt.rcParams['xtick.labelsize'] = 18 
# plt.rcParams['ytick.labelsize'] = 18
# plt.rcParams['axes.labelsize'] = 20
# plt.rcParams['axes.titlesize'] = 22



#nvox = 

# res_ = eval_xval(name='good_tr', imgs='tr', del_rest=False, scene_collapse=True, rmv_scram=True, k=[nvox], cf=True)
res = eval_xval(name='none', imgs='beta', del_rest=False,
				scene_collapse=False, scene_DS=False, rmv_scram=False, rmv_indoor=False, binarize=False,
				p='all', save_dict=ppa_prepped, k=['all'], cf=True)
#res.res.to_csv('%sgraphing/localizer_decoding/%s.csv'%(data_dir,res.res.analysis[2]))
#Out[129]: ['animal', 'indoor', 'outdoor', 'rest', 'scrambled', 'tool']

#Out[129]: ['animal', 'tool', 'outdoor', 'indoor', 'scrambled', 'rest']

# mat = np.array((res.cf_mat[0], res.cf_mat[5], res.cf_mat[2], res.cf_mat[1], res.cf_mat[4], res.cf_mat[3]))
lab = ['animal','tool', 'scene','rest']
lab = ['animal','tool','indoor','outdoor','scrambled']
plot_confusion_matrix(res.cf_mat, lab, normalize=True, title='Classifier Confusion Matrix', save='%sgraphing/whyamiupsolate.png'%(data_dir))


# ni = eval_xval(name='good_tr', imgs='tr', del_rest=False, scene_collapse=True, rmv_scram=True, k=cv_iter).res
# ni.to_csv('%sgraphing/quals/%s.csv'%(data_dir,ni.analysis[2]))
# ni = eval_xval(name='Classifier Confusion Matrix', imgs='tr', del_rest=False, scene_collapse=True, rmv_scram=True, k=[nvox], cf=True)
# plot_confusion_matrix(ni.cf_mat, ni.cf_labels, normalize=True, title='Classifier Confusion Matrix', save='%sgraphing/quals/tr_cfmat.pdf'%(data_dir))
# plot_confusion_matrix(ni.cf_mat, ni.cf_labels, normalize=True, title='Classifier Confusion Matrix', save='%sgraphing/quals/tr_cfmat.png'%(data_dir), pdf=False)

# pni = eval_xval(name='ptsd', imgs='tr', del_rest=False, scene_collapse=True, rmv_scram=True, k=cv_iter, p=True).res
# pni.to_csv('%sgraphing/quals/%s.csv'%(data_dir,pni.analysis[2]))
# pni = eval_xval(name='Classifier Confusion Matrix', imgs='tr', del_rest=False, scene_collapse=True, rmv_scram=True, k=[nvox], cf=True, p=True)
# plot_confusion_matrix(pni.cf_mat, pni.cf_labels, normalize=True, title='Classifier Confusion Matrix', save='%sgraphing/quals/ptsd_tr_cfmat.pdf'%(data_dir))

# ni_SC = eval_xval(name='DS_scene_collapse', imgs='tr', del_rest=False, scene_collapse=True, k=cv_iter).res
# ni_NR = eval_xval(name='no_rest', imgs='tr', del_rest=True, scene_collapse=False, k=iter_vals).res
# ni_NR_SC = eval_xval(name='no_rest_scene_collapse', imgs='tr', del_rest=True, scene_collapse=True, k=iter_vals).res

#3/7
# b = eval_xval(name='beta', imgs='beta', del_rest=False, scene_collapse=False, k=cv_iter).res
# b.to_csv('%sgraphing/localizer_decoding/%s.csv'%(data_dir,b.analysis[2]))

# b_NR = eval_xval(name='beta', imgs='beta', del_rest=True, scene_collapse=False, k=cv_iter).res
# b_NR.to_csv('%sgraphing/localizer_decoding/beta_no_rest.csv'%(data_dir))


# b_SC = eval_xval(name='DS_beta_scene_collapse', imgs='beta', del_rest=False, scene_collapse=True, k=cv_iter).res
# b_SC.to_csv('%sgraphing/localizer_decoding/%s.csv'%(data_dir,b_SC.analysis[2]))

# b_NR_SC = eval_xval(name='beta_scene_collapse', imgs='beta', del_rest=True, scene_collapse=True, k=cv_iter).res
# b_NR_SC.to_csv('%sgraphing/localizer_decoding/beta_no_rest_scene_collapse.csv'%(data_dir))



# ni_SC.to_csv('%sgraphing/localizer_decoding/%s.csv'%(data_dir,ni_SC.analysis[2]))
# ni_NR.to_csv('%sgraphing/localizer_decoding/%s.csv'%(data_dir,ni_NR.analysis[2]))
# ni_NR_SC.to_csv('%sgraphing/localizer_decoding/%s.csv'%(data_dir,ni_NR_SC.analysis[2]))

#3.19
# new_b = eval_xval(name='new_beta_DS_scene_collapse', imgs='beta', del_rest=False, scene_collapse=True, k=cv_iter).res
# new_b.to_csv('%sgraphing/localizer_decoding/%s.csv'%(data_dir,new_b.analysis[2]))
# new_b_300 = eval_xval(name='new_beta_DS_scene_collapse', imgs='beta', del_rest=False, scene_collapse=True, k=[300], cf=True)
# plot_confusion_matrix(new_b_300.cf_mat, new_b_300.cf_labels, normalize=True, title='new_beta_DS_scene_collapse', save='%sgraphing/localizer_decoding/cf_mats/new_beta_DS_scene_collapse.pdf'%(data_dir))



# ni_SC = eval_xval(name='DS_scene_collapse', imgs='tr', del_rest=False, scene_collapse=True, k=[1000], cf=True)
# plot_confusion_matrix(ni_SC.cf_mat, ni_SC.cf_labels, normalize=True, title='scene_collapse', save='%sgraphing/localizer_decoding/cf_mats/SC.pdf'%(data_dir))

# ni_NR = eval_xval(name='no_rest', imgs='tr', del_rest=True, scene_collapse=False, k=iter_vals, cf=True)
# plot_confusion_matrix(ni_NR.cf_mat, ni_NR.cf_labels, normalize=True, title='no_rest', save='%sgraphing/localizer_decoding/cf_mats/NR.pdf'%(data_dir))



# b = eval_xval(name='beta', imgs='beta', del_rest=False, scene_collapse=False, k=iter_vals, cf=True)
# plot_confusion_matrix(b.cf_mat, b.cf_labels, normalize=True, title='beta', save='%sgraphing/localizer_decoding/cf_mats/beta.pdf'%(data_dir))

# b_SC = eval_xval(name='beta_scene_collapse', imgs='beta', del_rest=False, scene_collapse=True, k=[500], cf=True)
# plot_confusion_matrix(b_SC.cf_mat, b_SC.cf_labels, normalize=True, title='beta_SC', save='%sgraphing/localizer_decoding/cf_mats/beta_SC.pdf'%(data_dir))



#NOT this one lol
#ni_NR_SC = eval_xval(name='no_rest_scene_collapse', imgs='tr', del_rest=True, scene_collapse=True, k=[300], cf=True)
#plot_confusion_matrix(ni_NR_SC.cf_mat, ni_NR_SC.cf_labels, normalize=True, title='no_rest_scene_collapse; k=300; mean_acc = %s'%(ni_NR_SC.res.acc[300]), save='%s%sNR_SC_300.pdf'%(data_dir, os.sep + 'graphing' + os.sep + 'localizer_decoding' + os.sep))

#sc = eval_xval(name='scene_collapse', imgs='tr', del_rest=False, scene_collapse=True, k=cv_iter)
#sc.res.to_csv('%s%sscene_collapse.csv'%(data_dir, os.sep + 'graphing' + os.sep + 'localizer_decoding' + os.sep))

#sc = eval_xval(name='scene_collapse', imgs='tr', del_rest=False, scene_collapse=True, k=[1000], cf=True)
#plot_confusion_matrix(sc.cf_mat, sc.cf_labels, normalize=True, title='scene_collapse', save='%sgraphing/localizer_decoding/sc_1000_15sub.pdf'%(data_dir))

#sc_500 = eval_xval(name='scene_collapse_500', imgs='tr', del_rest=False, scene_collapse=True, k=[500], cf=True)
#plot_confusion_matrix(sc_500.cf_mat, sc_500.cf_labels, normalize=True, title='scene_collapse; k=500; mean_acc = %s'%(sc_500.res.acc[500]), save='%s%ssc_500.pdf'%(data_dir, os.sep + 'graphing' + os.sep + 'localizer_decoding' + os.sep))

#sc_1000 = eval_xval(name='scene_collapse_1000', imgs='tr', del_rest=False, scene_collapse=True, k=[1000], cf=True)
#plot_confusion_matrix(sc_1000.cf_mat, sc_1000.cf_labels, normalize=True, title='scene_collapse; k=1000; mean_acc = %s'%(sc_1000.res.acc[1000]), save='%s%ssc_1000.pdf'%(data_dir, os.sep + 'graphing' + os.sep + 'localizer_decoding' + os.sep))


# b_1000 = eval_xval(name='b_1000', imgs='beta', del_rest=False, scene_collapse=False, k=[1000], cf=True)
# plot_confusion_matrix(b_1000.cf_mat, b_1000.cf_labels, normalize=True, title='beta_5way; k=1000; mean_acc = %s'%(b_1000.res.acc[1000]), save='%s%sb_1000.pdf'%(data_dir, os.sep + 'graphing' + os.sep + 'localizer_decoding' + os.sep))

# b_SC_300 = eval_xval(name='b_SC_300', imgs='beta', del_rest=False, scene_collapse=True, k=[300], cf=True)
# plot_confusion_matrix(b_SC_300.cf_mat, b_SC_300.cf_labels, normalize=True, title='beta_SC; k=300; mean_acc = %s'%(b_SC_300.res.acc[300]), save='%s%sb_SC_300.pdf'%(data_dir, os.sep + 'graphing' + os.sep + 'localizer_decoding' + os.sep))

# b_SC_1500 = eval_xval(name='b_SC_1500', imgs='beta', del_rest=False, scene_collapse=True, k=[1500], cf=True)
# plot_confusion_matrix(b_SC_1500.cf_mat, b_SC_1500.cf_labels, normalize=True, title='beta_SC; k=1500; mean_acc = %s'%(b_SC_1500.res.acc[1500]), save='%s%sb_SC_1500.pdf'%(data_dir, os.sep + 'graphing' + os.sep + 'localizer_decoding' + os.sep))


#ppa = eval_xval(name='pps_fs_mask', imgs='tr_PPA_fs', del_rest=True, scene_collapse=True, k=['all'], cf=True)
#ppa.res.to_csv('%sgraphing/localizer_decoding/%s.csv'%(data_dir,ppa.res.analysis['all']))

#plot_confusion_matrix(ppa.cf_mat, ppa.cf_labels, normalize=True, title='PPA_fs; mean_k = 345; mean_acc = %s'%(ppa.res.acc['all']), save='%s%s/PPA_fs.pdf'%(data_dir, os.sep + 'graphing' + os.sep + 'localizer_decoding' + os.sep))

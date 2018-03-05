from fc_decoding import loc_decode, eval_xval, cv_iter
from fc_config import data_dir, sub_args
from cf_mat_plot import plot_confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import os


# iter_vals = ['all']


# ni = eval_xval(name='as_is', imgs='tr', del_rest=False, scene_collapse=False, k=iter_vals).res
# ni_SC = eval_xval(name='scene_collapse', imgs='tr', del_rest=False, scene_collapse=True, k=iter_vals).res
# ni_NR = eval_xval(name='no_rest', imgs='tr', del_rest=True, scene_collapse=False, k=iter_vals).res
# ni_NR_SC = eval_xval(name='no_rest_scene_collapse', imgs='tr', del_rest=True, scene_collapse=True, k=iter_vals).res
# b = eval_xval(name='beta', imgs='beta', del_rest=False, scene_collapse=False, k=iter_vals).res
# b_SC = eval_xval(name='beta_scene_collapse', imgs='beta', del_rest=False, scene_collapse=True, k=iter_vals).res

# ni.to_csv('%sgraphing/localizer_decoding/%s.csv'%(data_dir,ni.analysis[2]))
# ni_SC.to_csv('%sgraphing/localizer_decoding/%s.csv'%(data_dir,ni_SC.analysis[2]))
# ni_NR.to_csv('%sgraphing/localizer_decoding/%s.csv'%(data_dir,ni_NR.analysis[2]))
# ni_NR_SC.to_csv('%sgraphing/localizer_decoding/%s.csv'%(data_dir,ni_NR_SC.analysis[2]))
# b.to_csv('%sgraphing/localizer_decoding/%s.csv'%(data_dir,b.analysis[2]))
# b_SC.to_csv('%sgraphing/localizer_decoding/%s.csv'%(data_dir,b_SC.analysis[2]))


# ni = eval_xval(name='as_is', imgs='tr', del_rest=False, scene_collapse=False, k=iter_vals, cf=True)
# plot_confusion_matrix(ni.cf_mat, ni.cf_labels, normalize=True, title='as_is', save='%sgraphing/localizer_decoding/cf_mats/as_is.pdf'%(data_dir))

# ni_SC = eval_xval(name='scene_collapse', imgs='tr', del_rest=False, scene_collapse=True, k=iter_vals, cf=True)
# plot_confusion_matrix(ni_SC.cf_mat, ni_SC.cf_labels, normalize=True, title='scene_collapse', save='%sgraphing/localizer_decoding/cf_mats/SC.pdf'%(data_dir))

# ni_NR = eval_xval(name='no_rest', imgs='tr', del_rest=True, scene_collapse=False, k=iter_vals, cf=True)
# plot_confusion_matrix(ni_NR.cf_mat, ni_NR.cf_labels, normalize=True, title='no_rest', save='%sgraphing/localizer_decoding/cf_mats/NR.pdf'%(data_dir))



# b = eval_xval(name='beta', imgs='beta', del_rest=False, scene_collapse=False, k=iter_vals, cf=True)
# plot_confusion_matrix(b.cf_mat, b.cf_labels, normalize=True, title='beta', save='%sgraphing/localizer_decoding/cf_mats/beta.pdf'%(data_dir))

# b_SC = eval_xval(name='beta_scene_collapse', imgs='beta', del_rest=False, scene_collapse=True, k=iter_vals, cf=True)
# plot_confusion_matrix(b_SC.cf_mat, b_SC.cf_labels, normalize=True, title='beta_SC', save='%sgraphing/localizer_decoding/cf_mats/beta_SC.pdf'%(data_dir))



#NOT this one lol
#ni_NR_SC = eval_xval(name='no_rest_scene_collapse', imgs='tr', del_rest=True, scene_collapse=True, k=[300], cf=True)
#plot_confusion_matrix(ni_NR_SC.cf_mat, ni_NR_SC.cf_labels, normalize=True, title='no_rest_scene_collapse; k=300; mean_acc = %s'%(ni_NR_SC.res.acc[300]), save='%s%sNR_SC_300.pdf'%(data_dir, os.sep + 'graphing' + os.sep + 'localizer_decoding' + os.sep))

#sc = eval_xval(name='scene_collapse', imgs='tr', del_rest=False, scene_collapse=True, k=cv_iter)
#sc.res.to_csv('%s%sscene_collapse.csv'%(data_dir, os.sep + 'graphing' + os.sep + 'localizer_decoding' + os.sep))
sc = eval_xval(name='scene_collapse', imgs='tr', del_rest=False, scene_collapse=True, k=[1000], cf=True)
plot_confusion_matrix(sc.cf_mat, sc.cf_labels, normalize=True, title='scene_collapse', save='%sgraphing/localizer_decoding/sc_1000_15sub.pdf'%(data_dir))

#sc_500 = eval_xval(name='scene_collapse_500', imgs='tr', del_rest=False, scene_collapse=True, k=[500], cf=True)
#plot_confusion_matrix(sc_500.cf_mat, sc_500.cf_labels, normalize=True, title='scene_collapse; k=500; mean_acc = %s'%(sc_500.res.acc[500]), save='%s%ssc_500.pdf'%(data_dir, os.sep + 'graphing' + os.sep + 'localizer_decoding' + os.sep))

#sc_1000 = eval_xval(name='scene_collapse_1000', imgs='tr', del_rest=False, scene_collapse=True, k=[1000], cf=True)
#plot_confusion_matrix(sc_1000.cf_mat, sc_1000.cf_labels, normalize=True, title='scene_collapse; k=1000; mean_acc = %s'%(sc_1000.res.acc[1000]), save='%s%ssc_1000.pdf'%(data_dir, os.sep + 'graphing' + os.sep + 'localizer_decoding' + os.sep))


#ppa = eval_xval(name='pps_fs_mask', imgs='tr_PPA_fs', del_rest=True, scene_collapse=True, k=['all'], cf=True)
#ppa.res.to_csv('%sgraphing/localizer_decoding/%s.csv'%(data_dir,ppa.res.analysis['all']))

#plot_confusion_matrix(ppa.cf_mat, ppa.cf_labels, normalize=True, title='PPA_fs; mean_k = 345; mean_acc = %s'%(ppa.res.acc['all']), save='%s%s/PPA_fs.pdf'%(data_dir, os.sep + 'graphing' + os.sep + 'localizer_decoding' + os.sep))

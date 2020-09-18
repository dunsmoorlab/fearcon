from fc_config import *

##signal change in dACC and insula
fc_betas = pd.read_csv(os.path.join(data_dir,'graphing','signal_change','run002_beta_values.csv'))
dacc = fc_betas[['sub','CSp_CSm']][fc_betas.roi == 'dACC_beta'].reset_index(drop=True).rename(columns={'sub':'subject'})
dacc['group'] = dacc.subject.apply(lgroup)
dacc.to_csv('/Users/ach3377/Desktop/dacc_CSp_CSm_contrast.csv',index=False)

##mvpa decoding ROC AUC
from fc_decoding import loc_decode, eval_xval
res = eval_xval(name='none', imgs='beta', del_rest=False,
                scene_collapse=True, scene_DS=False, rmv_scram=False, rmv_indoor=False, binarize=True,
                p='all', save_dict=beta_ppa_prepped, k=['all'], cf=True)
scores = res.auc.drop(columns=['label'])
scores['group'] = scores.subject.apply(lgroup)
scores.to_csv('~/Desktop/mvpa_decoding.csv',index=False)

##between group corr with vmPFC activity
mask = nib.load(os.path.join(data_dir,'group_roi_masks','vmPFC_mask.nii.gz'))
coor = np.where(mask.get_data() == 1)

# cimg = nib.load(os.path.join('/Volumes','DunsmoorRed','group_ev_corr','control_ev_corr.gfeat','cope3.feat','filtered_func_data.nii.gz'))
# pimg = nib.load(os.path.join('/Volumes','DunsmoorRed','group_ev_corr','ptsd_ev_corr.gfeat','cope3.feat','filtered_func_data.nii.gz'))
cimg = nib.load('/Users/ACH/Desktop/control_data.nii.gz')
pimg = nib.load('/Users/ACH/Desktop/ptsd_data.nii.gz')

cval = cimg.get_data()[coor]
pval = pimg.get_data()[coor]

# creg = pd.read_csv(os.path.join('/Volumes','DunsmoorRed','group_ev_corr','control_ev_values.csv'))
# preg = pd.read_csv(os.path.join('/Volumes','DunsmoorRed','group_ev_corr','ptsd_ev_values.csv'))

creg = pd.read_csv('/Users/ACH/Desktop/control_ev_values.csv')
preg = pd.read_csv('/Users/ACH/Desktop/ptsd_ev_values.csv')



for i in range(24):
    cval[:,i] *= creg.reg[i]
    pval[:,i] *= preg.reg[i]

cout = cval.mean(axis=0)
pout = pval.mean(axis=0)

corr = pd.DataFrame({'val':np.concatenate([cout,pout])}).reset_index(drop=True)
corr['subject'] = all_sub_args
corr['group'] = corr.subject.apply(lgroup)

corr.to_csv('~/Desktop/uni_mvpa_corr.csv',index=False)

#do it the way joey wants
cmask = cval.mean(axis=1)
import scipy
cmask = scipy.stats.zscore(cmask)
cmask_coor = np.where(cmask>=3.1)

cval = cval[cmask_coor]
pval = pval[cmask_coor]

cout = cval.mean(axis=0)
pout = pval.mean(axis=0)

corr = pd.DataFrame({'val':np.concatenate([cout,pout])}).reset_index(drop=True)
corr['subject'] = all_sub_args
corr['group'] = corr.subject.apply(lgroup)

pg.ttest(corr.val[corr.group=='control'],corr.val[corr.group=='ptsd'])

corr.to_csv('~/Desktop/uni_mvpa_corr.csv',index=False)

#pcl mediator
#just go get prmod from nbc.py, its easier

pcl = pd.read_csv(data_dir+'Demographics_Survey/pcl_part3.csv')
prmod['pcl'] = pcl.groupby('subject').sum()['score'].values
prmod = prmod.drop(index=np.where(prmod.pcl==0)[0])
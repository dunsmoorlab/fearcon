import os
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from scipy.special import expit

from fc_config import *

subjects = {'control':sub_args,
            'ptsd'   :p_sub_args}

ctr = pd.read_csv(os.path.join(data_dir,'graphing','nbc19','control_exp_ev_df.csv'))
ptr = pd.read_csv(os.path.join(data_dir,'graphing','nbc19','ptsd_exp_ev_df.csv'))
df = pd.concat((ctr,ptr))
df = df[df.condition == 'scene'].set_index('tr').sort_index()
df = df.loc[-2:0].reset_index() #.loc does inclusive indexing!
_response = df.groupby(['subject','trial']).first()['response']
df = df.groupby(['subject','trial']).mean().drop(columns='tr')
df['response'] = (_response == 'no').astype(int)
df = df.reset_index()
df['group'] = df.subject.apply(lgroup)
df['sub_xval'] = df.subject
df = df.set_index(['group','subject']).sort_index()
R = np.random.RandomState(42)
# def within_subject(df):
#     logreg = LogisticRegression(solver='lbfgs')
#     res = {}
#     for group in subjects:
#         res[group] = {}
#         for sub in subjects[group]:
#             X = df.loc[(group,sub),'evidence'].values.reshape(-1,1)
#             y = df.loc[(group,sub),'response'].values
#             if len(np.unique(y)) == 1: print(sub, y)#pass #skip subs with only 1 type of response
#             else:
#                 logreg.fit(X,y)
#                 res[group][sub] = logreg.coef_[0][0]
#     res = pd.DataFrame.from_dict(res).reset_index(
#                      ).rename(columns={'index':'subject'})
#     res = res.melt(id_vars='subject',var_name='group',value_name='beta'
#             ).dropna().reset_index(drop=True)
#     return res
# ws = within_subject(df)
# fig, ax = plt.subplots()
# sns.pointplot(data=ws,x='group',y='beta',palette=gpal)
# sns.swarmplot(data=ws,x='group',y='beta',color='black')
# xl = ax.get_xlim(); ax.hlines(0,xl[0],xl[1],color='black')
# ax.set_title('Within subject 4CS+ trials logreg beta coef.')
# #OK cool! so we show that within subjects we can predict response,
# #but what about the subjects who only have 1 response? lets look at them:
# pred_subs = {group:[sub for sub in subjects[group] if sub not in ws.subject.unique()] for group in subjects}
# #before we try to .predict(), lets make sure that if we .fit() on all the
# # "good" subjects, we still have predictive power. Should probably Xval first
# fit_subs = {group:[sub for sub in subjects[group] if sub in ws.subject.unique()] for group in subjects}

# idx = pd.IndexSlice
# logo = LeaveOneGroupOut()

# #OK so i don't think this is going to be the best way to show this...
# bgr = {}
# for group in subjects:
#     logreg = LogisticRegression(solver='lbfgs')   
#     X = df.loc[idx[group,fit_subs[group]],'evidence'].values.reshape(-1,1)
#     y = df.loc[idx[group,fit_subs[group]],'response'].values
#     xval_groups = df.loc[idx[group,fit_subs[group]],'sub_xval'].values
#     logreg.fit(X,y)

#     bgr[group] = cross_val_score(logreg,X,y=y,groups=xval_groups,scoring='roc_auc',cv=logo)
#     X_pred = df.loc[idx[group,pred_subs[group]],'evidence'].values.reshape(-1,1)
#     y_true = df.loc[idx[group,pred_subs[group]],'response'].values
#     y_guess = logreg.fit(X,y).predict(X_pred)
#     print(np.mean(y_true == y_guess))

# n_c = bgr['control'].shape[0];n_p = bgr['ptsd'].shape[0]
# bgr = np.concatenate((bgr['control'],bgr['ptsd']))
# bgr = pd.DataFrame({'auc':bgr,'group':np.concatenate((np.repeat(['control'],n_c),np.repeat(['ptsd'],n_p)))})
# fig, ax = plt.subplots()
# sns.pointplot(data=bgr,x='group',y='auc',ax=ax,palette=gpal)
# sns.swarmplot(data=bgr,x='group',y='auc',color='black')
# xl = ax.get_xlim(); ax.hlines(.5,xl[0],xl[1],color='black')
# ax.set_title('ROC_AUC, subs with 2 types of response,\n leave one subject cross val')



# #Lets see what it looks like if we train on all subjects in the xval framework?
# bgx = {}
# for group in subjects:
#     logreg = LogisticRegression(solver='lbfgs')
#     X = df.loc[group,'evidence'].values.reshape(-1,1)
#     y = df.loc[group,'response'].values
#     xval_groups = df.loc[group,'sub_xval'].values

#     bgx[group] = cross_val_score(logreg,X,y=y,groups=xval_groups,scoring='accuracy',cv=logo)

# bgx = pd.DataFrame.from_dict(bgx).melt(var_name='group',value_name='acc')
# fig, ax = plt.subplots()
# sns.pointplot(data=bgx,x='group',y='acc',ax=ax,palette=gpal)
# sns.swarmplot(data=bgx,x='group',y='acc',color='black')
# xl = ax.get_xlim(); ax.hlines(.5,xl[0],xl[1],color='black')
# ax.set_title('Accuracy, all subs, leave one subject cross val')

# ev = pd.read_csv(os.path.join(data_dir,'graphing','signal_change','mvpa_ev.csv'))
# tev = df.reset_index().groupby(['subject']).mean().reset_index().drop(columns=['trial','sub_xval'])
# tev['group'] = tev.subject.apply(lgroup)
# tev = tev.set_index('group')
# def linboot(group,n_boot=1000):
#     linreg = LinearRegression()
#     dat = tev.loc[group].reset_index()
#     boot_res = np.zeros(n_boot)
#     for i in range(n_boot):
#         _samp = R.randint(0,dat.shape[0],dat.shape[0])
#         X = dat.loc[_samp,'evidence'].values.reshape(-1,1)
#         y = dat.loc[_samp,'response'].values
#         linreg.fit(X,y)
#         boot_res[i] = linreg.coef_[0]
#     pval = 1 - np.mean(boot_res > 0)
#     print(group,pval.round(3))
#     return boot_res
# c = linboot('control')
# p = linboot('ptsd')
# g = pd.DataFrame({'beta':np.concatenate((c,p)),
#                   'group':np.repeat(['control','ptsd'],1000)})
# sns.violinplot(data=g,x='group',y='beta',palette=gpal)



#and finally, correct super-soldier
def trialbeta(group,n_boot=1000):
    logreg = LogisticRegression(solver='lbfgs')

    dat = df.loc[group]
    subjects = {'control':sub_args,
                'ptsd':   p_sub_args}
    boot_res = np.zeros(n_boot)
    curve    = np.zeros((n_boot,100))

    for i in range(n_boot):
        y = np.zeros(96)
        while len(np.unique(y)) == 1:
            _samp = R.choice(subjects[group],len(subjects[group]))
            X = dat.loc[_samp,'evidence'].values.reshape(-1,1)
            y = dat.loc[_samp,'response'].values
        logreg.fit(X,y)
        boot_res[i] = logreg.coef_[0][0]
        curve[i,:] = expit(np.linspace(0,1,100) * logreg.coef_ + logreg.intercept_)[0]

    pval = 1 - np.mean(boot_res > 0)
    print('%s super-subject pval %s'%(group,pval.round(3)))
    # res = pd.DataFrame().from_dict(res,orient='index').rename(columns={0:'beta'})
    return {'beta':boot_res,'fit':curve}
cbeta = trialbeta('control')
pbeta = trialbeta('ptsd')
ss = pd.DataFrame({'beta':np.concatenate((cbeta['beta'],pbeta['beta'])),
                  'group':np.repeat(['control','ptsd'],1000)})
ss_stats = ss.groupby('group').mean()
ss_stats['ci'] = ss.groupby('group').apply(lambda x: np.percentile(x,[5,100]))
ss['x'] = 'group'

fig, ax = plt.subplots()
sns.violinplot(data=ss,y='beta',x='x',hue='group',scale='area',split=True,
               palette=gpal,ax=ax,inner=None)
xl = ax.get_xlim();ax.hlines(0,xl[0],xl[1],color='black',linestyle='--')
ax.legend_.remove()
offset = [-.02,.02]
for i, group in enumerate(subjects):
    xval = ax.get_xticks() + offset[i]
    y2 = ss_stats.loc[group,'beta']; ax.scatter(xval,y2,color='black') #draw line for mean effect
    ef = ss_stats.loc[group,'ci']
    ax.vlines(xval,ef[0],ef[1],color='black',linewidth=3).set_capstyle('round') #underline the 95% CI of effect in red
    # ax.vlines(0,ef[0],ef[1],color=desat_r,linewidth=4).set_capstyle('round') #underline the 95% CI of effect
diff = cbeta['beta'] - pbeta['beta']
dp = 1 - np.mean(diff > 0)
# ax.set_title('Super subject results, first 4 CS+ trials')

# X_test = np.linspace(0,1,100)

# ctest = pd.DataFrame(cbeta['fit'].T).melt(var_name='iter',value_name='y_pred')
# ctest['X_test'] = np.tile(X_test,1000)
# ptest = pd.DataFrame(pbeta['fit'].T).melt(var_name='iter',value_name='y_pred')
# ptest['X_test'] = np.tile(X_test,1000)


# fig, ax = plt.subplots()
# sns.lineplot(data=ptest,x='X_test',y='y_pred',units='iter',estimator=None,alpha=.1,ax=ax,color=gpal[1])
# sns.lineplot(data=ctest,x='X_test',y='y_pred',units='iter',estimator=None,alpha=.1,ax=ax,color=gpal[0])
# sns.lineplot(data=ctest,x='X_test',y='y_pred',ci=None,color='white',lw=5)
# sns.lineplot(data=ptest,x='X_test',y='y_pred',ci=None,color='white',lw=5)
# ax.set_xlim(0,1);ax.set_ylim(0,1)
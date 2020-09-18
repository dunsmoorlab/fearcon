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

#run the boostrap
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
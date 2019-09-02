import os
import numpy as np
import pandas as pd
import pingouin as pg
from fc_config import data_dir
from itertools import combinations
from scipy.stats import f

raw = pd.read_csv(os.path.join(data_dir,'group_ER','mOFC','mOFC_group_stats.csv'))
# balanced = raw.drop(index=np.where(raw.response == 'MO')[0])
balanced = raw.groupby(['subject','encode','trial_type','hc_acc']).mean().reset_index()
balanced.set_index(['encode','trial_type','hc_acc','subject'],inplace=True)
balanced.loc[('fear_conditioning','CS+','M',115),'rsa'] = balanced.loc[('fear_conditioning','CS+','M'),'rsa'].mean()
balanced.reset_index(inplace=True)
cell_count = {}
for index in range(balanced.shape[0]): 
    key = [] 
    for col in ['encode','trial_type','hc_acc']: 
         key.append(balanced[col].iloc[index]) 
    key = tuple(key) 
    if key in cell_count: 
         cell_count[key] = cell_count[key] + 1 
    else: 
         cell_count[key] = 1 

between = ['group']
# within = ['encode','trial_type','hc_acc']
within = ['encode','trial_type']
subject = 'subject'
dv = 'rsa'

n_factors = len(within)
#lets call factors 1,2,3 etc
keys = {}
keys['subject'] = subject
for i, factor in enumerate(within):
	keys[factor] = i+1

# a, b, c = within
subs_within = within.copy()
#need this list for pandas indexing
subs_within.insert(0,subject)
data = raw.groupby(subs_within).mean().reset_index()


#going to try to write this with N within factors, dictionary may be best approach
_N = {}
for i, factor in enumerate(subs_within):	
	_N[factor] = data[factor].nunique()

_single_means = {}
for i, factor in enumerate(subs_within):
	_single_means[factor] = data.groupby(factor)[dv].mean()

#need the means of each combination of factors
_combined_means = {}
for combo in combinations(subs_within,2):
	combo_key = str(combo[0]) + '__' + str(combo[1])
	_combined_means[combo_key] = data.groupby([combo[0],combo[1]])[dv].mean()

#grand mean
mu = data[dv].mean()

### Sums of squares
SS = {}
SS['total'] = np.sum((data[dv] - mu)**2)
#single factor
for factor in subs_within:
	others = [f for f in subs_within if f != factor]
	
	_nobs = 1
	for f in others:
		_nobs *= _N[f]

	SS[factor] = _nobs * np.sum((_single_means[factor] - mu)**2)

#combined
for combo in combinations(subs_within,2):
	combo_key = str(combo[0]) + '__' + str(combo[1])

	others = [f for f in subs_within if f not in combo]

	_nobs = 1
	for f in others:
		_nobs *= _N[f]

	combo_SS_err =  _nobs * np.sum((_combined_means[combo_key] - mu)**2)
	SS[combo_key] = combo_SS_err - SS[combo[0]] - SS[combo[1]]

#all factors
SS['all'] = SS['total']
for factor in subs_within:
	SS['all'] -= SS[factor]
for combo in combinations(subs_within,2):
	combo_key = str(combo[0]) + '__' + str(combo[1])
	SS['all'] -= SS[combo_key]


### Degrees of Freedom
DOF = {}
for factor in subs_within:
	DOF[factor] = _N[factor] - 1

for combo in combinations(subs_within,2):
	combo_key = str(combo[0]) + '__' + str(combo[1])

	combo_DOF_err = _N[combo[0]] * _N[combo[1]] - 1
	DOF[combo_key] = combo_DOF_err - DOF[combo[0]] - DOF[combo[1]]

#total
DOF['total'] = np.product([n for n in _N.values()]) - 1

#all factors
DOF['all'] = DOF['total']
for factor in subs_within:
	DOF['all'] -= DOF[factor]
for combo in combinations(subs_within,2):
	combo_key = str(combo[0]) + '__' + str(combo[1])
	DOF['all'] -= DOF[combo_key]

### Mean squares
MS = {}
for factor in within:
	MS[factor] = SS[factor] / DOF[factor]

for combo in combinations(subs_within,2):
	combo_key = str(combo[0]) + '__' + str(combo[1])
	MS[combo_key] = SS[combo_key] / DOF[combo_key]

MS['all'] = SS['all'] / DOF['all']


### F-values
F = {}
for factor in within:
	F[factor] = MS[factor] / MS[subject + '__' + str(factor)]
for combo in combinations(within,2):
	combo_key = str(combo[0]) + '__' + str(combo[1])
	F[combo_key] 

	#oops need to go back and do each pair in regards to each subject not just the total...


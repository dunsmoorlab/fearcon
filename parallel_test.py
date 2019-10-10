import numpy as np
import pandas as pd
import pingouin as pg
from joblib import Parallel, delayed

from fc_config import *
from glm_timing import glm_timing

group = 'control'

subjects = {'control':sub_args,
            'ptsd':p_sub_args,
            'all':all_sub_args}

nvox    = 77779

df = pd.read_csv(os.path.join(data_dir,'group_ER','%s_template_df.csv'%(group)))

data = np.load(os.path.join(data_dir,'group_ER','%s_std_item_ER.npy'%(group)))

    # tests = ['encode','trial_type','intrx']
pres = np.zeros((nvox,3))
# fres = np.zeros((nvox,3))

pres_str = os.path.join(data_dir,'group_ER','pingouin_stats','parallel_%s_rm_p.npy'%(group))
# fres_str = os.path.join(data_dir,'group_ER','pingouin_stats','parallel_%s_rm_f.npy'%(group))

def procInput(i):
        vdf = df.copy()
        vdf.rsa = data[:,:,i].flatten()

        anova = pg.rm_anova(data=vdf,dv='rsa',within=['encode','trial_type'],subject='subject')#repeated measures anova
        # if i%1000 == 0:print(i)#some marker of progress
        return anova['p-unc'].values
        # pres[i,:] = anova['p-unc'].values
        # fres[i,:] = anova['F'].values


from time import time
for n in [1,2,4,8,16,32]:
    start = time()
    print(n,'start')        
    Parallel(n_jobs=n)(delayed(procInput)(i) for i in range(5000))
    end = time() - start
    print(n,end)
np.save(pres_str,pres)
np.save(fres_str,fres)


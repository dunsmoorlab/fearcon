import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pingouin as pg
from fc_config import *
from wesanderson import wes_palettes
from nilearn.input_data import NiftiMasker
from glm_timing import glm_timing
from mvpa_analysis import group_decode
from signal_change import collect_ev

#day 1
day1_scr = pd.read_csv(os.path.join(data_dir,'graphing','SCR','day1_all_scr.csv'))
#remove baseline
day1_scr = day1_scr[day1_scr.phase != 'baseline']
#don't need CS+US trials
day1_scr = day1_scr[day1_scr.condition != 'CS+US']
day1_scr = day1_scr.drop(columns=['trial','raw'])
day1_scr.reset_index(inplace=True, drop=True)
#collect means for each phase half
day1_scr = day1_scr.groupby(by=['phase','subject','half','condition']).mean()
day1_scr.reset_index(inplace=True)
day1_scr['group'] = ''
day1_scr.group = np.repeat(['control','ptsd','control','ptsd'],int(24*4))
day1_scr['quarter'] = 0
day1_scr.set_index(['phase','half'],inplace=True)
day1_scr.loc[('fear_conditioning',1),'quarter'] = 1
day1_scr.loc[('fear_conditioning',2),'quarter'] = 2
day1_scr.loc[('extinction',1),'quarter'] = 3
day1_scr.loc[('extinction',2),'quarter'] = 4
day1_scr.reset_index(inplace=True)
day1_scr['gc'] = day1_scr.group + '_' + day1_scr.condition

day2_scr = pd.read_csv(os.path.join(data_dir,'graphing','SCR','day2_all_scr.csv'))
day2_scr = day2_scr.drop(columns=['trial','raw'])
day2_scr = day2_scr.groupby(by=['phase','subject','half','condition']).mean()
day2_scr['group'] = ''
day2_scr.loc[('renewal',sub_args),'group'] = 'control'
day2_scr.loc[('renewal',p_sub_args),'group'] = 'ptsd'
day2_scr.reset_index(inplace=True)
day2_scr['gc'] = day2_scr.group + '_' + day2_scr.condition

SCR = pd.concat([day1_scr,day2_scr],sort=True)
SCR = SCR.sort_values(by=['quarter','group','subject','condition'])

SCR.to_csv(os.path.join(data_dir,'graphing','SCR','fc_scr.csv'),index=False)


SCR = SCR.set_index(['quarter','condition'],drop=True)
####
comp = pd.DataFrame(index=pd.MultiIndex.from_product(
                    [[1,2,3,4,5,6],all_sub_args],names=['quarter','subject']))
comp['scr'] = 0
for quarter in [1,2,3,4,5,6]: comp.loc[quarter,'scr'] = SCR.loc[(quarter,'CS+'),'scr'].values - SCR.loc[(quarter,'CS-'),'scr'].values
comp.reset_index(inplace=True)
comp['group'] = np.tile(np.repeat(['control','ptsd'],24),6)

comp.to_csv(os.path.join(data_dir,'graphing','SCR','fc_scr_comp.csv'),index=False)
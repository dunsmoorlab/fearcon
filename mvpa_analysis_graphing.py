from mvpa_analysis import group_decode, vis_cond_phase, get_bar_stats, phase_bar_plot, vis_event_res
from fc_config import data_dir, PPA_fs_prepped
from scipy.stats import sem
import numpy as np

res = group_decode(k=1000)


#phase_bar_plot(res, title='lololol')


#res_stats = get_bar_stats(VTC_LOC)
#res_stats.to_csv('%s/graphing/mvpa_analysis/stats_vtc_loc.csv'%(data_dir))




for phase in res.event_res.keys():
	vis_event_res(res.event_res, phase, title='WRONG')


# scenes = {}
# for phase in res.event_res.keys():
# 	scenes[phase] = {}
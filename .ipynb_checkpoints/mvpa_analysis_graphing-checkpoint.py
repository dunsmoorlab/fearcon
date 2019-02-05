from mvpa_analysis import group_decode, decode#, vis_cond_phase, get_bar_stats, phase_bar_plot, vis_event_res
from fc_config import data_dir
from scipy.stats import sem
import numpy as np

# beta_res = group_decode(imgs='beta', k=500)
# tr = group_decode(imgs='tr', k=1000)
# new_beta_res = group_decode(imgs='beta', k=1000)
# {new_beta_res.vis_cond_phase(phase=phase, title='new_beta_sc_ds') for phase in decode.decode_runs}
# {new_beta_res.vis_event_res(phase, title='Train on NEW Betas, scene_ds, k=300') for phase in decode.decode_runs}
# new_beta_res.exp_event()


nvox = 500

res = group_decode(imgs='tr', k=nvox, SC=True, S_DS=True, rmv_scram=True)
# {res.vis_cond_phase(phase=phase, title='new_beta_sc_ds') for phase in decode.decode_runs}
# {res.vis_event_res(phase, title='Train on NEW Betas, scene_ds, k=1000') for phase in decode.decode_runs}
res.exp_event()
res.vis_exp_event()

# for trial in [1,2]:
# 	res.exp_stats(trial_=trial)
	# print(res.pair_df)
	# print(res.ind_df)

# pres = group_decode(imgs='tr', k=nvox, SC=True, S_DS=True, rmv_scram=True, p=True)
# # {res.vis_cond_phase(phase=phase, title='new_beta_sc_ds') for phase in decode.decode_runs}
# # {res.vis_event_res(phase, title='Train on NEW Betas, scene_ds, k=1000') for phase in decode.decode_runs}
# pres.exp_event()
# pres.vis_exp_event()

# for trial in [1,2]:
# 	pres.exp_stats(trial_=trial)
# 	print(pres.pair_df)
# 	print(pres.ind_df)
'''
2       no   0  2.076388  0.064588
3       no   1  3.663724  0.004362
0   1.390299  0.182374
1   3.653655  0.001966
'''
# pres = group_decode(imgs='tr', k=nvox, SC=True, S_DS=True, rmv_scram=True, p=True)
# pres.exp_event()
# pres.vis_exp_event()



#phase_bar_plot(res, title='lololol')


# {beta_res.vis_cond_phase(phase=phase, title='beta_sc_ds') for phase in decode.decode_runs}
# {tr.vis_cond_phase(phase=phase, title='tr_sc_ds') for phase in decode.decode_runs}

# #res_stats = get_bar_stats(VTC_LOC)
# #res_stats.to_csv('%s/graphing/mvpa_analysis/stats_vtc_loc.csv'%(data_dir))




# {beta_res.vis_event_res(phase, title='Train on Betas, scene_ds, k=500') for phase in decode.decode_runs}
# {tr.vis_event_res(phase, title='TR, scene_ds, k=1000') for phase in decode.decode_runs}

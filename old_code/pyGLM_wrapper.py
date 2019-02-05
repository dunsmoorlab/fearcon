#run the level 1s
from fc_config import data_dir, sub_args

from pygmy import first_level
from L2_pygmy import second_level

def level1_univariate(phase=None,day1_mem=False,interaction=False,display=False):
	
	for sub in sub_args:

		first_level(subj=sub,phase=phase,day1_mem=day1_mem,
			interaction=interaction,display=display)



# for phase in ['baseline', 'fear_conditioning', 'extinction', 'extinction_recall']:
	# level1_univariate(phase=phase)
	# second_level(phase=phase)

#for phase in ['baseline', 'fear_conditioning', 'extinction']:
for phase in ['fear_conditioning']:
	level1_univariate(phase=phase, day1_mem=True, interaction=True, display=False)
	for stat in ['z_map']:#,'eff_size','eff_var']:
		second_level(phase=phase, day1_mem=True, interaction=True, stat_type=stat)
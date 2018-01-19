#run the level 1s
from fc_config import data_dir, sub_args

from pygmy import first_level

class level1_univariate(object):

	def __init__(self,phase,display=False):

		self.run_sub_lev1(phase,display)

	def run_sub_lev1(self,phase,display=False):
	
		for sub in sub_args:

			first_level(sub,phase,display)
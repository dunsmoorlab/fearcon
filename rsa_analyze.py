import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from fc_config import data_dir, get_sub_args

from lighthouse import rsa


class group_rsa(object):

	sub_args = get_sub_args()

	def __init__(self):
		
		print('WOWDODWHAHW')

		self.combined = pd.DataFrame([])

		self.a_combined = pd.DataFrame([])
		self.t_combined = pd.DataFrame([])

		self.base_combined = pd.DataFrame([])
		self.fear_combined = pd.DataFrame([])
		self.ext_combined = pd.DataFrame([])
		self.mem1_combined = pd.DataFrame([])
		self.mem2_combined = pd.DataFrame([])
		self.mem3_combined = pd.DataFrame([])

		self.a_base_combined = pd.DataFrame([])
		self.a_fear_combined = pd.DataFrame([])
		self.a_ext_combined = pd.DataFrame([])
		self.a_mem1_combined = pd.DataFrame([])
		self.a_mem2_combined = pd.DataFrame([])
		self.a_mem3_combined = pd.DataFrame([])

		self.t_base_combined = pd.DataFrame([])
		self.t_fear_combined = pd.DataFrame([])
		self.t_ext_combined = pd.DataFrame([])
		self.t_mem1_combined = pd.DataFrame([])
		self.t_mem2_combined = pd.DataFrame([])
		self.t_mem3_combined = pd.DataFrame([])


		self.get_sub_rsa()

		self.a_combined = self.clean_group_rsa(self.a_combined)
		self.t_combined = self.clean_group_rsa(self.t_combined)

		self.combined = self.clean_group_rsa(self.combined)

		self.base_combined = self.clean_group_rsa(self.base_combined)
		self.fear_combined = self.clean_group_rsa(self.fear_combined)
		self.ext_combined = self.clean_group_rsa(self.ext_combined)
		self.mem1_combined = self.clean_group_rsa(self.mem2_combined)
		self.mem2_combined = self.clean_group_rsa(self.mem2_combined)
		self.mem3_combined = self.clean_group_rsa(self.mem3_combined)

		self.a_base_combined = self.clean_group_rsa(self.a_base_combined)
		self.a_fear_combined = self.clean_group_rsa(self.a_fear_combined)
		self.a_ext_combined = self.clean_group_rsa(self.a_ext_combined)
		self.a_mem1_combined = self.clean_group_rsa(self.a_mem1_combined)
		self.a_mem2_combined = self.clean_group_rsa(self.a_mem2_combined)
		self.a_mem3_combined = self.clean_group_rsa(self.a_mem3_combined)

		self.t_base_combined = self.clean_group_rsa(self.t_base_combined)
		self.t_fear_combined = self.clean_group_rsa(self.t_fear_combined)
		self.t_ext_combined = self.clean_group_rsa(self.t_ext_combined)
		self.t_mem1_combined = self.clean_group_rsa(self.t_mem1_combined)
		self.t_mem2_combined = self.clean_group_rsa(self.t_mem2_combined)
		self.t_mem3_combined = self.clean_group_rsa(self.t_mem3_combined)


		self.a_combined.to_csv('%s/graphing/RSA/comp_mats/a_combined.csv'%(data_dir),sep=',')
		self.t_combined.to_csv('%s/graphing/RSA/comp_mats/t_combined.csv'%(data_dir),sep=',')

		self.combined.to_csv('%s/graphing/RSA/comp_mats/combined.csv'%(data_dir),sep=',')

		self.base_combined.to_csv('%s/graphing/RSA/comp_mats/base_combined.csv'%(data_dir),sep=',')
		self.fear_combined.to_csv('%s/graphing/RSA/comp_mats/fear_combined.csv'%(data_dir),sep=',')
		self.ext_combined.to_csv('%s/graphing/RSA/comp_mats/ext_combined.csv'%(data_dir),sep=',')
		self.mem1_combined.to_csv('%s/graphing/RSA/comp_mats/mem2_combined.csv'%(data_dir),sep=',')
		self.mem2_combined.to_csv('%s/graphing/RSA/comp_mats/mem2_combined.csv'%(data_dir),sep=',')
		self.mem3_combined.to_csv('%s/graphing/RSA/comp_mats/mem3_combined.csv'%(data_dir),sep=',')

		self.a_base_combined.to_csv('%s/graphing/RSA/comp_mats/a_base_combined.csv'%(data_dir),sep=',')
		self.a_fear_combined.to_csv('%s/graphing/RSA/comp_mats/a_fear_combined.csv'%(data_dir),sep=',')
		self.a_ext_combined.to_csv('%s/graphing/RSA/comp_mats/a_ext_combined.csv'%(data_dir),sep=',')
		self.a_mem1_combined.to_csv('%s/graphing/RSA/comp_mats/a_mem1_combined.csv'%(data_dir),sep=',')
		self.a_mem2_combined.to_csv('%s/graphing/RSA/comp_mats/a_mem2_combined.csv'%(data_dir),sep=',')
		self.a_mem3_combined.to_csv('%s/graphing/RSA/comp_mats/a_mem3_combined.csv'%(data_dir),sep=',')

		self.t_base_combined.to_csv('%s/graphing/RSA/comp_mats/t_base_combined.csv'%(data_dir),sep=',')
		self.t_fear_combined.to_csv('%s/graphing/RSA/comp_mats/t_fear_combined.csv'%(data_dir),sep=',')
		self.t_ext_combined.to_csv('%s/graphing/RSA/comp_mats/t_ext_combined.csv'%(data_dir),sep=',')
		self.t_mem1_combined.to_csv('%s/graphing/RSA/comp_mats/t_mem1_combined.csv'%(data_dir),sep=',')
		self.t_mem2_combined.to_csv('%s/graphing/RSA/comp_mats/t_mem2_combined.csv'%(data_dir),sep=',')
		self.t_mem3_combined.to_csv('%s/graphing/RSA/comp_mats/t_mem3_combined.csv'%(data_dir),sep=',')


	def get_sub_rsa(self):

		for sub in self.sub_args:

			self.sub_rsa = rsa(sub)


			self.combined = pd.concat((self.combined,self.sub_rsa.comp_mat))

			self.base_combined = pd.concat((self.base_combined,self.sub_rsa.comp_phase_stim_patterns('baseline')))
			self.fear_combined = pd.concat((self.fear_combined,self.sub_rsa.comp_phase_stim_patterns('fear_conditioning')))
			self.ext_combined = pd.concat((self.ext_combined,self.sub_rsa.comp_phase_stim_patterns('extinction')))
			self.mem1_combined = pd.concat((self.mem1_combined,self.sub_rsa.comp_phase_stim_patterns('memory_run_1')))
			self.mem2_combined = pd.concat((self.mem2_combined,self.sub_rsa.comp_phase_stim_patterns('memory_run_2')))
			self.mem3_combined = pd.concat((self.mem3_combined,self.sub_rsa.comp_phase_stim_patterns('memory_run_3')))


			if self.sub_rsa.csplus == 'animal':

				self.a_combined = pd.concat((self.a_combined,self.sub_rsa.comp_mat))

				self.a_base_combined = pd.concat((self.a_base_combined,self.sub_rsa.comp_phase_stim_patterns('baseline')))
				self.a_fear_combined = pd.concat((self.a_fear_combined,self.sub_rsa.comp_phase_stim_patterns('fear_conditioning')))
				self.a_ext_combined = pd.concat((self.a_ext_combined,self.sub_rsa.comp_phase_stim_patterns('extinction')))
				self.a_mem1_combined = pd.concat((self.a_mem1_combined,self.sub_rsa.comp_phase_stim_patterns('memory_run_1')))
				self.a_mem2_combined = pd.concat((self.a_mem2_combined,self.sub_rsa.comp_phase_stim_patterns('memory_run_2')))
				self.a_mem3_combined = pd.concat((self.a_mem3_combined,self.sub_rsa.comp_phase_stim_patterns('memory_run_3')))


			if self.sub_rsa.csplus == 'tool':

				self.t_combined = pd.concat((self.t_combined,self.sub_rsa.comp_mat))

				self.t_base_combined = pd.concat((self.t_base_combined,self.sub_rsa.comp_phase_stim_patterns('baseline')))
				self.t_fear_combined = pd.concat((self.t_fear_combined,self.sub_rsa.comp_phase_stim_patterns('fear_conditioning')))
				self.t_ext_combined = pd.concat((self.t_ext_combined,self.sub_rsa.comp_phase_stim_patterns('extinction')))
				self.t_mem1_combined = pd.concat((self.t_mem1_combined,self.sub_rsa.comp_phase_stim_patterns('memory_run_1')))
				self.t_mem2_combined = pd.concat((self.t_mem2_combined,self.sub_rsa.comp_phase_stim_patterns('memory_run_2')))
				self.t_mem3_combined = pd.concat((self.t_mem3_combined,self.sub_rsa.comp_phase_stim_patterns('memory_run_3')))


	def clean_group_rsa(self,long_mat):

		self.comb_row_index = long_mat.groupby(long_mat.index,sort=False)

		return self.comb_row_index.mean()


# s1 = rsa(1).comp_mat
# s2 = rsa(2).comp_mat
# s3 = rsa(3).comp_mat
# s4 = rsa(4).comp_mat
# s5 = rsa(5).comp_mat
# s6 = rsa(6).comp_mat
# s7 = rsa(7).comp_mat
# s8 = rsa(8).comp_mat
# s9 = rsa(9).comp_mat
# s10 = rsa(10).comp_mat

# s_combined = pd.concat((s1,s2,s3,s4,s5,s6,s7,s8,s9,s10))
# s_combined = pd.concat((s1,s2,s3))


# comb_row_index = s_combined.groupby(s_combined.index,sort=False)

# mean_rsa = comb_row_index.mean()




# plt.pcolor(mean_rsa)
# plt.yticks(np.arange(0.5, len(mean_rsa.index), 1), mean_rsa.index)
# plt.xticks(np.arange(0.5, len(mean_rsa.columns), 1), mean_rsa.columns)
# plt.show()
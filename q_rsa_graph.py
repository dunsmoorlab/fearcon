import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from fc_config import data_dir
from mpl_toolkits.axes_grid1 import ImageGrid


a_combined = pd.read_csv('%s/graphing/RSA/comp_mats/a_combined.csv'%(data_dir),sep=',',index_col=0)
t_combined = pd.read_csv('%s/graphing/RSA/comp_mats/t_combined.csv'%(data_dir),sep=',',index_col=0)

combined = pd.read_csv('%s/graphing/RSA/comp_mats/combined.csv'%(data_dir),sep=',',index_col=0)

base_combined = pd.read_csv('%s/graphing/RSA/comp_mats/base_combined.csv'%(data_dir),sep=',',index_col=0)
fear_combined = pd.read_csv('%s/graphing/RSA/comp_mats/fear_combined.csv'%(data_dir),sep=',',index_col=0)
ext_combined = pd.read_csv('%s/graphing/RSA/comp_mats/ext_combined.csv'%(data_dir),sep=',',index_col=0)
mem1_combined = pd.read_csv('%s/graphing/RSA/comp_mats/mem2_combined.csv'%(data_dir),sep=',',index_col=0)
mem2_combined = pd.read_csv('%s/graphing/RSA/comp_mats/mem2_combined.csv'%(data_dir),sep=',',index_col=0)
mem3_combined = pd.read_csv('%s/graphing/RSA/comp_mats/mem3_combined.csv'%(data_dir),sep=',',index_col=0)

a_base_combined = pd.read_csv('%s/graphing/RSA/comp_mats/a_base_combined.csv'%(data_dir),sep=',',index_col=0)
a_fear_combined = pd.read_csv('%s/graphing/RSA/comp_mats/a_fear_combined.csv'%(data_dir),sep=',',index_col=0)
a_ext_combined = pd.read_csv('%s/graphing/RSA/comp_mats/a_ext_combined.csv'%(data_dir),sep=',',index_col=0)
a_mem1_combined = pd.read_csv('%s/graphing/RSA/comp_mats/a_mem1_combined.csv'%(data_dir),sep=',',index_col=0)
a_mem2_combined = pd.read_csv('%s/graphing/RSA/comp_mats/a_mem2_combined.csv'%(data_dir),sep=',',index_col=0)
a_mem3_combined = pd.read_csv('%s/graphing/RSA/comp_mats/a_mem3_combined.csv'%(data_dir),sep=',',index_col=0)

t_base_combined = pd.read_csv('%s/graphing/RSA/comp_mats/t_base_combined.csv'%(data_dir),sep=',',index_col=0)
t_fear_combined = pd.read_csv('%s/graphing/RSA/comp_mats/t_fear_combined.csv'%(data_dir),sep=',',index_col=0)
t_ext_combined = pd.read_csv('%s/graphing/RSA/comp_mats/t_ext_combined.csv'%(data_dir),sep=',',index_col=0)
t_mem1_combined = pd.read_csv('%s/graphing/RSA/comp_mats/t_mem1_combined.csv'%(data_dir),sep=',',index_col=0)
t_mem2_combined = pd.read_csv('%s/graphing/RSA/comp_mats/t_mem2_combined.csv'%(data_dir),sep=',',index_col=0)
t_mem3_combined = pd.read_csv('%s/graphing/RSA/comp_mats/t_mem3_combined.csv'%(data_dir),sep=',',index_col=0)

_rsa = [combined,base_combined,fear_combined,ext_combined,mem1_combined,mem2_combined,mem3_combined]
a_rsa = [a_combined,a_base_combined,a_fear_combined,a_ext_combined,a_mem1_combined,a_mem2_combined,a_mem3_combined]
t_rsa = [t_combined,t_base_combined,t_fear_combined,t_ext_combined,t_mem1_combined,t_mem2_combined,t_mem3_combined]

phases = ['All_Phases','Baseline','Fear_Conditioning','Extinction','Memory_Run_1','Memory_Run_2','Memory_Run_3']

color_map = 'RdBu'

for i, phase in enumerate(phases):

	plt1 = _rsa[i]
	plt2 = a_rsa[i]
	plt3 = t_rsa[i]

	plt.figure(figsize=(27,9))
	plt.suptitle(phase,fontsize = 20)


	plt.subplot(1,3,1)
	plt.pcolor(plt1, cmap = color_map, vmin=-1, vmax=1 )
	plt.yticks(np.arange(0.5, len(plt1.index), 1), plt1.index)
	plt.xticks(np.arange(0.5, len(plt1.columns), 1), plt1.columns, rotation='vertical')
	plt.title('Combined')
	plt.colorbar()


	plt.subplot(1,3,2)
	plt.pcolor(plt2, cmap = color_map, vmin=-1, vmax=1 )
	plt.yticks(np.arange(0.5, len(plt2.index), 1), plt2.index)
	plt.xticks(np.arange(0.5, len(plt2.columns), 1), plt2.columns, rotation='vertical')
	plt.title('Animal CS+')
	plt.colorbar()


	plt.subplot(1,3,3)
	plt.pcolor(plt3, cmap = color_map, vmin=-1, vmax=1 )
	plt.yticks(np.arange(0.5, len(plt3.index), 1), plt3.index)
	plt.xticks(np.arange(0.5, len(plt3.columns), 1), plt3.columns, rotation='vertical')
	plt.colorbar()
	plt.title('Tool CS+')

	plt.tight_layout()
	plt.subplots_adjust(top=.87)

	plt.savefig('%s/graphing/RSA/%s.png'%(data_dir,phase))

#plt.show()







# Generate some data that where each slice has a different range
# (The overall range is from 0 to 2)



# # Plot each slice as an independent subplot
# fig, axes = plt.subplots(nrows=1, ncols=3)
# for dat, ax in zip(data2graph, axes.flat):
#     # The vmin and vmax arguments specify the color limits
#     im = ax.pcolor(dat, vmin=-1, vmax=1)

# # Make an axis for the colorbar on the right side
# cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
# fig.colorbar(im, cax=cax)

# plt.tight_layout()    






plt1.sim = 0
for cond in plt1:
	plt1.sim += ( sum(plt1[cond][cond:]) - plt1[cond][cond] )
plt1.sim /= ( (plt1.shape[0] * plt1.shape[1]) - plt1.shape[0] ) / 2



plt2.sim = 0
for cond in plt2:
	plt2.sim += ( sum(plt2[cond][cond:]) - plt2[cond][cond] )
plt2.sim /= ( (plt2.shape[0] * plt2.shape[1]) - plt2.shape[0] ) / 2



plt3.sim = 0
for cond in plt3:
	plt3.sim += ( sum(plt3[cond][cond:]) - plt3[cond][cond] )
plt3.sim /= ( (plt3.shape[0] * plt3.shape[1]) - plt3.shape[0] ) / 2


print(plt1.sim,plt2.sim,plt3.sim)























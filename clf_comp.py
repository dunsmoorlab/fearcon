from fc_decoding import loc_decode, eval_xval
from fc_config import *
from cf_mat_plot import plot_confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns


imgs = ['tr','beta']
save_dicts = [ppa_prepped,mvpa_masked_prepped]

for _img in imgs:
	for _save_dict in save_dicts:
		if _save_dict is ppa_prepped: k='all'
		else: k = 300
		eval_xval(name='all cat', imgs=_img, del_rest=True,
				scene_collapse=False, scene_DS=False, rmv_scram=False, rmv_indoor=False, binarize=False,
				p='all', save_dict=_save_dict, k=[k], cf=True)


		eval_xval(name='binary', imgs=_img, del_rest=False,
			scene_collapse=False, scene_DS=False, rmv_scram=False, rmv_indoor=False, binarize=True,
			p='all', save_dict=_save_dict, k=['all'], cf=True)


eval_xval(name='binary', imgs='beta', del_rest=False,
	scene_collapse=False, scene_DS=False, rmv_scram=False, rmv_indoor=False, binarize=True,
	p='all', save_dict=beta_ppa_prepped, k=['all'], cf=True)

for imgs in ['tr','beta']:
	eval_xval(name='all cats', imgs=imgs, del_rest=False,
		scene_collapse=False, scene_DS=False, rmv_scram=False, rmv_indoor=False, binarize=False,
		p='all', save_dict=mvpa_masked_prepped, k=[300], cf=True)



eval_xval(name='Collapse Scenes & No Scrambled', imgs='tr', del_rest=False,
		scene_collapse=True, scene_DS=True, rmv_scram=True, rmv_indoor=False, binarize=False,
		p='all', save_dict=ppa_prepped, k=['all'], cf=True)


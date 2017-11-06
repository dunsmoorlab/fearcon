import os
import argparse
import numpy as np
import pandas as pd
from math import sqrt

#set arguments
parser = argparse.ArgumentParser(description='Function arguments')
#set the subject argument as a string so that it can take arguments like '1' or 'all'
parser.add_argument('-s','--subj', nargs = '+', help='Subject number', default=00, type=str)
parser.add_argument('-p','--phase', help='Phase', default = '', type=str)
args = parser.parse_args()

#point to the data direct
data_dir = '/Users/ach3377/GoogleDrive/FC_FMRI_DATA/'

if args.phase == 'er':
	phase = 'extinction_recall'

#goal of this script is to appropriately collapse 3 different Autonomate analyses (5s, 4.5s, 4s) for each subject
scr_meta = pd.read_csv('%s/SCR_Analyzed/%s_full_batch.txt'%(data_dir,phase), delimiter='\t')

for sub in args.subj:

	SUBJ = 'Sub00%s'%(sub)

	if args.phase == 'er':
		sphase = '%s_day2ER'%(SUBJ)

	phase_meta = pd.read_csv('%s/%s/behavior/run_logs/%s_meta.csv'%(data_dir,SUBJ,phase))

	sub_scr = scr_meta[scr_meta.File == sphase]

	sub_out = pd.DataFrame([],columns=['Sub','Phase','Trial','CStype','Duration','t2p','Onset','scr_Duration'])

	sub_out.Duration = phase_meta['stim.Duration'] / 1000

	sub_out.CStype = phase_meta.cstype

	sub_out.Sub = [SUBJ] * len(sub_out)

	sub_out.Phase = [phase] * len(sub_out)

	sub_out.Trial = phase_meta.Block

	for i, trial in enumerate(sub_out.Trial):

		trail_dat = sub_scr.loc

		sub_out.loc[i,'t2p'] = sub_scr.t2pValue.loc[sub_scr.Event == trial,].loc[sub_scr.Analysis == '%ssec'%(sub_out.Duration[i]),].values[0]

		#(sub_scr[sub_scr.Event == trial][sub_scr.Analysis == '%ssec'%(sub_out.Duration[i])])

	for i, scr in enumerate(sub_out.t2p):
		sub_out.loc[i,'t2p'] = sqrt(scr)


	sub_out.to_csv('%s/%s/SCR/%s_analyzed_scr.csv'%(data_dir,SUBJ,phase),sep=',')
import numpy as np
import pandas as pd

def lil_bootstrap():
	ev = pd.read_csv('/Users/ach3377/Db_lpl/STUDY/FearCon/graphing/signal_change/mvpa_ev.csv')

	c = np.array(ev.ev.loc[np.where(ev.Group == 'Control')[0]])
	p = np.array(ev.ev.loc[np.where(ev.Group== 'PTSD')[0]])

	n_rep = 1000
	n_boot = 1000
	n_c = len(c)
	n_p = len(p)

	summary = np.zeros(n_rep)

	for k in range(n_rep):

		diff = np.zeros(n_boot)

		for i in range(n_boot):

			c_b = np.zeros(n_c)
			p_b = np.zeros(n_p)

			for j in range(n_c): c_b[j] = c[np.random.randint(n_c)]
			for j in range(n_p): p_b[j] = p[np.random.randint(n_p)]

			diff_b = c_b.mean() - p_b.mean()

			if diff_b > 0: 
				diff[i] = 1
			else:
				diff[i] = 0

		d = diff.sum() / n_boot
		
		summary[k] = d

	print('mean = %s, 5/95 = %s/%s'%(summary.mean(),np.percentile(summary,5),np.percentile(summary,95)))




  response  tr    t_stat     p_val
0   expect   0 -0.214474  0.835543
1   expect   1 -1.486525  0.175450
2       no   0  3.733330  0.003889
3       no   1  2.141599  0.057879
   statistic    pvalue
0   2.231224  0.038623
1   2.525618  0.021149


See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
  self._setitem_with_indexer(indexer, value)
1                         sum_sq    df         F    PR(>F)
C(group)              0.064599   1.0  0.349267  0.558221
C(response)           0.847251   1.0  4.580820  0.039179
C(group):C(response)  0.648347   1.0  3.505408  0.069309
Residual              6.658423  36.0       NaN       NaN
Ttest_indResult(statistic=0.5486797072507276, pvalue=0.5864348373244812)
1 c_no v c_exp Ttest_indResult(statistic=2.65641746066733, pvalue=0.016069389498580177)
1 c_no v p_no Ttest_indResult(statistic=1.8071460278300706, pvalue=0.08580755053835737)
1 c_no v p_exp Ttest_indResult(statistic=1.658177798571691, pvalue=0.11460588275721283)
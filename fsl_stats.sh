#this is the general call
#the -D flag demeans the data
#add variace smoothing with -v 5
randomise -i 4d_input_data -o output_rootname -d design.mat -t design.con -m mask.nii.gz -n 5000 -D -T -v 5

#one sample t-test
randomise -i filtered_func_data -m mask -o onesamp_t/greater_zero -1 -T -n 5000 -v 5
#tfce is 1-p

#two sample t-test
#have to make the design matrix and contrast file using Glm_gui
randomise -i two_samp_data -o two_samp/twosamp -d two_samp/desmat.mat -t two_samp/desmat.con -T -n -v 5

#min/max in an image
fslstats tfce_corrp_tstat1.nii.gz -R

#how many above significance?
fslstats tfce_corrp_tstat1.nii.gz -l 0.95 -V


randomise -i filtered_func_data.nii.gz -o ttest_stats -d rand_design.mat -t rand_design.con -m group_func_vmPFC_mask.nii.gz -n 1000 -D -T -v 5

#/work/05426/ach3377/lonestar/group_glm/run003/ext_ev/control_ext_ev_corr.gfeat/cope3.feat/
#/work/05426/ach3377/lonestar/group_glm/randomise/
#/work/05426/ach3377/lonestar/group_glm/run004/group_ev_corr/control_ev_corr.gfeat/cope3.feat/

fslmerge -t control_xval.nii.gz xval_std_1.nii.gz xval_std_2.nii.gz xval_std_3.nii.gz xval_std_4.nii.gz xval_std_5.nii.gz xval_std_6.nii.gz xval_std_7.nii.gz xval_std_8.nii.gz xval_std_9.nii.gz xval_std_10.nii.gz xval_std_12.nii.gz xval_std_13.nii.gz xval_std_14.nii.gz xval_std_15.nii.gz xval_std_16.nii.gz xval_std_17.nii.gz xval_std_18.nii.gz xval_std_19.nii.gz xval_std_20.nii.gz xval_std_21.nii.gz xval_std_23.nii.gz xval_std_24.nii.gz xval_std_25.nii.gz xval_std_26.nii.gz
fslmerge -t ptsd_xval.nii.gz xval_std_101.nii.gz xval_std_102.nii.gz xval_std_103.nii.gz xval_std_104.nii.gz xval_std_105.nii.gz xval_std_106.nii.gz xval_std_107.nii.gz xval_std_108.nii.gz xval_std_109.nii.gz xval_std_110.nii.gz xval_std_111.nii.gz xval_std_112.nii.gz xval_std_113.nii.gz xval_std_114.nii.gz xval_std_115.nii.gz xval_std_116.nii.gz xval_std_117.nii.gz xval_std_118.nii.gz xval_std_120.nii.gz xval_std_121.nii.gz xval_std_122.nii.gz xval_std_123.nii.gz xval_std_124.nii.gz xval_std_125.nii.gz
fslmerge -t all_xval.nii.gz control_xval.nii.gz ptsd_xval.nii.gz

fslmerge -t control_rnw.nii.gz rnw_std_1.nii.gz rnw_std_2.nii.gz rnw_std_3.nii.gz rnw_std_4.nii.gz rnw_std_5.nii.gz rnw_std_6.nii.gz rnw_std_7.nii.gz rnw_std_8.nii.gz rnw_std_9.nii.gz rnw_std_10.nii.gz rnw_std_12.nii.gz rnw_std_13.nii.gz rnw_std_14.nii.gz rnw_std_15.nii.gz rnw_std_16.nii.gz rnw_std_17.nii.gz rnw_std_18.nii.gz rnw_std_19.nii.gz rnw_std_20.nii.gz rnw_std_21.nii.gz rnw_std_23.nii.gz rnw_std_24.nii.gz rnw_std_25.nii.gz rnw_std_26.nii.gz
fslmerge -t ptsd_rnw.nii.gz rnw_std_101.nii.gz rnw_std_102.nii.gz rnw_std_103.nii.gz rnw_std_104.nii.gz rnw_std_105.nii.gz rnw_std_106.nii.gz rnw_std_107.nii.gz rnw_std_108.nii.gz rnw_std_109.nii.gz rnw_std_110.nii.gz rnw_std_111.nii.gz rnw_std_112.nii.gz rnw_std_113.nii.gz rnw_std_114.nii.gz rnw_std_115.nii.gz rnw_std_116.nii.gz rnw_std_117.nii.gz rnw_std_118.nii.gz rnw_std_120.nii.gz rnw_std_121.nii.gz rnw_std_122.nii.gz rnw_std_123.nii.gz rnw_std_124.nii.gz rnw_std_125.nii.gz
fslmerge -t all_rnw.nii.gz control_rnw.nii.gz ptsd_rnw.nii.gz

fslmerge -t control_ext.nii.gz ext_std_1.nii.gz ext_std_2.nii.gz ext_std_3.nii.gz ext_std_4.nii.gz ext_std_5.nii.gz ext_std_6.nii.gz ext_std_7.nii.gz ext_std_8.nii.gz ext_std_9.nii.gz ext_std_10.nii.gz ext_std_12.nii.gz ext_std_13.nii.gz ext_std_14.nii.gz ext_std_15.nii.gz ext_std_16.nii.gz ext_std_17.nii.gz ext_std_18.nii.gz ext_std_19.nii.gz ext_std_20.nii.gz ext_std_21.nii.gz ext_std_23.nii.gz ext_std_24.nii.gz ext_std_25.nii.gz ext_std_26.nii.gz
fslmerge -t ptsd_ext.nii.gz ext_std_101.nii.gz ext_std_102.nii.gz ext_std_103.nii.gz ext_std_104.nii.gz ext_std_105.nii.gz ext_std_106.nii.gz ext_std_107.nii.gz ext_std_108.nii.gz ext_std_109.nii.gz ext_std_110.nii.gz ext_std_111.nii.gz ext_std_112.nii.gz ext_std_113.nii.gz ext_std_114.nii.gz ext_std_115.nii.gz ext_std_116.nii.gz ext_std_117.nii.gz ext_std_118.nii.gz ext_std_120.nii.gz ext_std_121.nii.gz ext_std_122.nii.gz ext_std_123.nii.gz ext_std_124.nii.gz ext_std_125.nii.gz
fslmerge -t all_ext.nii.gz control_ext.nii.gz ptsd_ext.nii.gz

fslmerge -t control_day1.nii.gz day1_std_1.nii.gz day1_std_2.nii.gz day1_std_3.nii.gz day1_std_4.nii.gz day1_std_5.nii.gz day1_std_6.nii.gz day1_std_7.nii.gz day1_std_8.nii.gz day1_std_9.nii.gz day1_std_10.nii.gz day1_std_12.nii.gz day1_std_13.nii.gz day1_std_14.nii.gz day1_std_15.nii.gz day1_std_16.nii.gz day1_std_17.nii.gz day1_std_18.nii.gz day1_std_19.nii.gz day1_std_20.nii.gz day1_std_21.nii.gz day1_std_23.nii.gz day1_std_24.nii.gz day1_std_25.nii.gz day1_std_26.nii.gz
fslmerge -t ptsd_day1.nii.gz day1_std_101.nii.gz day1_std_102.nii.gz day1_std_103.nii.gz day1_std_104.nii.gz day1_std_105.nii.gz day1_std_106.nii.gz day1_std_107.nii.gz day1_std_108.nii.gz day1_std_109.nii.gz day1_std_110.nii.gz day1_std_111.nii.gz day1_std_112.nii.gz day1_std_113.nii.gz day1_std_114.nii.gz day1_std_115.nii.gz day1_std_116.nii.gz day1_std_117.nii.gz day1_std_118.nii.gz day1_std_120.nii.gz day1_std_121.nii.gz day1_std_122.nii.gz day1_std_123.nii.gz day1_std_124.nii.gz day1_std_125.nii.gz
fslmerge -t all_day1.nii.gz control_day1.nii.gz ptsd_day1.nii.gz


randomise -i $WORK/group_glm/searchlight/stats/all_xval0.nii.gz -o $WORK/group_glm/searchlight/stats/all_xval_1samp -1 -T -n 1000 -v 5
randomise -i $WORK/group_glm/searchlight/stats/all_rnw.nii.gz -o $WORK/group_glm/searchlight/stats/rnw_2samp -d $WORK/group_glm/searchlight/stats/rand_2samp.mat -t $WORK/group_glm/searchlight/stats/rand_2samp.con -e $WORK/group_glm/searchlight/stats/rand_2samp.grp -T -n 1000 -T -v 5
randomise -i $WORK/group_glm/searchlight/stats/all_ext.nii.gz -o $WORK/group_glm/searchlight/stats/ext_2samp -d $WORK/group_glm/searchlight/stats/rand_2samp.mat -t $WORK/group_glm/searchlight/stats/rand_2samp.con -e $WORK/group_glm/searchlight/stats/rand_2samp.grp -T -n 1000 -T -v 5
randomise -i $WORK/group_glm/searchlight/stats/all_day1.nii.gz -o $WORK/group_glm/searchlight/stats/day1_2samp -d $WORK/group_glm/searchlight/stats/rand_2samp.mat -t $WORK/group_glm/searchlight/stats/rand_2samp.con -T -n 1000 -T -v 5



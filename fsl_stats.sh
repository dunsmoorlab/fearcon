#this is the general call
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
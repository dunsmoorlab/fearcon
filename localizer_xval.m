%FearCon localizer classification with logreg and ANOVA feature selection
function [out_perf] = localizer_xval(sub, mask, cat, thresh)

data_dir = '/Users/ach3377/GoogleDrive/FC_FMRI_DATA/';


%uncomment this for easy debugging
%SUBJ = 'Sub001'

%set subject for each iteration
SUBJ = strcat('Sub00',sub)
%as well as the subject directory
sub_dir = strcat(data_dir, SUBJ);

%initialize the subject for the toolbox
subj = init_subj('FearCon',SUBJ);


%point to the mask
%mask input should be 'VTC' or 'LOC_VTC'
mask_path = strcat(sub_dir, '/mask/', mask, '_mask.nii.gz');
%load in the mask
subj = load_spm_mask(subj, mask, mask_path);


%point to the bold dir
bold_dir = strcat(sub_dir, '/bold');
%point to the motion-corrected functional runs
run_files = {strcat(bold_dir,'/day2/run008/mc_run008.nii.gz'),strcat(bold_dir,'/day2/run009/mc_run009.nii.gz')}; 
%load in the runs
subj = load_spm_pattern(subj, 'Localizer', mask, run_files);


%point the toolbox folder in the model, for regs and sels
toolbox = strcat(sub_dir,'/model/MVPA/toolbox');

if strcmp(cat,'scene')
    day1_regs = strcat(cat,'_localizer_regs');
elseif strcmp(cat, '4cat')
    day1_regs = strcat(cat,'_localizer_regs');
end


%initialize regressors
subj = init_object(subj, 'regressors', '4_cat');
%load them in and set them in the subject
localizer_regs = load(strcat(toolbox, '/4cat_localizer_regs'));
subj = set_mat(subj,'regressors', '4_cat', localizer_regs);
%name and set the conditionins, descending
condnames = {'animal','tool','scene','scrambled'};
subj = set_objfield(subj, 'regressors', '4_cat', 'condnames', condnames);


%load in the selectors
subj = init_object(subj, 'selector', 'runs');
localizer_sels = load(strcat(toolbox, '/2run_localizer_sels'));
subj = set_mat(subj, 'selector', 'runs', localizer_sels);


%detrend
subj = detrend_runs(subj,'Localizer','runs', 'order', 1);


%shift the regressors
subj = shift_regressors(subj, '4_cat','runs', 3);


%and then exclude rest
regs = get_mat(subj, 'regressors','4_cat_sh3');
temp_sel = ones(1,size(regs,2));
temp_sel(sum(regs)==0) = 0;%%%%%%%%%%%%%%%%%%%%%%% example has 'find', but matlab suggests 'sum'
subj = init_object(subj, 'selector', 'no_rest');
%this works by chopping out rest from the regular selectors, so need both
subj = set_mat(subj, 'selector', 'no_rest', temp_sel);


%z-score
subj = zscore_runs(subj,'Localizer_d','runs');


%set up n-fold cross validation to exclude rest
subj = create_xvalid_indices(subj, 'runs', 'actives_selname','no_rest');


%ANOVA feature selection
%thresh input should be format 0.00
subj = feature_select(subj,'Localizer_d_z','4_cat_sh3','runs_xval','thresh',thresh);

%set the fs_name
thresh_str = num2str(thresh);
fs_name = strcat('Localizer_d_z_thresh',thresh_str);

%initialize MVPA classification
class_args.train_funct_name = 'train_logreg';
class_args.test_funct_name = 'test_logreg';
class_args.penalty = 50;


%do MVPA!
[subj results] = cross_validation(subj, 'Localizer_d_z', '4_cat_sh3', 'runs_xval', fs_name, class_args);
out_perf = results.total_perf;


end
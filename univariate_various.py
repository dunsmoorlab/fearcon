from fc_config import *
from preprocess_library import meta
from glm_timing import *
from shutil import copytree, move, copyfile
from nilearn.input_data import NiftiMasker

def pop_fsl_timing(subs=None):
    # if p: subs = p_sub_args
    # else: subs = sub_args

    for sub in subs:
        # for phase in ['baseline','fear_conditioning','extinction','extinction_recall']:
        # if sub == 107: loc_phases = ['localizer_1']
        # else: loc_phases = ['localizer_1','localizer_2']
        for phase in ['memory_run_1','memory_run_2','memory_run_3']:
            # for resp_con in [True, False]:
        # for phase in loc_phases:
            for resp_con in [False]:
                glm_timing(sub,phase).fsl_timing(resp=resp_con)

def run_univariate(sub=0, phase=0, con=None):
    
    subj = meta(sub)

    fsf = os.path.join(subj.model_dir, 'GLM', 'designs', py_run_key[phase], '%s.fsf'%(con))

    os.system('feat %s'%(fsf))

def make_fsf(sub=0, phase=None, con=None):

    day = day2phase[phase][:4]

    template = os.path.join(data_dir,'fsfs','preregistered', day + '.fsf')

    subj = meta(sub)

    design_folder = os.path.join(subj.model_dir, 'GLM', 'designs')
    outdir = os.path.join(design_folder, py_run_key[phase])
    if not os.path.exists(design_folder): os.mkdir(design_folder)
    if not os.path.exists(outdir): os.mkdir(outdir)

    outfile = os.path.join(outdir, '%s.fsf'%(con))

    replacements = {'SUBJID':subj.fsub, 'RUNID':py_run_key[phase],
                    'DAY':day, 'CONID':con}

    with open(template) as temp:

        with open(outfile,'w') as out:

            for line in temp:

                for src, target in replacements.items():
                    line = line.replace(src, target)

                out.write(line)

def warp_input(sub=0, phase=None):

    subj = meta(sub)

    bold = os.path.join(subj.bold_dir, nifti_paths[phase])

    std = os.path.join(subj.bold_dir, std_paths[phase])

    if os.path.exists(std):
        print('Input Already Warped to MNI')
    else:
        func2struct = os.path.join(subj.reg_dir, 'func2struct.mat')
        struct_warp_img = os.path.join(subj.reg_dir, 'struct2std_warp.nii.gz')
        os.system('applywarp --in=%s --ref=$FSL_DIR/data/standard/MNI152_T1_1mm_brain.nii.gz --out=%s --premat=%s --warp=%s'%(
            bold, std, func2struct, struct_warp_img))

def move2tacc():

    # dest = os.path.join('/Volumes/DunsmoorRed','fc')
    dest = os.path.join('/Users/ach3377/Desktop','fc')
    # dest = os.path.join('C:\\Users','ACH','Desktop','fc')
    if not os.path.exists(dest): os.mkdir(dest)

    # for sub in [23,24,25,26,122,123,124,125]:
    for sub in all_sub_args:
        subj = meta(sub)
        sub_dir = os.path.join(dest,subj.fsub)
        os.mkdir(sub_dir)
        bold_dir = os.path.join(sub_dir,'bold')
        # orig = os.path.join(bold_dir,'orig')
        betas = os.path.join(bold_dir,'beta')
        os.makedirs(betas)

        # scr_src = os.path.join(subj.subj_dir,'SCR')
        # behavior_src = os.path.join(subj.subj_dir,'behavior')

        # scr_dest = os.path.join(sub_dir,'SCR')
        # os.system('cp -R %s %s'%(scr_src,sub_dir))
        # os.system('cp -R %s %s'%(behavior_src,sub_dir))
        # os.copytree(scr_src,sub_dir)
        # os.copytree(behavior,sub_dir)

        # behavior_dest = os.path.join(sub_)
        
        # os.makedirs(orig)
        # std = os.path.join(bold_dir, 'std')
        # os.mkdir(std)
        # anat = os.path.join(sub_dir,'anat')
        # os.makedirs(anat)
        
        # os.mkdir(os.path.join(fc,'%s'%(subj.fsub)))

        for phase in fsl_betas:
            if phase == 'localizer_2' and sub == 117: pass
            else:
                run = os.path.join(subj.bold_dir, fsl_betas[phase])
                cpy = os.path.join(betas, os.path.split(fsl_betas[phase])[-1])
                copyfile(run,cpy)

        # for phase in nifti_paths:
            # if phase == 'localizer_2' and sub == 117: pass
            # if 'extinction_recall' in phase:
            # run = os.path.join(subj.bold_dir, nifti_paths[phase])
            # cpy = os.path.join(orig, os.path.split(nifti_paths[phase])[-1])
            # copyfile(run,cpy)
        
        # orig_whole = os.path.join(subj.anatomy,'orig.nii.gz')
        # brainmask = os.path.join(subj.anatomy,'brainmask.nii.gz')
        # orig_brain_new = os.path.join(anat,'orig_brain.nii.gz')
        # os.system('fslmaths %s -mas %s %s'%(orig_whole, brainmask, orig_brain_new))
    
        # model_dir = os.path.join(sub_dir,'model') 
        # os.mkdir(model_dir)
        # pm_dest = os.path.join(model_dir,'onsets','run004')
        # os.makedirs(pm_dest)
        # onsets = os.path.join(subj.model_dir,'GLM','onsets')
        # for file in os.listdir(os.path.join(onsets,'run004')):
            # if 'beta' in file:
                # os.system('cp %s %s'%(os.path.join(onsets,'run004',file), os.path.join(pm_dest,file)))
        # copytree(onsets, os.path.join(model_dir,os.path.split(onsets)[-1]))

        # for run in ['localizer_1','localizer_2','extinction_recall']:
            # orig_fold = os.path.join(subj.model_dir,'GLM','onsets',py_run_key[run])
            # dest_fold = os.path.join(sub_dir,'model','onsets',py_run_key[run])
            # os.makedirs(dest_fold)
            # for file in os.listdir(orig_fold):
                # if 'localizer' in run:
                    # if 'all' in file:
                        # orig = os.path.join(orig_fold,file)
                        # cpy = os.path.join(dest_fold,file)
                        # copyfile(orig,cpy)
        #       else:
        #           orig = os.path.join(orig_fold,file)
        #           cpy = os.path.join(dest_fold,file)
        #           copyfile(orig,cpy)

        # orig1 = os.path.join(subj.anatomy,'orig.nii.gz')
        # cpy = os.path.join(anat, os.path.split(orig1)[-1])
        # copyfile(orig1,cpy)
        # reg_dir = os.path.join(sub_dir,'reg')
        # os.mkdir(reg_dir)
        
        # func2struct = os.path.join(subj.reg_dir, 'func2struct.mat')
        # struct_warp_img = os.path.join(subj.reg_dir, 'struct2std_warp.nii.gz')
        # for src in [func2struct, struct_warp_img]:
        #   copyfile(src, os.path.join(reg_dir, os.path.split(src)[-1]))
        



    # scp -r "/Users/ach3377/Desktop/fc" ach3377@ls5.tacc.utexas.edu:"/work/05426/ach3377/lonestar/"
    # scp -r "/mnt/c/Users/ACH/Desktop/fc" ach3377@ls5.tacc.utexas.edu:"/work/05426/ach3377/lonestar/"
    # scp -r "/Users/ach3377/Db_lpl/STUDY/FearCon/Sub003/anatomy/orig_brain.nii.gz" ach3377@ls5.tacc.utexas.edu:"/work/05426/ach3377/lonestar/fc/Sub003/anat/"
    #/work/05426/ach3377/lonestar/fc/Sub0SUBID/model/onsets/run004/early_CS+.txt
    #scp -r "/mnt/c/Users/ACH/Dropbox (LewPeaLab)/STUDY/FearCon/loc_roi/group_ppa_mask.nii.gz" ach3377@ls5.tacc.utexas.edu:"/work/05426/ach3377/lonestar/fc/"
    #scp -r ach3377@ls5.tacc.utexas.edu:"/work/05426/ach3377/lonestar/fc/group_glm/localizer/sub_masks/" "/mnt/c/Users/ACH/Dropbox (LewPeaLab)/STUDY/FearCon/loc_roi/"
    #scp -r ach3377@ls5.tacc.utexas.edu:"/work/05426/ach3377/lonestar/fc/Sub003/model/run008/locblock.feat" "/Users/ach3377/Desktop/"

def fs_roi_mask(sub):
    subj = meta(sub)

    srcdir = subj.anatomy
    outdir = os.path.join(subj.mask,'roi')
    if not os.path.exists(outdir): os.mkdir(outdir)
    aparc_aseg = os.path.join(srcdir,'aparc+aseg.nii.gz')

    lh_mOFC  = [os.path.join(outdir,'lh_mOFC.nii.gz'),1014]
    rh_mOFC  = [os.path.join(outdir,'rh_mOFC.nii.gz'),2014]

    lh_lOFC  = [os.path.join(outdir,'lh_lOFC.nii.gz'),1012]
    rh_lOFC  = [os.path.join(outdir,'rh_lOFC.nii.gz'),2012]
    
    lh_amyg = [os.path.join(outdir,'lh_amyg.nii.gz'),18]
    rh_amyg = [os.path.join(outdir,'rh_amyg.nii.gz'),54]
    
    lh_hpc = [os.path.join(outdir,'lh_hpc.nii.gz'),17]
    rh_hpc = [os.path.join(outdir,'rh_hpc.nii.gz'),53]

    lh_dACC = [os.path.join(outdir,'lh_dACC.nii.gz'),1002]
    rh_dACC = [os.path.join(outdir,'rh_dACC.nii.gz'),2002]

    for roi in [lh_mOFC,rh_mOFC,lh_lOFC,rh_lOFC]:os.system('fslmaths %s -thr %s -uthr %s %s'%(aparc_aseg, roi[1], roi[1], roi[0]))
    for roi in [lh_mOFC,rh_mOFC]:os.system('fslmaths %s -thr %s -uthr %s %s'%(aparc_aseg, roi[1], roi[1], roi[0]))
    for roi in [lh_amyg,rh_amyg]:   os.system('fslmaths %s -thr %s -uthr %s %s'%(aparc_aseg, roi[1], roi[1], roi[0]))
    for roi in [lh_hpc,rh_hpc]:os.system('fslmaths %s -thr %s -uthr %s %s'%(aparc_aseg, roi[1], roi[1], roi[0]))
    for roi in [lh_dACC,rh_dACC]:os.system('fslmaths %s -thr %s -uthr %s %s'%(aparc_aseg, roi[1], roi[1], roi[0]))

    vmPFC_out = os.path.join(outdir, 'vmPFC_mask.nii.gz')
    mOFC_out = os.path.join(outdir, 'mOFC_mask.nii.gz')
    amyg_out = os.path.join(outdir,'amygdala_mask.nii.gz')
    hpc_out = os.path.join(outdir,'hippocampus_mask.nii.gz')
    dACC_out = os.path.join(outdir,'dACC_mask.nii.gz')

    os.system('fslmaths %s -add %s -add %s -add %s  -bin %s'%(lh_mOFC[0], rh_mOFC[0], lh_lOFC[0], rh_lOFC[0], vmPFC_out))
    os.system('fslmaths %s -add %s -bin %s'%(lh_mOFC[0], rh_mOFC[0], mOFC_out))
    os.system('fslmaths %s -add %s -bin %s'%(lh_amyg[0], rh_amyg[0], amyg_out))
    os.system('fslmaths %s -add %s -bin %s'%(lh_hpc[0], rh_hpc[0], hpc_out))
    os.system('fslmaths %s -add %s -bin %s'%(lh_dACC[0], rh_dACC[0], dACC_out))
    
    struct = os.path.join(srcdir, 'struct.nii.gz')
    o2h = os.path.join(srcdir, 'orig-struct0GenericAffine.mat')
    interp = 'NearestNeighbor'
    
    for mask in [vmPFC_out,mOFC_out,amyg_out,hpc_out,dACC_out]:
        #register the mask to freesurfer_structural space, which is what was used to register to functional space
        #this puts the mask in the same space as orig_brain.nii.gz
        os.system('antsApplyTransforms -i {} -o {} -r {} -t {} -n {}'.format(mask, mask, struct, o2h, interp))

        anat2func = os.path.join(subj.refvol_dir, 'fm', 'epireg_inv.mat')

        #register and resample mask to reference brain extracted functional
        os.system('flirt -in %s -ref %s -applyxfm -init %s -out %s -interp trilinear'%(
            mask, subj.refvol_be, anat2func, mask))

        os.system('fslmaths %s -bin %s'%(mask,mask))

def pe_beta_masker(sub,phase='extinction_recall',roi='PPA',save_dict=None):
    subj = meta(sub)

    masker = NiftiMasker(mask_img=os.path.join(subj.roi,'%s_mask.nii.gz'%(roi)), standardize=True)
    # masker = NiftiMasker(mask_img=subj.ctx_maskvol, standardize=True)

    print('masking runs')
    
    beta_img = masker.fit_transform(os.path.join(subj.bold_dir, phase2rundir[phase],
                                    'fsl_betas','%s_beta.nii.gz'%(py_run_key[phase]) ))

    print('Saving')

    np.savez_compressed( '%s'%(os.path.join(subj.bold_dir, save_dict[phase])),  beta_img )

    # np.savez_compressed( '%s'%(os.path.join(subj.bold_dir, phase2rundir[phase],
    #                   '%s_b_pp_%s.npz'%(roi,py_run_key[phase]))),  beta_img )

def thresh_zmap(thr=2.5758,roi=None):
    roi_dir = os.path.join(data_dir,'group_roi_masks','%s_roi'%(roi))
    thr_z = os.path.join(roi_dir,'thr_z')
    if not os.path.exists(thr_z): os.mkdir(thr_z)
    for file in os.listdir(os.path.join(roi_dir,'unthr_z')):
        if '.nii.gz' in file:
            os.system('fslmaths -dt int %s -thr %s -bin %s'%(os.path.join(roi_dir,'unthr_z',file),thr,os.path.join(thr_z,'thr_'+file)))

def mass_add():

    os.system('fslmaths thr_Sub001_c3z.nii.gz -add thr_Sub002_c3z.nii.gz -add thr_Sub003_c3z.nii.gz -add thr_Sub004_c3z.nii.gz -add thr_Sub005_c3z.nii.gz -add thr_Sub006_c3z.nii.gz -add thr_Sub007_c3z.nii.gz -add thr_Sub008_c3z.nii.gz -add thr_Sub009_c3z.nii.gz -add thr_Sub010_c3z.nii.gz -add thr_Sub012_c3z.nii.gz -add thr_Sub013_c3z.nii.gz -add thr_Sub014_c3z.nii.gz -add thr_Sub015_c3z.nii.gz -add thr_Sub016_c3z.nii.gz -add thr_Sub017_c3z.nii.gz -add thr_Sub018_c3z.nii.gz -add thr_Sub019_c3z.nii.gz -add thr_Sub020_c3z.nii.gz -add thr_Sub021_c3z.nii.gz -add thr_Sub101_c3z.nii.gz -add thr_Sub102_c3z.nii.gz -add thr_Sub103_c3z.nii.gz -add thr_Sub104_c3z.nii.gz -add thr_Sub105_c3z.nii.gz -add thr_Sub106_c3z.nii.gz -add thr_Sub108_c3z.nii.gz -add thr_Sub109_c3z.nii.gz -add thr_Sub110_c3z.nii.gz -add thr_Sub111_c3z.nii.gz -add thr_Sub112_c3z.nii.gz -add thr_Sub113_c3z.nii.gz -add thr_Sub114_c3z.nii.gz -add thr_Sub115_c3z.nii.gz -add thr_Sub116_c3z.nii.gz -add thr_Sub117_c3z.nii.gz -add thr_Sub118_c3z.nii.gz -add thr_Sub120_c3z.nii.gz -add thr_Sub121_c3z.nii.gz ../all_sub_c3')
    
    os.system('fslmaths Sub001.nii.gz -add Sub002.nii.gz -add Sub003.nii.gz -add Sub004.nii.gz -add Sub005.nii.gz -add Sub006.nii.gz -add Sub007.nii.gz -add Sub008.nii.gz -add Sub009.nii.gz -add Sub010.nii.gz -add Sub012.nii.gz -add Sub013.nii.gz -add Sub014.nii.gz -add Sub015.nii.gz -add Sub016.nii.gz -add Sub017.nii.gz -add Sub018.nii.gz -add Sub019.nii.gz -add Sub020.nii.gz -add Sub021.nii.gz -add Sub023.nii.gz -add Sub024.nii.gz -add Sub025.nii.gz -add Sub026.nii.gz -add Sub101.nii.gz -add Sub102.nii.gz -add Sub103.nii.gz -add Sub104.nii.gz -add Sub105.nii.gz -add Sub106.nii.gz -add Sub108.nii.gz -add Sub109.nii.gz -add Sub110.nii.gz -add Sub111.nii.gz -add Sub112.nii.gz -add Sub113.nii.gz -add Sub114.nii.gz -add Sub115.nii.gz -add Sub116.nii.gz -add Sub117.nii.gz -add Sub118.nii.gz -add Sub120.nii.gz -add Sub121.nii.gz -add Sub122.nii.gz -add Sub123.nii.gz -add Sub124.nii.gz -add Sub125.nii.gz ../gmasks/group_c3.nii.gz') 

    os.system('fslmaths all_sub_c3.nii.gz -thr 36 -bin thr_c3z')

    os.sytem('cluster --in=all_sub_c3.nii.gz --thresh=36 --mm -o cluster_thr_c3z > 36_cluster.txt')

    os.system('fslmaths -dt int cluster_thr_c3z -thr 2 -bin group_ppa_mask')

def uni_roi_mask(p=False):


    mask_loc = os.path.join(data_dir,'group_roi_masks','loc_roi','sub_masks')
    for sub in all_sub_args:
        print(sub)
        subj = meta(sub)
        sub_mask = os.path.join(mask_loc,'%s_ppa_mask.nii.gz'%(subj.fsub))
        dest = os.path.join(subj.roi,'PPA_mask.nii.gz')

        # os.system('cp %s %s'%(sub_mask,dest))

        os.system('fslmaths %s -bin %s'%(sub_mask, dest))

def combine_masks():

    for sub in all_sub_args:
        subj = meta(sub)

        ppa = os.path.join(subj.roi,'PPA_mask.nii.gz')

        old = os.path.join(subj.roi,'LOC_PPA_mask.nii.gz')
        new = os.path.join(subj.roi,'VTC_PPA_mask.nii.gz')
        os.system('rm %s'%(old))
        os.system('fslmaths %s -add %s -bin %s'%(ppa,subj.ctx_maskvol,new))

















import os
import nibabel as nib
from fc_config import *
from preprocess_library import meta
from subprocess import Popen

# os.system('source fsdev.sh')

for sub in all_sub_args:
    print(sub)
    subj = meta(sub)

    for hemi in ['r','l']:
        vol = os.path.join(subj.fs_mask,hemi+'h_HCA.nii.gz')
        l2v =['mri_label2vol',
                '--subject', '%s/%sfs'%(subj.fsub,subj.fsub) , 
                '--seg', os.path.join(subj.fs_dir,'mri',hemi+'h.hippoAmygLabels-T1.v21.HBT.FSvoxelSpace.mgz'),
                '--temp', subj.refvol_be,
                '--reg', subj.fs_regmat,
                '--fillthresh', '.3' ,
                '--o', vol]

        Popen(l2v).wait()

        hc_rois = {'hc_head':232,
                   'hc_body':231,
                   'hc_tail':226}
        
        for roi in hc_rois:
            thr = str(hc_rois[roi])
            out = os.path.join(subj.fs_mask,hemi+'h_%s.nii.gz'%(roi))
            hc_cmd = ['fslmaths', vol,
                          '-thr', thr,
                         '-uthr', thr,
                          '-bin', out]
            Popen(hc_cmd).wait()

        amyg_rois = {'amyg_bla':[7001,7003],
                     'amyg_cem':[7004,7010]}
        for roi in amyg_rois:
            lthr = str(amyg_rois[roi][0])
            uthr = str(amyg_rois[roi][1])
            out = os.path.join(subj.fs_mask,hemi+'h_%s.nii.gz'%(roi))
            amyg_cmd = ['fslmaths', vol,
                            '-thr', lthr,
                           '-uthr', uthr,
                            '-bin', out ]
            Popen(amyg_cmd).wait()



    for roi in ['hc_head','hc_body','hc_tail','amyg_bla','amyg_cem']:
        cmd = ['fslmaths', os.path.join(subj.fs_mask,'rh_'+roi+'.nii.gz'),
                   '-add', os.path.join(subj.fs_mask,'lh_'+roi+'.nii.gz'),
                   '-bin', os.path.join(subj.fs_mask,roi+'.nii.gz')]
        Popen(cmd).wait()

    # rois = {'vmPFC':'medialorbitofrontal',
    #         'dACC' :'caudalanteriorcingulate'}
    # for roi in rois:
    #     for hemi in ['r','l']:
    #         vol = os.path.join(subj.fs_mask,hemi+'h_%s.nii.gz'%(roi))
    #         cmd =['mri_label2vol',
    #                 '--subject', '%s/%sfs'%(subj.fsub,subj.fsub) , 
    #                 '--label', os.path.join(subj.fs_dir,'annot2label','%sh.%s.label'%(hemi,rois[roi])),
    #                 '--temp', subj.refvol_be,
    #                 '--proj', 'frac', '0', '1', '.1' ,
    #                 '--reg', subj.fs_regmat,
    #                 '--hemi', '%sh'%(hemi),
    #                 '--fillthresh', '.3' ,
    #                 '--o', vol]
    #         Popen(cmd).wait()  
  
    #     cmd = ['fslmaths', os.path.join(subj.fs_mask,'rh_%s.nii.gz'%(roi)),
    #             '-add', os.path.join(subj.fs_mask,'lh_%s.nii.gz'%(roi)),
    #             '-bin', os.path.join(subj.fs_mask,'%s.nii.gz'%(roi))]
    #     Popen(cmd).wait()
    fastcopy(os.path.join(subj.roi,'mOFC_mask.nii.gz'),os.path.join(subj.fs_mask,'vmPFC.nii.gz'))
    fastcopy(os.path.join(subj.roi,'dACC_mask.nii.gz'),os.path.join(subj.fs_mask,'dACC.nii.gz'))
    
    #make the full circuit mask
    cmd = ['fslmaths', os.path.join(subj.fs_mask,'hc_head.nii.gz'),
           '-add', os.path.join(subj.fs_mask,'hc_body.nii.gz'),
           '-add', os.path.join(subj.fs_mask,'hc_tail.nii.gz'),
           '-add', os.path.join(subj.fs_mask,'amyg_bla.nii.gz'),
           '-add', os.path.join(subj.fs_mask,'amyg_cem.nii.gz'),
           '-add', os.path.join(subj.fs_mask,'vmPFC.nii.gz'),
           '-add', os.path.join(subj.fs_mask,'dACC.nii.gz'),
           '-bin', os.path.join(subj.fs_mask,'circuit_mask.nii.gz')]
    Popen(cmd).wait()
    




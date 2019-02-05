import os
from glob import glob
import codecs

fcdir = '/work/05426/ach3377/lonestar/fc'
os.system('rm final_l1s.txt')
for sub in range(1,22):
        if sub == 11: pass
        else:
                _sub = 'Sub0{0:02d}'.format(sub)
                subdir = os.path.join(fcdir,_sub,'model')

                for run in range(1,5):
                        rundir = os.path.join(subdir,'run00%s'%(run))

                        for con in ['all','early','late']:
                                condir = os.path.join(rundir,con + '.feat')
                                
                                log = os.path.join(condir,'report_log.html')

                                f = codecs.open(log,'r')

                                print(_sub,run,con)
                                os.system('fslstats %s -r'%(os.path.join(condir,'stats','cope1')))
                                if not os.path.exists(os.path.join(condir,'stats','cope2.nii.gz')):
                                #       if os.path.exists(condir):os.system('rm -r %s'%(condir))                                
                                #       os.system('echo "feat /home1/05426/ach3377/CodeBase/level_1_scripts/%s_run00%s_%s.fsf" >> missing_1s.txt'%(_sub,run,con))
                                        print(_sub, run, con)

#os.system('chmod u+x missing_1s.txt')
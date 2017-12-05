# This script will generate each subjects design.fsf, but does not run it.
# It depends on your system how will launch feat

#Exectute any of these via 'feat design_Sub000_run000.fsf' 

import os
from glob import glob

# Set this to the directory all of the sub### directories live in
data_dir = '/Users/ach3377/GoogleDrive/FC_FMRI_DATA'
run = 4

# Set this to the directory where you'll dump all the fsf files
# May want to make it a separate directory, because you can delete them all o
#   once Feat runs
fsfdir='%s/fsfs'%(data_dir)

# Get all the paths!  Note, this won't do anything special to omit bad subjects
subdirs = glob("%s/Sub[0-9][0-9][0-9]/bold/day[1-2]/run00[%s]"%(data_dir,run))

for dir in list(subdirs):
  splitdir = dir.split('/')
  # YOU WILL NEED TO EDIT THIS TO GRAB sub001
  SUBJ = splitdir[5]
  #  YOU WILL ALSO NEED TO EDIT THIS TO GRAB THE PART WITH THE run000
  PHASE = splitdir[8]

  print(SUBJ + ', ' + PHASE)

  
  #maybe need this for day 2 but for now all runs on day 1 have the same number of TRs
  #ntime = os.popen('fslnvols %s/bold.nii.gz'%(dir)).read().rstrip()

  #put NVOLS in here if needed
  replacements = {'SUBJ':SUBJ, 'PHASE':PHASE}
  with open("%s/template_lev1_run%s_nl.fsf"%(fsfdir,run)) as infile: 
    with open("%s/lev1/design_%s_%s.fsf"%(fsfdir, SUBJ, PHASE), 'w') as outfile:
        for line in infile:
          # Note, since the video, I've changed "iteritems" to "items"
          # to make the following work on more versions of python
          #  (python 3 no longer has iteritems())  
          for src, target in replacements.items():
            line = line.replace(src, target)
          outfile.write(line)
  
#SUBJ = Sub001
#PHASE = run000
#NVOLS = TRs

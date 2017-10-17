import os
import glob


# We will start with the registration png files
outfile = "/Users/ach3377/GoogleDrive/FC_FMRI_DATA/lev1_QA.html"
#outfile = '/mnt/c/Users/ACH/Google Drive/FC_FMRI_DATA/lev1_QA.html'
os.system("rm %s"%(outfile))

all_feats = glob.glob('/Users/ach3377/GoogleDrive/FC_FMRI_DATA/Sub00[1-9]/model/run00*.feat/')
#all_feats = glob.glob('/mnt/c/Users/ACH/Google Drive/FC_FMRI_DATA/Sub00[1-9]/model/run00*.feat/')
f = open(outfile, "w")
for file in list(all_feats):
  f.write("<p>============================================")
  f.write("<p>%s"%(file))
  f.write("<IMG SRC=\"%s/design.png\">"%(file))
  f.write("<IMG SRC=\"%s/design_cov.png\" >"%(file))
  f.write("<IMG SRC=\"%s/mc/disp.png\">"%(file))
  f.write("<IMG SRC=\"%s/mc/trans.png\" >"%(file))
  f.write("<p><IMG SRC=\"%s/reg/example_func2highres.png\" WIDTH=1200>"%(file))
  f.write("<p><IMG SRC=\"%s/reg/example_func2standard.png\" WIDTH=1200>"%(file))
  f.write("<p><IMG SRC=\"%s/reg/highres2standard.png\" WIDTH=1200>"%(file))
f.close()

require(lme4)
require(MASS)
require(car)
require(ez)

ER_dir <- '/Users/ach3377/Db_lpl/STUDY/FearCon/group_ER'#working dir
dfs_dir <- paste0(ER_dir,'/voxel_dfs')#point to df dir
output_dir <- paste0(ER_dir,'/r_stats')#where to save

nvox <- 77779 #hardcoded

#initialize output - have to actually run these once to know the order of columsn - pretty sure its alphabetical
no_mem_effects <- c("group","encode","trial_type","group:encode","group:trial_type","encode:trial_type","group:encode:trial_type")
no_mem_mat = matrix(nrow=nvox,ncol=length(no_mem_effects))


for (i in 0:77778){
  
  
  vdf <- read.csv(paste0(dfs_dir,sprintf('/voxel_%s.csv',i))) #read in the voxel dataframe
  
  #NO MEM - we don't have to exlude subs
  no_mem_ag <- aggregate(vdf$rsa,by=list(vdf$subject,vdf$encode,vdf$trial_type),mean)#aggregate down to cell means
  colnames(no_mem_ag) <- c('subject','encode','trial_type','rsa')  #rename the columns for clarity
  no_mem_ag['group'] <- ifelse(no_mem_ag$subject < 100,1,2)#recreate the group column
  for (q in c('subject','group')){no_mem_ag[[q]] <- as.factor(no_mem_ag[[q]])}  #factorize things that need it
  
  #run the ANOVA
  no_mem_res <- ezANOVA(data=no_mem_ag,dv=.(rsa),wid=.(subject),within=.(encode,trial_type),between=.(group),type=3)
  rowi <- i+1 #correct for pythonic indexing
  no_mem_mat[rowi,] <- no_mem_res$ANOVA$p #save the results
  
  #Include memory - we have to exclude some subs
  
  
  if (i%%250 == 0){print(i)}#give us some readout of progress
}
no_mem_df = data.frame(no_mem_mat)
colnames(no_mem_df) <- no_mem_effects
write.csv(no_mem_df,paste0(output_dir,'/no_mem_ANOVA.csv'))




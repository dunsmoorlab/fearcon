make_regs_sels <- function(subj=NULL){
  require(MASS)
  
  SUBJ = paste0('Sub00',subj)
  
  MVPA_dir = paste0('/Users/ach3377/GoogleDrive/FC_FMRI_DATA/',SUBJ,'/model/MVPA')
  
  label_dir = paste0(MVPA_dir,'/labels')
  toolbox = paste0(MVPA_dir,'/toolbox')
  
#localizer xval 
#####################################################
  #make run selectors first (easy)
  selectors <- matrix(nrow = 1, ncol = 480)
  selectors[,1:240] <- 1
  selectors[,241:480] <- 2
  write.matrix(selectors,file = paste0(toolbox,'/2run_localizer_sels'))
  
  #load in the numpy TR labels for each run
  loc1 <- read.csv(paste0(toolbox,'/localizer_1_labels_RAW.csv'),header=FALSE)
  loc2 <- read.csv(paste0(toolbox,'/localizer_2_labels_RAW.csv'),header=FALSE)
  
  #add the right number of TRs to the end of the localizer runs (6; functional runs have 240 TRs)
  loc1[234:240,] <- 'rest'
  loc2[235:240,] <- 'rest'
  
  #combine the 2 localizer runs
  loc_cats <- rbind(loc1,loc2)
  
  loc_cat2num <- matrix(nrow=dim(loc_cats)[1], ncol=1)
  #for internal reference, the rows of the reg mat are in descending order: ANIMAL, TOOL, SCENE, SCRAMBLED, REST
  #convert the category strings to be numbers(animal:1, tool:2, scene:3, scrambled:4, rest:5)
  for (q in 1:dim(loc_cats)[1]){
    
    cat = loc_cats[q,1]
    
    if (cat == 'animal'){
      loc_cat2num[q,1] <- 1}
    
    else if (cat == 'tool'){
      loc_cat2num[q,1] <- 2}
    
    else if (cat == 'indoor' | cat == 'outdoor'){
      loc_cat2num[q,1] <- 3}
    
    else if (cat == 'scrambled'){
      loc_cat2num[q,1] <- 4}
    
    else if (cat == 'rest'){
      loc_cat2num[q,1] <- 5}
    
  }
  
  #collect the categories in easy mode
  real_cats <- as.integer(loc_cat2num[,1])
  
  #initialize the output matrix
  loc_regs <- matrix(nrow=4, ncol=480)
  #set everything to 0
  loc_regs[is.na(loc_regs)] <- 0
  
  #for internal reference, the rows of the reg mat are in descending order: ANIMAL, TOOL, SCENE, SCRAMBLED, REST
  #populate the regs mat with 1s for the correct category at each TR
  for (i in 1:length(real_cats)){
    if (real_cats[i] == 5){}
    else {
      loc_regs[real_cats[i],i] <- 1
      }
  }
  
  #save them!
  write.matrix(loc_regs,file = paste0(toolbox,'/4cat_localizer_regs'))
  
##################################################  
  #make binary scene regs, where there are just 2 conditions, scene or not scene. Also model rest
  
  loc_cat2scene <- matrix(nrow=dim(loc_cats)[1], ncol=1)
  #for internal reference, the rows of the reg mat are in descending order: ANIMAL, TOOL, SCENE, SCRAMBLED, REST
  #convert the category strings to be numbers(scene:1, not scene2, rest:3)
  for (q in 1:dim(loc_cats)[1]){
    
    cat = loc_cats[q,1]
    
    if (cat == 'indoor' | cat == 'outdoor'){
      loc_cat2scene[q,1] <- 1}
    
    else if (cat == 'rest'){
      loc_cat2scene[q,1] <- 3}
    
    else {
      loc_cat2scene[q,1] <- 2
    }
    
  }
  #collect the categories in easy mode
  scene_cats <- as.integer(loc_cat2scene[,1])
  
  #initialize the output matrix
  loc_scene_regs <- matrix(nrow=2, ncol=480)
  #set everything to 0
  loc_scene_regs[is.na(loc_scene_regs)] <- 0
  
  #for internal reference, the rows of the reg mat are in descending order: ANIMAL, TOOL, SCENE, SCRAMBLED, REST
  #populate the regs mat with 1s for the correct category at each TR
  for (i in 1:length(scene_cats)){
    if (scene_cats[i] == 3){}
    else {
      loc_scene_regs[scene_cats[i],i] <- 1
    }
  }
  
  #save them!
  write.matrix(loc_scene_regs,file = paste0(toolbox,'/scene_localizer_regs'))
  
  
  
  #prelim mental context analysis
  #################################################
  
  #Day1 + LOCALIZER first
  #Day1 has 259 TRs
  sel_base <- matrix(nrow = 1, ncol = 739)
  sel_base[,1:259] <- 1
  sel_base[,260:739] <- 2
  
  write.matrix(sel_base, file = paste0(toolbox,'/day1_localizer_sels'))
  
  sel_base_proc <- matrix(nrow = 1, ncol = 739)
  sel_base_proc[,1:259] <- 1
  sel_base_proc[,260:499] <- 2
  sel_base_proc[,499:739] <- 3
  
  write.matrix(sel_base_proc, file = paste0(toolbox,'/proc_day1_localizer_sels'))
  
  #so for this analysis, don't actually care about classifying stimuli, just want general scene activation
  #We can hard code regs for each phase with every TR set to scene
  base_regs <- matrix(nrow=4, ncol = 259)
  base_regs[is.na(base_regs)] <- 0
  base_regs[3,] <- 1
  
  base_regs <- cbind(base_regs,loc_regs)
  
  write.matrix(base_regs,file = paste0(toolbox,'/day1_4cat_regs'))
  
  #Now again but just for scene/not scene
  base_scene_regs <- matrix(nrow=2, ncol = 259)
  base_scene_regs[is.na(base_scene_regs)] <- 0
  base_scene_regs[1,] <- 1
  
  base_scene_regs <- cbind(base_scene_regs,loc_scene_regs)
  
  write.matrix(base_scene_regs,file = paste0(toolbox,'/day1_scene_regs'))
##########################################################################  
  #now for extinction recall
  #extinction recall has 135 TRs
  sel_er <- matrix(nrow = 1, ncol = 615)
  sel_er[,1:135] <- 1
  sel_er[,136:615] <- 2
  
  write.matrix(sel_er, file = paste0(toolbox,'/extinction_recall_localizer_sels'))
  
  sel_er_proc <- matrix(nrow = 1, ncol = 615)
  sel_er_proc[,1:135] <- 1
  sel_er_proc[,136:375] <- 2
  sel_er_proc[,376:615] <- 3
  
  write.matrix(sel_er_proc, file = paste0(toolbox,'/proc_extinction_recall_localizer_sels'))
  
  
  #so for this analysis, don't actually care about classifying stimuli, just want general scene activation
  #We can hard code regs for each phase with every TR set to scene
  er_regs <- matrix(nrow=4, ncol = 135)
  er_regs[is.na(er_regs)] <- 0
  er_regs[3,] <- 1
  
  er_regs <- cbind(er_regs,loc_regs)
  
  write.matrix(er_regs,file = paste0(toolbox,'/extinction_recall_4cat_regs'))
  
  #Now again but just for scene/not scene
  er_scene_regs <- matrix(nrow=2, ncol = 135)
  er_scene_regs[is.na(er_scene_regs)] <- 0
  er_scene_regs[1,] <- 1
  
  er_scene_regs <- cbind(er_scene_regs,loc_scene_regs)
  
  write.matrix(er_scene_regs,file = paste0(toolbox,'/extinction_recall_scene_regs'))
  
}

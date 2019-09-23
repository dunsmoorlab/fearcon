require(ez)

TR <- read.csv('/Users/ach3377/Db_lpl/STUDY/FearCon/graphing/signal_change/group_tr_df.csv')
for (q in c('subject','trial','tr')){TR[[q]] <- as.factor(TR[[q]])}
#str(TR)
#TR <- subset(TR, tr %in% c(-2,-1,0))
#Between groups
tr_res <- ezANOVA(data=TR,dv=.(evidence),wid=.(subject),within=.(tr),between=.(response,group),type=2)
tr_res$ANOVA

#Control only
cTR <- subset(TR, group %in% 'control')
c_res <- ezANOVA(data=cTR,dv=.(evidence),wid=.(subject),within=.(tr),between=.(response),type=2)
c_res$ANOVA
cTR_ag <- aggregate(cTR$evidence,by=list(cTR$subject),mean)




#PTSD only
pTR <- subset(TR, group %in% 'ptsd')
p_res <- ezANOVA(data=pTR,dv=.(evidence),wid=.(subject),within=.(tr),between=.(response),type=2)
p_res$ANOVA

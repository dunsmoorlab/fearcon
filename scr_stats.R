require(ez)

SCR <- read.csv('/Users/ach3377/Db_lpl/STUDY/FearCon/graphing/SCR/SCR_rm_na.csv')

SCR$quarter <- sapply(SCR$quarter,factor)
SCR$subject <- sapply(SCR$subject,factor)

full <- ezANOVA(data=SCR,dv=.(scr),wid=.(subject),within=.(condition,quarter),between=.(group),type=2)
full$ANOVA

#####################
comp = read.csv('/Users/ach3377/Db_lpl/STUDY/FearCon/graphing/SCR/fc_scr_comp.csv')

comp$quarter <- sapply(comp$quarter,factor)
comp$subject <- sapply(comp$subject,factor)

diff <- ezANOVA(data=SCR,dv=.(scr),wid=.(subject),within=.(quarter),between=.(group),type=2)
diff$ANOVA

c_ev <- read.csv('C:\\Users\\ACH\\Desktop\\control_ev.csv')
#g.bt <- aov(cr ~ (condition * phase * group) + Error(subject/(condition*phase)), data=cr_dat)

c.aov <- aov(evidence ~ (trial * condition) + Error(subject/(condition*trial)), data=c_ev)
#g_ev <- read.csv('C:\\Users\\ACH\\Dropbox (LewPeaLab)\\STUDY\\FearCon\\Group_GLM\\tr-1_0_pm_mvpa_ev.csv')
#g_ev <- read.csv('C:\\Users\\ACH\\Dropbox (LewPeaLab)\\STUDY\\FearCon\\Group_GLM\\tr_0_pm_mvpa_ev.csv')
g_ev <- read.csv('C:\\Users\\ACH\\Dropbox (LewPeaLab)\\STUDY\\FearCon\\Group_GLM\\tr_0-1_pm_mvpa_ev.csv')
g_ev['group'] <- ''
g_ev$group[0:480] <- 'control'
g_ev$group[481:960] <- 'ptsd'

g.aov <- aov(evidence ~ (condition * trial * group) + Error(subject/(condition*trial)), data=g_ev)
summary(g.aov)

early <- subset(g_ev, trial %in% c(1,2,3,4))
ecsp <- subset(early, condition %in% c('CS+'))

ecsp.aov <- aov(evidence ~ (trial * group) + Error(subject/(trial)), data=ecsp)
ecsp.aov <- aov(evidence ~ group + Error(subject/trial), data=ecsp)

summary(ecsp.aov)



cr.p <- subset(group_mem, phase %in% c('false_alarm'))

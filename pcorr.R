require(ggplot2)
require(ppcor)
require(hmisc)
require(mediation)
require(lavaan)
#############################################
#ev2 = read.csv('/Users/ach3377/Db_lpl/STUDY/FearCon/graphing/signal_change/mvpa_ev.csv')
r_psc = read.csv('/Users/ach3377/Db_lpl/STUDY/FearCon/graphing/signal_change/beta_values.csv')
#e_psc = read.csv('/Users/ach3377/Db_lpl/STUDY/FearCon/graphing/signal_change/run003_beta_values.csv')

ev = read.csv('/Users/ach3377/Db_lpl/STUDY/FearCon/graphing/signal_change/mvpa_ev.csv')
#ev = read.csv('C:\\Users\\ACH\\Dropbox (LewPeaLab)\\STUDY\\FearCon\\graphing\\signal_change\\mvpa_ev.csv')
#scr = read.csv('C:\\Users\\ACH\\Dropbox (LewPeaLab)\\STUDY\\FearCon\\graphing\\SCR\\c_e_rnw_scr.csv')
scr = read.csv('/Users/ach3377/Db_lpl/STUDY/FearCon/graphing/SCR/c_e_rnw_scr.csv')
scr = rbind(scr,scr)

#r_psc = read.csv('C:\\Users\\ACH\\Dropbox (LewPeaLab)\\STUDY\\FearCon\\graphing\\signal_change\\beta_values.csv')

cmap = which(ev$Group == 'Control')
pmap = which(ev$Group == 'PTSD')

amyg = cbind(r_psc[which(r_psc$roi == 'amygdala_beta'),],e_psc[which(e_psc$roi == 'amygdala_beta'),], ev)

hpc = cbind(r_psc[which(r_psc$roi == 'hippocampus_beta'),],e_psc[which(e_psc$roi == 'hippocampus_beta'),], ev)

mOFC = cbind(r_psc[which(r_psc$roi == 'mOFC_beta'),],e_psc[which(e_psc$roi == 'mOFC_beta'),], ev)

vmPFC = cbind(r_psc[which(r_psc$roi == 'vmPFC_beta'),],e_psc[which(e_psc$roi == 'vmPFC_beta'),], ev)

dACC = cbind(r_psc[which(r_psc$roi == 'dACC_beta'),],e_psc[which(e_psc$roi == 'dACC_beta'),], ev)

SvmPFC = cbind(r_psc[which(r_psc$roi == 'mOFC_beta'),],scr,ev)
###########################################

group=cmap
ROI = mOFC
data1 <- data.frame(
                      renewal=ROI[group,'early_CSp_CSm'],
                      extinction=ROI[group,'CSp_CSm'],
                      evidence=ROI[group,'ev']
)
pcor(data1)
pcres <- pcor.test(data1$extinction,data1$renewal,data1$evidence,method="pearson")
print(pcres)



model.context <- 'evidence ~ extinction
                  renewal ~ evidence + extinction'
model.dat = data1
context.fit <- sem(model.context, data=model.dat)
summary(context.fit)

model.context2 <- 'evidence ~ a*extinction
                   renewal ~ b*evidence + c*extinction

                   indirect := a*b
                   direct   := c
                   total    := c + (a*b)'
context.fit2 <- sem(model.context2, data=model.dat)
summary(context.fit2)
################################
group=cmap
ROI = SvmPFC
sdata <- data.frame(
  renewal=ROI[group,'early_CSp_CSm'],
  scr=ROI[group,'scr'],
  evidence=ROI[group,'ev']
)

med_fit <- lm(evidence ~ renewal,data=sdata)
out_fit <- lm(scr ~ evidence + renewal,data=sdata)

med.out <- mediate(med_fit, out_fit, treat = "renewal", mediator = "evidence", boot=TRUE, sims = 1000)
summary(med.out)
plot(med.out)




####################################
group=pmap
ROI1 = mOFC
ROI2 = amyg
rdat <- data.frame(
  roi1=ROI1[group,'early_CSp_CSm'],
  roi2=ROI2[group,'early_CSp_CSm'],
  evidence=ev[group,'ev']
)
model.r <- 'evidence ~ a*roi1
            roi2 ~ b*evidence + c*roi1

            indirect := a*b
            direct   := c
            total    := c + (a*b)'
fit.r <- sem(model.r, data=rdat)
summary(fit.r)

med_fit <- lm(evidence ~ roi1,data=rdat)
out_fit <- lm(roi2 ~ evidence + roi1,data=rdat)

med.out <- mediate(med_fit, out_fit, treat = "roi1", mediator = "evidence", boot=TRUE, sims = 10000)
summary(med.out)
plot(med.out)

###################################

group <- pmap
netdat <- data.frame(
    vmPFC=mOFC[group,'early_CSp_CSm'],
    amygdala=amyg[group,'early_CSp_CSm'],
    hippocampus=hpc[group,'early_CSp_CSm']
)
pcor(netdat)
pcres <- pcor.test(data$extinction,data$renewal,data$evidence,method="pearson")


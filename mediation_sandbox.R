require(MBESS)
require(lavaan)

cv = read.csv('/Users/ach3377/Db_lpl/STUDY/FearCon/graphing/mediation/control_vmPFC.csv')
med1 <- MBESS::mediation(x=cv$ev,mediator=cv$extinction,dv=cv$renewal,bootstrap = T, B = 500)

rmod <- read.csv('/Users/ach3377/Db_lpl/STUDY/FearCon/graphing/mediation/renewal_model.csv')
cmap <- which(rmod$group == 'control')
pmap <- which(rmod$group == 'ptsd')

crmod <- rmod[cmap,]
cor.test(crmod$ev,crmod$vmPFC)

prmod <- rmod[pmap,]
cor.test(prmod$ev,prmod$amyg)

cmed.model <- 'amyg ~ b1*vmPFC + b2*HC + c*ev
               vmPFC ~ a1*ev
               HC ~ a2*ev
               indirect1 := a1*b1
               indirect2 := a2*b2
               total := c + (a1*b1) + (a2*b2)
               vmPFC ~~ HC
               contrast := indirect2 + indirect1'

cmed.model_con <- 'amyg ~ b1*vmPFC + b2*HC + c*ev
               vmPFC ~ a1*ev
               HC ~ a2*ev
               indirect1 := a1*b1
               indirect2 := a2*b2
               total := c + (a1*b1) + (a2*b2)
               vmPFC ~~ HC
               indirect1 == indirect2'

cmed.fit <- sem(model=cmed.model,data=crmod, se='bootstrap', bootstrap=1000)
cmed.fit_constr <- sem(model=cmed.model_con,data=crmod, se='bootstrap', bootstrap=5000)
anova(cmed.fit, cmed.fit_constr)
summary(cmed.fit,fit.measures=TRUE, standardize=TRUE, rsquare=TRUE,estimates = TRUE, ci = TRUE)
##################################
ser.model <- 'HC ~ a1 * ev
              vmPFC ~ a2 * ev + d21 * HC
              amyg ~ cp * ev + b1 * HC + b2 * vmPFC
              indirect := a1 * d21 * b2'
ser.fit <- lavaan::sem(model=ser.model,data=crmod,se='bootstrap',bootstrap=1000)


summary(ser.fit,fit.measures=TRUE, standardize=TRUE, rsquare=TRUE,estimates = TRUE, ci = TRUE)
lavaan::parameterEstimates(ser.fit,boot.ci.type = 'bca.simple')

q = lm('vmPFC ~ ev*bgroup',data=rmod)
###################################
mod.model <- 'amyg ~ b1*vmPFC + b2*HC + c*ev
              vmPFC ~ a1*ev + a1w*evg
              HC ~ a2*ev + a2w*evg
              indirect1 := a1*b1
              indirect2 := a2*b2
              total := c + (a1*b1) + (a2*b2)
              vmPFC ~~ HC
              contrast := indirect2 + indirect1'
###########################

med_fit <- lm(extinction ~ ev,data=cv)
out_fit <- lm(renewal ~ ev + extinction,data=cv)

med.out <- mediation::mediate(med_fit, out_fit, treat = "ev", mediator = "extinction", boot=TRUE, sims = 10000)
summary(med.out)
plot(med.out)


#############################

med.model <- 'extinction ~ a*ev
              renewal ~ b*extinction + c*ev

              indirect := a*b
              direct   := c
              total    := c + (a*b)'
med.fit <- lavaan::sem(med.model, data=cv)
summary(med.fit, standardize=T, fit.measures=T, rsq=T)

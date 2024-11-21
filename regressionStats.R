## Aviary Stats Analysis 

library(lme4)
library(DHARMa)
library(MuMIn)
library(lmerTest)

males.df <- read.csv("~/Documents/Scripts/AviaryAnalysis/male_df.csv")
females.df <- read.csv("~/Documents/Scripts/AviaryAnalysis/female_df.csv")
cohesion.df <- read.csv("~/Documents/Scripts/AviaryAnalysis/cohesion_df.csv")
prediction.df <- read.csv("~/Documents/Scripts/AviaryAnalysis/prediction_df.csv")
aviary.df <- read.csv("~/Documents/Scripts/AviaryAnalysis/aviary_df.csv")
season.df <- read.csv("~/Documents/Scripts/AviaryAnalysis/season_df.csv")
production.df <- read.csv("~/Documents/Scripts/AviaryAnalysis/production_df.csv")
all_females.df <- read.csv("~/Documents/Scripts/AviaryAnalysis/all_females_df.csv")

### Note: Egg counts don't factor in the "cutoffs" 
### This doesn't matter for egg score (which takes that into account), 
## but this does matter if you look at the actual eggs. 
## Aviaries 4 and 7 were manipulated prior to egg production
## Aviaries 0,2 and 10,14 had females switch mid egg production, so 
## best to drop those when looking at underlying egg stuff. 

females.df$Aviary <- as.factor(females.df$Aviary)
males.df$Aviary <- as.factor(males.df$Aviary)
cohesion.df$Aviary <- as.factor(cohesion.df$Aviary)
season.df$Aviary <- as.factor(season.df$Aviary)
production.df$Aviary <- as.factor(production.df$Aviary)
all_females.df$Aviary <- as.factor(all_females.df$Aviary)

### Add a couple additional columns I forgot to add in python: 
production.df[,'PropBonded'] <- production.df$nBonded / production.df$nFemales
production.df[,'PropPairsLaying'] <- production.df$nLaying / production.df$nBonded

## Mixed regression for demonstrating cohesion

aviary.df[,'EggsPerFemale'] <- aviary.df$EggCount / aviary.df$nFemales

cohesion.present <- lmer(FemaleSong ~ GroupFemaleSong + (1|MaleID) + (1|Aviary),data=cohesion.df)
summary(cohesion.present)
r.squaredGLMM(cohesion.present)

## Variation in nEggs by aviary
egg.aov <- aov(formula=fEggs~Aviary,data=females.df)
summary(egg.aov)

## Variation in cohesion by aviary
cohesion.aov <- aov(formula=Cohesion ~ Aviary,data=season.df)
summary(cohesion.aov)
rep.season <- rpt(Cohesion ~ (1|Aviary),grname='Aviary',datatype='Gaussian', data=season.df)
print(rep.season)

## We should probably include day as a fixed effect here in case it varies
rep2.season <- rpt(Cohesion ~ Day + (1|Aviary),grname='Aviary',datatype='Gaussian', data=season.df)
print(rep2.season)

## No obvious variation
season.lm <- lmer(Cohesion ~ Day + (1|Aviary),data=season.df)
summary(season.lm)

par(mfrow=c(1,1))
plot(season.df$Day,season.df$Cohesion)

## Simple correlation (one-sided test) between egg score and cohesion: 

res.cohesion <- cor.test(aviary.df$EggScore,aviary.df$Cohesion,alternative='greater')
res.cohesion
## Regression of cohesion vs egg score
cohesion.aviary <- lm(EggScore ~ Cohesion, data=aviary.df)
summary(cohesion.aviary)

simulationOutput <- simulateResiduals(fittedModel = cohesion.aviary, plot = T)
plot(simulationOutput)

par(mfrow = c(2, 2))
plot(cohesion.aviary)

## Remove the two aviaries with prior pairbonds:
aviary.df
aviary.naive <- aviary.df[-c(11,13),]
aviary.naive <- aviary.df[-c(2,11,13),]
aviary.naive$Cohesion

cohesion.aviaryNaive <- lm(EggScore ~ Cohesion, data=aviary.naive)
summary(cohesion.aviaryNaive)

## note this is robust to the specific window: 
cohesion.aviaryNaive2 <- lm(EggScore ~ CohesionWindowed, data=aviary.naive)
summary(cohesion.aviaryNaive2)

res.cohesion3 <- cor.test(aviary.naive$EggScore,aviary.naive$CohesionWindowed)
res.cohesion3

## Back to including all the aviaires, averaging across all possible window sizes you get roughly the same corr
## but now not quite significant at p=0.05. 
res.cohesion2 <- cor.test(aviary.df$EggScore,aviary.df$CohesionWindowed)
res.cohesion2

## We initially calculated the window-averaged value using bootstrapped confidence intervals
## (see jupyter notebook)
## This is because 
## a) our sample size is quite small so it seemed appropriate
## b) when I was initially doing this, python's pearsonr p-value documentation 
####   described it as being a rough estimate, which I didn't love, so bootstrapping
####  seemed appropriate, especially since I was shuffling randomizing window size anyway

## I've since rerun the stats in R in a few different ways and realized that the effect significance is really marginal, 
##   so some ways are just above p=0.05 and others just below, but I am sticking 
##   with what I did first. 
## I think given the explorational nature of this, this is ok, but it's obviously 
## a caveat here. 
## All of this is just a detail in the paper to say that changing window size for 60s does not eliminate this effect, 

## Log eggs doesn't improve things really. 
#aviary.df[,'LogEggs'] = log(aviary.df$EggScore)

### FYI: 
## This excludes any aviaries where females switch during egg production

aviary.good_ <- production.df[production.df$Countable == 'True',]
res.eggsPer <- aov(EggScore ~ EggsPerParticipating,data=aviary.good)

## Increase in number of eggs by participating females:
res.eggsPerPair <- aov(EggScore ~ EggsPerPair,data=aviary.good_)
summary(res.eggsPerPair)
res.eggsPerLayer <- aov(EggScore ~ EggsPerLayer,data=aviary.good_)
summary(res.eggsPerLayer)

#aviary.good_[,'PropBonded'] <- aviary.good_$nBonded / aviary.good_$nFemales
## Increase in participation:
res.propPairing <- aov(EggScore ~ PropBonded,data=aviary.good_)
summary(res.propPairing)
res.propLaying <- aov(EggScore ~ PropLaying,data=aviary.good_)
summary(res.propLaying)

## Not quite significant here, but obviously this seems like the mechanism 
#aviary.good_[,'PropPairsLaying'] <- aviary.good_$nLaying / aviary.good_$nBonded
res.propPairsLaying <- aov(EggScore ~ PropPairsLaying,data=aviary.good_)
summary(res.propPairsLaying)

## You might be wondering about how many eggs come from non-pairbonded females:
## Quite a lot in some cases. 

table(all_females.df[all_females.df$Pairbonded == 'False','nEggs'])
eggsBreakdown <- all_females.df %>% 
  group_by(Aviary,Pairbonded) %>%
  summarise(singleLadies = sum(nEggs))

## Remember there's some messiness here because songs are cutoff but eggs are not, 
## This only matters in the disrupted aviaries
print(eggsBreakdown)

## Not surprisingly, being pairbonded is very good for egg production
## Pairbond is defined as > 70% of songs (at least 20 songs) from one male
all_females.good <- all_females.df[all_females.df$Disrupted == 'True',]
res.eggsBonds <- lmer(nEggs ~ Pairbonded + (1|Aviary),data=all_females.good)
summary(res.eggsBonds)

## Could it be driven by more unpaired layers? 
res.bothBonds <- lm(EggScore ~ PropSingleLaying + PropPairsLaying,data=aviary.good_)
summary(res.bothBonds)

## So it doesn't seem to be explained by unpaired, which makes sense since those 
## Are a pretty small subset for most aviaries. 
res.unpaired <- aov(EggScore ~ EggsPerSingleLayer,data=aviary.good_)
summary(res.unpaired)
res.unpaired2 <- aov(EggScore ~ PropSingleLaying,data=aviary.good_)
summary(res.unpaired2)

### Does cohesion actually predict this?

## Not really, I mean maybe? Our power is pretty low here. 
res.cohesionBond <- aov(PropLaying ~ Cohesion,data=aviary.good_)
summary(res.cohesionBond)
res.cohesionBond2 <- aov(PropPairsLaying ~ Cohesion,data=aviary.good_)
summary(res.cohesionBond2)
res.cohesionBond3 <- aov(PropBonded ~ Cohesion,data=aviary.good_)
summary(res.cohesionBond3)

## So pairbonding and participating in egg laying is driving egg score
## We can't confirm that cohesion impacts those, so it's certainly possible 
## that some other confound (e.g., n Females) is driving both. 
## nFemales is correlated with both, but the direction of causality is unclear

## What about what from the oft forgotten male perspective? 

## Mean male cohesion does not predict the eggs they get
res.maleCohesion <- lmer(Eggs ~ Cohesion + (1|Aviary),data=males.df)
summary(res.maleCohesion)

## But degree-cohesion (the number of males they're linked to) does
## We're controlling within aviaries here, so things are a bit simpler. 
res.maleDegree <- lmer(Eggs ~ Degree + (1|Aviary),data=males.df)
summary(res.maleDegree)

## Model is a bit underdispersed here, but that isn't actually too concerning 
simulationOutput <- simulateResiduals(fittedModel = res.maleDegree, plot = T)
plot(simulationOutput)

## What does aviary-level cohesion do for similarly connected males:

## I don't love how iffy this feels...call it highly explorational. 
males.good <- drop_na(males.df,"EggScore")
males.good <- males.good[!is.infinite(males.good$EggScore),]
res.cohesionDegree <- lmer(EggScore ~ DegreeScore + (1|Degree),data=males.good)
summary(res.cohesionDegree)



participating.all <- females.df[females.df$fEggs > 0,]
participating.sum <- table(participating.all$Aviary)



participating.good <- participating.sum[-c(1,2,3,4,5,8,11,15)]
aviary.good <- aviary.df[-c(1,2,3,4,12,15),]
aviary.good[,'ParticipatingFemales'] <- participating.good
aviary.good[,'ProportionParticipating'] <- participating.good / aviary.good$nFemales
aviary.good[,'EggsPerParticipating'] <- aviary.good$EggCount / participating.good 

res.eggsPer <- aov(EggScore ~ EggsPerParticipating,data=aviary.good)
summary(res.eggsPer)
cor.test(aviary.good$EggScore,aviary.good$EggsPerParticipating)

res.nFemales <- aov(EggScore ~ ParticipatingFemales,data=aviary.good)
summary(res.nFemales)
cor.test(aviary.good$EggScore,aviary.good$ParticipatingFemales)


### Check whether proportion breed or eggs per breeder predicts egg score

## Proportion laying eggs
res.femalesPer <- aov(EggScore ~ ProportionParticipating,data=aviary.good)
summary(res.femalesPer)
cor.test(aviary.good$EggScore,aviary.good$ProportionParticipating)

cor.test(aviary.good$Cohesion,aviary.good$ParticipatingFemales)
cor.test(aviary.good$Cohesion,aviary.good$EggsPerParticipating)

aviary.df[,'ParticipatingFemales'] <- table(foo$Aviary)
aviary.df[,'EggsPerPartipating'] <- aviary.df$EggCount / aviary.df$ParticipatingFemales

cohesion.log <- lm(LogEggs ~ Cohesion, data = aviary.df)
summary(cohesion.log)
simulationOutput <- simulateResiduals(fittedModel = cohesion.log, plot = T)
plot(simulationOutput)

par(mfrow = c(2, 2))
plot(cohesion.log)

## Transformation doesn't improve things too much
## I think it's safe to say our p-values are imprecise,  
## but given how explorational this is anyway, I'm not convinced that's critical
## Especially since a lot of the problem is due to those two outlier points
## that we didn't realize had slightly different treatments, but removing them 
## Feels more like p-hacking, since it strengthens our results so much. 

## Regressions with countersong
cohesion.countersong <- lm(EggScore ~ Countersong, data=aviary.df)
summary(cohesion.countersong)

cohesion.countersong2 <- lm(EggScore ~ CountersongRatio, data=aviary.df)
summary(cohesion.countersong2)

cohesion.countersong3 <- lm(Cohesion ~ CountersongRatio, data=aviary.df)
summary(cohesion.countersong3)

cohesion.countersong4 <- lm(Cohesion ~ Countersong + nMales, data=aviary.df)
summary(cohesion.countersong4)
## So, countersong doesn't predict egg score, and countersong doesn't predict cohesion

## Regression of cohesion (ignoring countersong) vs egg score
cohesion.noCounter <- lm(EggScore ~ Cohesion_, data=aviary.df)
summary(cohesion.noCounter)

## Annoyingly, removing countersong weakens the effect such that it's not significant, 
## But everything is so marginal here that I don't know that we can conclude much from that

## Checking the n-birds confounds
## It's certainly possible that we are just seening nBirds driving it

## nBirds is the most significant (negative) predictor of EggScore, 
## but they're all pretty similar 
cohesion.nBirds <- lm(EggScore ~ nBirds,data=aviary.df)
summary(cohesion.nBirds)

cohesion.nMales <- lm(EggScore ~ nMales,data=aviary.df)
summary(cohesion.nMales)

cohesion.nFemales <- lm(EggScore ~ nFemales,data=aviary.df)
summary(cohesion.nFemales)

## Cohesion is most strongly (negative) predicted by nMales
cohesion.nBirds2 <- lm(Cohesion ~ nMales,data=aviary.df)
summary(cohesion.nBirds2)

## The two-effect model weakens effect, but it's similar. 
## Df wrecks significance though. 

cohesion.combo <- lm(EggScore ~ Cohesion + nMales,data=aviary.df)
summary(cohesion.combo)

## As a reminder, this is the simple model: 
summary(cohesion.aviary)

## Mixed regressions comparing leaders and followers. 
cohesion.leader <- lmer(GroupFemaleSong ~ LastFemaleSong + (1|MaleID) + (1|Aviary),data=cohesion.df)
summary(cohesion.leader)
r.squaredGLMM(cohesion.leader)

cohesion.follower <- lmer(FemaleSong ~ LastGroupFemaleSong + (1|MaleID) + (1|Aviary),data=cohesion.df)
summary(cohesion.follower)
r.squaredGLMM(cohesion.follower)

### The python script cohesion_regression.py calculates each cohesion, which are stored in 
## leaders_df.csv


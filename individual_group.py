#! /usr/bin/env python

import pandas as pd
import numpy as np

#from sklear.linear_model import LinearRegression as LR
from pymer4.models import Lmer,Lm

male_df = pd.read_csv('./male_df.csv')
female_df = pd.read_csv('./female_df.csv')

male_df = male_df[male_df['Aviary'] != 12]
male_df = male_df[male_df['Aviary'] != 14]
male_df = male_df[male_df['Aviary'] != 4]
male_df = male_df[male_df['Aviary'] != 7]
## Calculate mean degree

# WTF? 

# get proportion of males pairbonded I guess?
#model1 = LR().fit(male_df['Degree'],male_df['nPairbonds'])

## Clean some weird division values

if True:

    #male_df['nPairbonds'][male_df['nPairbonds'] >= 1] = 1


    male_df = male_df[male_df['EggScore'] > 0]
    model = Lmer("Eggs ~ Degree + nSongs + (1|Aviary)",data=male_df,family='gaussian')
    print(model.fit())

    model = Lmer("nPairbonds ~ Degree + nSongs + (1|Aviary)",data=male_df,family='gaussian')
    print(model.fit())

    male_df['EggScore'].replace([np.inf, -np.inf], np.nan, inplace=True)

## Drop nan values, I couldn't do this before, because non pairbonded males have np.nan egg score
    male_df = male_df.dropna()
    model = Lmer("EggScore ~ Cohesion + (1|Degree)",data=male_df,family='gaussian')
    print(model.fit())
    model0 = Lmer("EggScore ~ Degree + nSongs + (1|Aviary)",data=male_df,family='gaussian')
    print(model0.fit())

    if False:
        male_df = male_df[male_df['EggScore'] > 0]

        male_df['LogSongs'] = np.log(male_df['nSongs'])
        male_df['LogScore'] = np.log(male_df['EggScore'])
        model = Lmer("LogScore ~ Degree + LogSongs + (1|Aviary)",data=male_df,family='gaussian')
        print(model.fit())

        sub_df = male_df[male_df['Eggs'] >= 1]
        model = Lmer("Eggs ~ Degree + LogSongs + (1|Aviary)",data=male_df)
        print(model.fit())


from matplotlib import pyplot as plt
import seaborn as sns

if False:
    fig,ax = plt.subplots()
    ax.hist(male_df['EggScore'])
    ax.hist(np.log(male_df['EggScore']),alpha=.4)

    fig2,ax2 = plt.subplots()
    ax2.scatter(male_df['EggScore'],male_df['Degree'])

else:
    fig,(ax1,ax2) = plt.subplots(2)
    ax1 = sns.regplot(ax=ax1,x="fits",y="EggScore",data=model0.data, fit_reg=True)
    ax2 = sns.regplot(ax=ax2,x="EggScore",y='residuals',data=model0.data,fit_reg=True)

    fig2,ax3 = plt.subplots()
    model0.plot("Degree",ax=ax3)
    ax3.scatter(male_df['Degree'],male_df['EggScore'],color='black',alpha=.3)
    ax3.set_ylim([-1.5,6])

plt.show()

## So, this is sort of superficial, but the best evidence so far suggests that 
# being connected is good. How connected other birds are in the aviary doesn't 
# matter as much for you. Maybe colabs or reviewers will have ideas on how to flesh
# this out, but I'm satisfied for now. 

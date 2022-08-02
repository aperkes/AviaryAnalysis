
#! /usr/bin/env python

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns

from cohesion_funcs import Meta_data,percent_to_females,sliding_bin_history

import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import spearmanr,pearsonr

from pymer4.models import Lmer

"""
## This needs a working pymer4 installation
# I had a hard time installing with conda as they suggest, but got it to work using the following order: 
conda create --name pymer4
conda activate pymer4
conda install rpy2=2.9.4
pip install pymer4
conda install -c conda-forge r-lmertest
python -c 'from pymer4.test_install import test_install; test_install()'
"""
## Read in saved data (built by jupyter notebook
historys = np.load('./historys.pkl.npy',allow_pickle=True)
metas = np.load('./metas.pkl.npy',allow_pickle=True)
sorteds = np.load('./sorteds.pkl.npy',allow_pickle=True)

## Build df of group and individual % song to females
columns = ['Bin','MaleID','Aviary','FemaleSong','GroupFemaleSong','LastFemaleSong','LastGroupFemaleSong']

win = 100
data_list = []
for a in range(len(metas)):
    n_males = metas[a].n_males
    n_females = metas[a].n_females
## Get the ratio to females and the original history bins, since I'll need the times
    f_ratio,[f_songs,all_songs] = percent_to_females(historys[a],metas[a],sorteds[a],window=win)
    history_bins,[history_rate_bins,ts,window_indices] = sliding_bin_history(sorteds[a],historys[a],metas[a],window=win)
    f_ratio = np.round(f_ratio,4)
## Calculate the 'other males' history bins and f_ratios
    for b in range(len(history_bins)):
        np.fill_diagonal(history_bins[b],0)
## Note that this is slight different, in that I'm summing all songs, rather than by males
## Trying summing by males to balance similar to cohesion
    for m in range(n_males):
        if np.sum(f_songs[:,m]) < 10:
            continue
        other_bins = history_bins[:,n_females:][:,np.arange(n_males) != m]

        if True: ## optionally prune off low singing males
            f_songs_sub = f_songs[:,np.arange(n_males) != m]
            other_bins = other_bins[:,np.sum(f_songs_sub,0) >= 10]

        f_songs_other_ = np.sum(other_bins[:,:,:n_females],axis=2)
        all_songs_other_ = np.sum(other_bins,axis=2)
        f_ratio_other_ = f_songs_other_/all_songs_other_

        f_songs_other = np.sum(other_bins[:,:,:n_females],axis=(2,1))
        all_songs_other = np.sum(other_bins,axis=(2,1))

        f_ratio_other = np.round(f_songs_other / all_songs_other,4)
        f_ratio_other_ = np.nanmean(f_ratio_other_,axis=1)
        if False:
            f_ratio_other = f_ratio_other_

## For each bin, make the line to be stored in the df
        for b in range(len(f_ratio)):
            male_f = f_ratio[b,m]
            other_f = f_ratio_other[b]

## If possible add previous bin data
            if np.abs(ts[b] - ts[b-1]) < 2*win: # check if the prior bin is relevent
                last_f = f_ratio[b-1,m]
                last_f_other = f_ratio_other[b-1] 
            else:
                last_f = np.nan
                last_f_other = np.nan
            data_list.append([b,metas[a].m_ids[m],a,male_f,other_f,last_f,last_f_other])
print('done!')            
df = pd.DataFrame(data_list,columns=columns)

## Reanlysis of cohesion using regression approach

# plot the main points
if False:
    fig,ax = plt.subplots()
    ax.scatter(df['GroupFemaleSong'],df['FemaleSong'],alpha=.2)
    plt.show()

sub_df = df.dropna()

## Both linear and binomial are significant, does residuals work for linear? 
if False:
    family='binomial'
    logistic=True
else:
    family='gaussian'
    logistic=False

model = Lmer("FemaleSong ~ GroupFemaleSong + (1+GroupFemaleSong|MaleID) + (1+GroupFemaleSong|Aviary)",data=sub_df,family=family)
print(model.fit())

## Bring in (or calculate) egg scores
egg_scores = [1.67838725, 0.7053763,  2.13952887, 0.86327322, np.nan, 0.56930228,
 0.82407624, np.nan, 0.88328062, 1.32651336, 0.95876593, 0.83819979,
 2.17645134, 1.23928377, 1.78822018, 1.24189453, 1.43569586, 0.61282845,
 1.05535107]
egg_scores = np.array(egg_scores)
#print('cohesion scores:',cohesion_scores)

modelc = Lmer("FemaleSong ~ (1+GroupFemaleSong|Aviary)",data=sub_df,family=family)
print(modelc.fit())
cohesion_scores = modelc.ranef['GroupFemaleSong']
## Compare to egg scores
print('including outliers:',pearsonr(cohesion_scores[~np.isnan(egg_scores)],egg_scores[~np.isnan(egg_scores)]))
if True:
    egg_scores[12] = np.nan
    egg_scores[14] = np.nan
print('excluding outliers:',pearsonr(cohesion_scores[~np.isnan(egg_scores)],egg_scores[~np.isnan(egg_scores)]))

# Out of curiousity, how does this cohesion measure relate to the correlation meausre: 
old_corrs = [0.2490299580487993, 0.03968002047721218, 0.17212216381734105, 0.07116904660078813, 0.09226579983769621, 0.0967354621272926, 0.09220313208486237, 0.11285284312675094, 0.1034829266120365, 0.13496520240543264, 0.11436237712035087, 0.08159966063997319, 0.0841179451107221, 0.09677698719438958, 0.08542174904029849, 0.17417702687471315, 0.23753171138749476, 0.12123793792560783, 0.09210035632413281]

print('new vs old:',pearsonr(cohesion_scores,old_corrs))

## The above method is not significant, unless we exclude the outliers, and I don't understand it well enough to really know how much that means 

## In any case, moving on.

## Calculate individual cohesion regression scores

# Does individual regression score relate to egg score? I'm going to put a pin in that for now.

## Compare to individual egg scores

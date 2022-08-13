
#! /usr/bin/env python

# Function to test what is driving these cohesive patterns

# Import some things:
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns

from cohesion_funcs import Meta_data,percent_to_females,sliding_bin_history

## Read in saved data (built by jupyter notebook
historys = np.load('./historys.pkl.npy',allow_pickle=True)
metas = np.load('./metas.pkl.npy',allow_pickle=True)
sorteds = np.load('./sorteds.pkl.npy',allow_pickle=True)

# So there are basically 3 options:
# - Leaders: a few individuals drive the behavior of the group
# - Emergence: the weak interactions of individual behavior shapes the group
# - Hidden: Shared internal or external forces are driving behavior


## Build df of group and individual % song to females
columns = ['Bin','MaleID','Aviary','FemaleSong','GroupFemaleSong','LastFemaleSong','LastGroupFemaleSong']

win = 30
data_list = []

from sklearn.linear_model import LinearRegression as LR
from sklearn.linear_model import LogisticRegression as LogR
from scipy.optimize import lsq_linear
from constrained_linear_regression import ConstrainedLinearRegression
from scipy.stats import pearsonr

# First, I need an array of male behavior at each time t.
# Not sure if I need the group behavior...

import warnings
warnings.filterwarnings("ignore", message="Mean of empty Slice")

for a in range(len(metas)):
    n_males = metas[a].n_males
    n_females = metas[a].n_females
## Get the ratio to females and the original history bins, since I'll need the times
    f_ratio,[f_songs,all_songs] = percent_to_females(historys[a],metas[a],sorteds[a],window=win)
    history_bins,[history_rate_bins,ts,window_indices] = sliding_bin_history(sorteds[a],historys[a],metas[a],window=win)
    f_ratio = np.round(f_ratio,4)
    f_ratio = np.array(f_ratio)
    f_songs = np.array(f_songs)
    X = f_songs[:-1]
    Y = f_songs[1:]
    pairwise_corrs = np.zeros([n_males,n_males])
    f_ratio[f_ratio == np.inf] = np.nan
    f_ratio[f_ratio == -np.inf] = np.nan
    group_prediction_scores = np.zeros([n_males])
    for n in range(n_males):
        x = X[:,n]
        x = x.reshape([len(x),1])
        sub_ratio = f_ratio[1:,np.arange(n_males) != n]
        y = np.nanmean(sub_ratio,1)
        #y = np.sum(Y[:,np.arange(n_males) != n],1)
        x = x[~np.isnan(y)]
        y = y[~np.isnan(y)]
        model = LR().fit(x,y)
        #print(model.score(x,y))
        group_prediction_scores[n] = model.score(x,y)
        for m in range(n_males):
            if m < n:
                pairwise_corrs[n,m] = pairwise_corrs[m,n]
            else:
                sub_X = X[:,n]
                sub_Y = Y[:,m]
                clean_X = sub_X[(~np.isnan(sub_X)) & (~np.isnan(sub_Y))]
                clean_Y = sub_Y[(~np.isnan(sub_X)) & (~np.isnan(sub_Y))]
                
                r,p = pearsonr(clean_X,clean_Y)
                pairwise_corrs[n,m] = r
    fig,(ax,ax2,ax3) = plt.subplots(3)
    ax.imshow(pairwise_corrs,vmax=1,vmin=-1,cmap='bwr')        
    auto_corrs = np.diagonal(pairwise_corrs)
    np.fill_diagonal(pairwise_corrs,0)
    predictive_power = np.nanmean(np.abs(pairwise_corrs),axis=0)
    predictive_zscore = (predictive_power - np.nanmean(predictive_power))/np.nanstd(predictive_power)
    if False:
        print('correlation scores:',np.sort(predictive_power)[::-1])
        print('z-scores:',np.sort(predictive_zscore)[::-1])
        print('prediction scores:',group_prediction_scores)
    ax2.bar(np.arange(n_males),np.sort(predictive_power)[::-1])
    ax2.set_ylim([0,np.nanmax(predictive_power)])
    ax3.bar(np.arange(n_males),np.sort(group_prediction_scores)[::-1])
    ax3.set_ylim([0,np.nanmax(group_prediction_scores)])
    if False:
        plt.show()
    #print(np.sort(predictive_power))
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
# Compare to individual egg scores

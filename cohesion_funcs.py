#! /usr/bin/env python

import numpy as np
from datetime import datetime,timedelta
from scipy.stats import pearsonr, spearmanr
import itertools

## Functions which are used in the cohesion paper

## Define some time handling functions
def time_to_sec(h_time):
    hms = h_time.split(':')
    secs = int(hms[0]) * 3600 + int(hms[1]) * 60 + int(hms[2])
    return secs

### Returns the day of year from the unix timestamp.
def day_of_year(datestamp_s):
    return datetime.fromtimestamp(datestamp_s).timetuple().tm_yday

def timestamp_to_ymd(timestamp,date_format = '%Y-%m-%d'):
    return datetime.utcfromtimestamp(timestamp).strftime(date_format)

def date_to_sec(date,date_format = 'ymd'):
    if date_format == 'ymd':
        ymd = date.split('-')
        timestamp = datetime.strptime(date,'%y-%m-%d').timestamp()
    elif date_format == 'mdy':
        mdy = date.split('-')
        ymd = [mdy[2],mdy[0],mdy[1]]
        timestamp = datetime.strptime(date,'%m-%d-%y').timestamp()
    elif date_format == 'dmy':
        dmy = date.split('-')
        ymd = [dmy[2],dmy[1],dmy[0]]
        timestamp = datetime.strptime(date,'%d-%m-%y').timestamp()
    elif date_format == 'Ymd':
        ymd = date.split('-')
        timestamp = datetime.strptime(date,'%Y-%m-%d').timestamp()
    return timestamp, ymd

def get_timestamp(aviary_data,t,date_format='ymd'):
    t_secs = time_to_sec(aviary_data[t,3])
    d_secs,_ = date_to_sec(aviary_data[t,5],date_format)
    return d_secs + t_secs

def get_difference(aviary_data,t0,t1,date_format='ymd'):
    difference = get_timestamp(aviary_data,t1,date_format) - get_timestamp(aviary_data,t0,date_format)
    return difference

## Sort of a hack to deal with older data format
def fake_timestamp(blocked_data,t,year):
    blocks = np.unique(blocked_data[:,5].astype(int))
    n_blocks = len(blocks)
    years = 12 * 31 * 24 * 3600 * year
    fake_day = np.argwhere(blocks == int(blocked_data[t,5]))[0][0]
    days = 24 * 3600 * fake_day
    return days + years

## These are just little wrappers that allow for handling arrays with all nans
def nansumwrapper(a, **kwargs):
    if np.isnan(a).all():
        return np.nan
    else:
        return np.nansum(a, **kwargs)

def nanmeanwrapper(a, **kwargs):
    if np.isnan(a).all():
        return np.nan
    else:
        return np.nanmean(a, **kwargs)

## Define this meta data class object. It also perfoms some data parsing to handle dates and such.
class Meta_data:
    def __init__(self,raw_data,name='unknown',date_format='ymd',start_day=None,cutoff=None):
        self.name = name
        self.n_points = len(raw_data)
        self.date_format = date_format

        self.datetime = np.empty([self.n_points],dtype=object)
        self.timestamps = np.empty([self.n_points])


        for n in range(self.n_points):
            self.timestamps[n] = get_timestamp(raw_data, n,self.date_format)
            self.datetime[n] = raw_data[n,5] + ' ' + raw_data[n,3]
        if cutoff != None:
            self.cutoff_dt = datetime.strptime(cutoff,'%y-%m-%d')
            self.cutoff = np.argmax(self.timestamps >= self.cutoff_dt.timestamp())
            self.cutoff_s = (self.cutoff_dt-datetime(1970,1,1)).total_seconds()
        else:
            self.cutoff = self.n_points
            self.cutoff_dt = None
            self.cutoff_s = None
        if start_day != None:
            self.start_dt = datetime.strptime(start_day,'%y-%m-%d')
            self.start = np.argmax(self.timestamps >= self.start_dt.timestamp())
            self.start_s = (self.start_dt-datetime(1970,1,1)).total_seconds()
        else:
            self.start=0
            self.start_dt,self.cutoff_s = None,None
        self.n_points = self.cutoff - self.start
        tmp_ids = np.unique(raw_data[self.start:self.cutoff,:2])
        b_ids = []
        for t in tmp_ids:
            if len(t) == 0:
                continue
            #if 'M' in t or 'F' in t:
            if t[0] == 'M' or t[0] == 'F':
                if t.upper() == 'FEMALE' or t.upper() == 'MALE':
                    continue
                else:
                    b_ids.append(t.upper()) ## There are some lower case that need to be resolved
        #print(b_ids,self.cutoff)
        self.bird_ids = np.unique(b_ids)
        self.n_birds = len(self.bird_ids)
        self.n_males = len(self.bird_ids[np.char.find(self.bird_ids,'M') == 0])
        self.n_females = len(self.bird_ids[np.char.find(self.bird_ids,'F') == 0])
        self.m_ids = self.bird_ids[np.char.find(self.bird_ids,'M') == 0]
        self.f_ids = self.bird_ids[np.char.find(self.bird_ids,'F') == 0]
        self.indices = dict(zip(self.bird_ids,range(len(self.bird_ids))))
        self.dates = np.unique([raw_data[:self.cutoff,5]])

## Pretty self explanatory function that sorts data by date, needs to know the format.
def sort_data(aviary_data, aviary='Unknown',date_format='ymd',start_day=None,cutoff=None):
    time_stamps = np.zeros([len(aviary_data)])
    for n in range(len(aviary_data)):
        time_stamps[n] = get_timestamp(aviary_data,n,date_format)
    sorted_data = aviary_data[time_stamps.argsort()]

    if False: ## You'd think duplicate timestamps were bad, but actually they happen
        sorted_timestamps = time_stamps[time_stamps.argsort()]
        good_times,indices,counts = np.unique(sorted_timestamps,return_counts=True,return_index=True)
        good_indices = indices[counts == 1]
        sorted_data = sorted_data[good_indices]

    sorted_data = sorted_data[sorted_data[:,3] != '00:00:00']
    #print(sorted_data)
    return sorted_data, Meta_data(sorted_data, aviary,date_format,start_day=start_day,cutoff=cutoff)


## Builds history plot of every interaction
def build_history(aviary_data, meta_data):
    interactions = np.zeros([meta_data.n_points, meta_data.n_birds,meta_data.n_birds])
    copulations = np.zeros([meta_data.n_points, meta_data.n_birds, meta_data.n_birds])
    print(meta_data.name,meta_data.start,meta_data.cutoff)
    print(meta_data.n_points)
    for p in range(meta_data.start,meta_data.cutoff):
        i_str = aviary_data[p,0].upper()
        j_str = aviary_data[p,1].upper()
        if i_str in meta_data.indices.keys():
            i = meta_data.indices[i_str]
        else:
            continue
        if j_str in meta_data.indices.keys():
            j = meta_data.indices[j_str]
        else:
            j = i
            #print(aviary_data[p])
        interactions[p-meta_data.start,i,j] = 1
        if aviary_data[p,2] == 'copulation':
            copulations[p-meta_data.start,i,j] = 5
    return interactions, copulations


## This is like build history, but it's cumulative, each point adds to itself
## Additionally, it decays over time. It's good for visualization
## it's pretty neat, play around with it if you want

def overlap_history(aviary_data, meta_data,decay = .99):
    interactions = np.zeros([meta_data.cutoff-meta_data.start, meta_data.n_birds, meta_data.n_birds])
    copulations = np.zeros_like(interactions)
    lag = decay
    for p in range(meta_data.start,meta_data.cutoff):
        i_str = aviary_data[p,0]
        j_str = aviary_data[p,1]
        i = meta_data.indices[i_str]
        if j_str in meta_data.indices.keys():
            j = meta_data.indices[j_str]
        else:
            j = i
        interactions[p,i,j] = 1
        interactions[p,:,:] = interactions[p,:,:] + interactions[p-1,:,:] * lag
        if aviary_data[p,2] == 'copulation':
            #print('Copulation!')
            copulations[p,i,j] = 5
        copulations[p,:,:] = copulations[p,:,:] + copulations[p-1,:,:] ** .99
    return interactions, copulations


## Simple version that just returns a binned version of history
# Returns history_bins,[history_rate_bins,ts,window_indices]
def bin_history(sorted_data,history_data,meta_data,window=100,a_filter = None):
    zero_hour = datetime.strptime('00:00:00','%H:%M:%S')

    if a_filter is not None:
        history_data = np.array(history_data) ## If you don't do this, it changes original
        history_data[a_filter] = 0
    ts, windows = [],[]
    window_indices = []
    count = 0
    history_bins = []
    history_rate_bins = []
    m_count,f_count,u_count = 0,0,0
    bin_start = meta_data.timestamps[meta_data.start]
    i_start = 0
    for t in range(meta_data.start,meta_data.cutoff): ## Don't forget that I'm cutting off some later stuff for now
        dt = datetime.fromtimestamp(meta_data.timestamps[t])
        if dt.time() == zero_hour.time():
            continue
        elif bin_start == 0:
            bin_start = meta_data.timestamps[t-1]
            i_start = t-1 - meta_data.start
        ## If it's been a while, store the previous bin and start a new bin
        #print(t,bin_start,meta_data.timestamps[t] - bin_start,count,m_count,windows)
        
        ## Need to define windows in order to pull out the history bins.
        ## Start_t = bin_start -> 
        if meta_data.timestamps[t] - bin_start >= window:
            if count == 1:  ## How do we account for windows of size 0...could use more info here
                pass
            elif count != 0:  ## What if we do have a long window with no songs?
                history_t = np.sum(history_data[i_start:t-1-meta_data.start],0)

                ts.append(bin_start)
                win_size = meta_data.timestamps[t-1] - bin_start + 1
                if win_size < window:
                    win_size = window
                    pass
                windows.append(win_size)
                window_indices.append([i_start,t-1])
                
                history_rate = history_t / win_size
                history_bins.append(history_t)
                history_rate_bins.append(history_rate)
                if np.sum(history_t) == 0:
                    import pdb
                    pdb.set_trace()
            count = 0
            bin_start = 0 #meta_data.timestamps[t]
        count += 1

        
    ts = np.array(ts)
    windows = np.array(windows)
    history_bins = np.array(history_bins)
    history_rate_bins = np.array(history_rate_bins)

    return history_bins,[history_rate_bins,ts,window_indices]
  
## As above, but I use a sliding bin to (hopefully) avoid edge artifacts
def sliding_bin_history(sorted_data,history_data,meta_data,window=100,a_filter=None):
    zero_hour = datetime.strptime('00:00:00','%H:%M:%S')
    if a_filter is not None:
        history_data = np.array(history_data) ## If you don't do this, it changes original
        history_data[a_filter] = 0
    ts, windows = [],[]
    window_indices = []
    count = 0
    history_bins = []
    history_rate_bins = []
    m_count,f_count,u_count = 0,0,0
    bin_start = meta_data.timestamps[meta_data.start]
    i_start = 0
    
    t = meta_data.start -1
    halfway = False
    #print(meta_data.cutoff,len(meta_data.timestamps))
    while t < meta_data.cutoff-1: ## This is not perfectly efficient, but it's much easier.
        t += 1
        #print(t)
        dt = datetime.fromtimestamp(meta_data.timestamps[t])
        if dt.time() == zero_hour.time():
            continue
        elif bin_start == 0:
            bin_start = meta_data.timestamps[t]
            i_start = t - meta_data.start
        ## If it's been a while, store the previous bin and start a new bin
        #print(t,bin_start,meta_data.timestamps[t] - bin_start,count,m_count,windows)
        
        ## Need to define windows in order to pull out the history bins.
        ## Start_t = bin_start ->
            
        if meta_data.timestamps[t] - bin_start >= window/2 and halfway == False:
            halfway = int(t)
        
        ## NOTE that this brings in a bit of weirdness at the gaps, but I guess that's the nature of smoothing
        if meta_data.timestamps[t] - bin_start >= window:
            #print('making a window at',t,meta_data.timestamps[t])
            if count == 1:  ## How do we account for windows of size 0...could use more info here
                pass
            elif count != 0:  ## What if we do have a long window with no songs?
                history_t = np.sum(history_data[i_start:t-meta_data.start],0)

                if np.sum(history_t) == 0: ## if you have a long window, just keep moving
                    ## findme
                    pass
                    #import pdb
                    #pdb.set_trace()
                else:
                    ts.append(bin_start)
                    win_size = meta_data.timestamps[t-1] - bin_start + 1
                    if win_size < window:
                        win_size = window
                        pass
                    windows.append(win_size)
                    window_indices.append([i_start,t-1])
                    
                    history_rate = history_t / win_size
                    history_bins.append(history_t)
                    history_rate_bins.append(history_rate)
                    #print('storying')
                    if halfway == t:
                        #t -= 1
                        halfway = False
                    else:
                        t = halfway
                        halfway = False
                    
            count = 0
            bin_start = 0 #meta_data.timestamps[t]

        count += 1
        #print(t,halfway,bin_start,meta_data.timestamps[t])
        
    ts = np.array(ts)
    windows = np.array(windows)
    history_bins = np.array(history_bins)
    history_rate_bins = np.array(history_rate_bins)

    return history_bins,[history_rate_bins,ts,window_indices] 
                        
                        
## Shuffle bins while keeping individual male behavior consistent. 
def shuffle_indy_bins(history_bins):
    n_birds = history_bins.shape[1]
    shuffle_bins = np.empty_like(history_bins)
    for s in range(n_birds):
        singer_bins = history_bins[:,s]
        singer_shuffle = np.random.permutation(singer_bins)
        shuffle_bins[:,s] = singer_shuffle
    return shuffle_bins

## Shuffle bins of the same day, to rule out season
def shuffle_day_bins(history_bins,ts,meta):
    all_days = np.array([day_of_year(t) for t in ts])
    unique_days = np.unique(all_days)
    shuffle_bins = np.empty_like(history_bins)
    for d in unique_days:
        day_indices = np.where(all_days == d)[0]
        for m in range(meta.n_males):
            singer_bins_day = history_bins[day_indices,meta.n_females+m,:]
            singer_shuffle = np.random.permutation(singer_bins_day)
            shuffle_bins[day_indices,meta.n_females+m] = singer_shuffle
    shuffled_bins = history_bins
    return shuffle_bins

def shift_indy_bins(history_bins):
    n_birds = history_bins.shape[1]
    shifted_bins = np.empty_like(history_bins)
    n_bins = history_bins.shape[0]
    for s in range(n_birds):
        singer_bins = history_bins[:,s]
        random_start = np.random.randint(0,n_bins)
        idx = np.mod(random_start + np.arange(n_bins), n_bins)
        singer_shift = np.array(singer_bins[idx])
        shifted_bins[:,s] = singer_shift
    return shifted_bins

def count_sequence(sorted_data,meta,window=100,plot_me=False,g_kernel = .5,t0=0):
    zero_hour = datetime.strptime('00:00:00','%H:%M:%S')

    all_counts_m = []
    all_counts_f = []
    all_counts_u = []
    all_counts_sum = []
    ts, windows = [],[]
    window_indices = []
    count = 0
    
    m_count,f_count,u_count = 0,0,0
    bin_start = meta.timestamps[0]
    i_start = 0
    for t in range(len(meta.timestamps)):
        dt = datetime.fromtimestamp(meta.timestamps[t])
        if dt.time() == zero_hour.time():
            continue
        elif bin_start == 0:
            bin_start = meta.timestamps[t-1]
            i_start = t-1
        ## If it's been a while, store the previous bin and start a new bin
        #print(t,bin_start,meta.timestamps[t] - bin_start,count,m_count,windows)
        
        ## Need to define windows in order to pull out the history bins.
        ## Start_t = bin_start -> 
        if meta.timestamps[t] - bin_start >= window:
            if count == 1:  ## How do we account for windows of size 0...could use more info here
                pass
            elif count != 0:  ## What if we do have a long window with no songs?
                all_counts_f.append(f_count)
                all_counts_m.append(m_count)
                all_counts_u.append(u_count)
                all_counts_sum.append(count)
                ts.append(bin_start)
                win_size = meta.timestamps[t-1] - bin_start + 1
                if win_size < window:
                    win_size = window
                    pass
                windows.append(win_size) ## 
                window_indices.append([i_start,t-1])
            count = 0
            m_count,f_count,u_count = 0,0,0
            bin_start = 0 #meta.timestamps[t]
            ## If it's a big jump, make it clear that it's not continuous
            if meta.timestamps[t] - meta.timestamps[t-1] >= 600:
                all_counts_f.extend([np.nan])
                all_counts_m.extend([np.nan])
                all_counts_u.extend([np.nan])
                all_counts_sum.extend([np.nan])
                ts.append(meta.timestamps[t-1] + 1) ## Add the start of a gap, 1s after the previous label
                windows.append(meta.timestamps[t] - meta.timestamps[t-1] - 1) ## Add length of gap
        singer = sorted_data[t,0]
        receiver = sorted_data[t,1]
        if singer[0].upper() == 'M':
            if receiver == '':
                # undirected
                u_count += 1
            elif receiver[0].upper() == 'M':
                # Male directed
                m_count += 1
            elif receiver[0].upper() == 'F':
                # Female directed
                f_count += 1
        else:
            pass ## No songs, but it was an event, still worth counting I think
        count += 1

    all_counts_m = np.array(all_counts_m)
    all_counts_f = np.array(all_counts_f)
    all_counts_u = np.array(all_counts_u)
    all_counts_sum = np.array(all_counts_sum)
    ts = np.array(ts)
    windows = np.array(windows)
    male_high = all_counts_m > all_counts_f
    female_high = all_counts_f > all_counts_m

    #print(all_counts_m,windows)
    male_counts_per_min = np.divide(all_counts_m,windows) * 60
    female_counts_per_min = np.divide(all_counts_f,windows) * 60
    undirected_counts_per_min = np.divide(all_counts_u,windows) * 60
    
    ## Count up the lengths of segments
    m_lengths = [ sum( 1 for _ in group ) for key, group in itertools.groupby( male_high ) if key == 1 ]
    f_lengths = [ sum( 1 for _ in group ) for key, group in itertools.groupby( female_high ) if key == 1 ]

    if plot_me == -1:
        male_counts_per_min = male_counts_per_min[~np.isnan(male_counts_per_min)]
        female_counts_per_min = female_counts_per_min[~np.isnan(female_counts_per_min)]
        undirected_counts_per_min = undirected_counts_per_min[~np.isnan(undirected_counts_per_min)]
        fig,ax = plt.subplots()
        ax.hist2d(male_counts_per_min,female_counts_per_min)
        #ax.scatter(male_counts_per_min,female_counts_per_min,alpha=.1)
        ax.set_xlabel('male songs per minute')
        ax.set_ylabel('female songs per minute')
        #print(male_counts_per_min.shape)
        ax.set_xlim([0,np.nanmax(male_counts_per_min)])
        ax.set_ylim([0,np.nanmax(female_counts_per_min)])
        #print(all_counts_m,male_counts_per_min)
        fig.show()
    
    if plot_me == -2:
        fig,ax = plt.subplots()
        nbins = 20
        male_counts_per_min = male_counts_per_min[~np.isnan(male_counts_per_min)]
        female_counts_per_min = female_counts_per_min[~np.isnan(female_counts_per_min)]  
        x,y = male_counts_per_min,female_counts_per_min
        k = kde.gaussian_kde([x,y],bw_method=g_kernel)
        xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        ax.set_title(meta.name)
        ax.set_xlabel('Male songs per minute')
        ax.set_ylabel('Female songs per minute')
        ax.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.jet)
        ax.contour(xi, yi, zi.reshape(xi.shape),cmap='viridis_r')
        fig.show()
        
    if plot_me > 0:
        fig,ax = plt.subplots()
        ax.hist(m_lengths,alpha=.4,label='Male Directed Sequences')
        ax.hist(f_lengths,alpha=.4,label='Female Directed Sequences')
        ax.set_title(meta.name)
        ax.set_xlabel('length of segments')
        ax.set_ylabel('counts')
        ax.set_xlim([1,10])
        ax.legend()
        fig.show()

    if plot_me > 1:
        xs = ts[~np.isnan(male_counts_per_min)]
        male_counts_per_min = male_counts_per_min[~np.isnan(male_counts_per_min)]
        female_counts_per_min = female_counts_per_min[~np.isnan(female_counts_per_min)]
        undirected_counts_per_min = undirected_counts_per_min[~np.isnan(undirected_counts_per_min)]
        
        fig,(ax,ax1) = plt.subplots(2)

        ax.hist(all_counts_m[~np.isnan(all_counts_m)],alpha=.5)
        ax.hist(all_counts_f[~np.isnan(all_counts_f)],alpha=.5)

        ax.hist(all_counts_u[~np.isnan(all_counts_u)],alpha=.5)

        print(ts.shape)
        
        xs_nan = ts[np.isnan(all_counts_m)]
        
        #for x in xs_nan:
        #    ax1.axvline(x,color='black')
        if False:
            ax1.scatter(xs,GF(all_counts_m[~np.isnan(all_counts_m)],g_kernel),label='male directed',marker='.')
            ax1.scatter(xs,GF(all_counts_f[~np.isnan(all_counts_m)],g_kernel),label='female directed',marker='.')

            ax1.scatter(xs,GF(all_counts_u[~np.isnan(all_counts_m)],g_kernel),label='undirected',marker='.')
        
        #xs = np.arange(len(all_counts_m[~np.isnan(all_counts_m)]))
        #print(male_counts_per_min.shape,xs.shape)
        if False:
            ax1.scatter(xs,GF(male_counts_per_min,g_kernel),label='male directed',marker='.')
            ax1.scatter(xs,GF(female_counts_per_min,g_kernel),label='female directed',marker='.')
            ax1.scatter(xs,GF(undirected_counts_per_min,g_kernel),label='undirected',marker='.')
        
        #print(xs,male_counts_per_min)
        ax1.scatter(xs,male_counts_per_min,label='male directed',marker='.')
        ax1.scatter(xs,female_counts_per_min,label='female directed',marker='.')
        ax1.scatter(xs,undirected_counts_per_min,label='undirected',marker='.')
        
        ax1.plot(xs,male_counts_per_min,linestyle=':')
        ax1.plot(xs,female_counts_per_min,linestyle=':')
        ax1.plot(xs,undirected_counts_per_min,linestyle=':')
        
        ax1.set_ylabel('Songs per minute')
        
        ax1.set_xlim([ts[t0]-30,ts[t0]+2030])
        #ax1.set_xlim([300,350])
        ax1.legend()
        
        fig.set_size_inches([10,5])
        fig.show()
        
    return [all_counts_m,all_counts_f,all_counts_u],window_indices


## Get the indices where copulations occured
def get_cop_indices(copulation_history):
    bin_counts = np.sum(copulation_history,axis=(1,2))
    return np.arange(len(bin_counts))[bin_counts >= 1]

## Convert history indices into bin indices
def get_cop_bins(cop_indices,window_indices):
    cop_bins = []
    for i in range(len(window_indices)):
        wins = window_indices[i]
        for j in cop_indices:
            if j>=wins[0] and j<=wins[1]:
                cop_bins.append(i)
    return np.array(cop_bins)


## Define Function for Parsing egg data

# This spits out the egg_array, which is n_dates x n_females x 3: [Fertile, Not_fertile, unknown]
# It also returns datestamps, which is just a list of dates that they checked eggs on. Voila!

def parse_eggs(aviary_eggs,meta,date_format = None,start_day = None):
    ## Build date array and key
    if date_format == None:
        date_format = meta.date_format
    if date_format == 'ymd':
        date_format2 = '%y-%m-%d'
    elif date_format == 'Ymd':
        date_format2 = '%Y-%m-%d'
    else:
        print('Somebody set us up a bad format.')
    date_stamps = []
    dates = np.unique(aviary_eggs[:,1])

    if start_day is not None:
        start_stamp,_ = date_to_sec(start_day,date_format)
        date_stamps.append(start_stamp)


    for d in range(len(dates)):
        timestamp,_ = date_to_sec(dates[d],date_format)
        if start_day is not None:
            if timestamp < start_stamp:
                continue
        date_stamps.append(timestamp)

    # Sort, and identify timespan
    date_stamps = sorted(date_stamps)
    delta_s = date_stamps[-1] - date_stamps[0]
    delta_d = delta_s / 3600 / 24

    ## Fill in the missing dates
    date_stamps = np.arange(date_stamps[0],date_stamps[-1] + 3600,3600*24)
    dates = [timestamp_to_ymd(t,date_format2) for t in date_stamps]

    #print(len(dates),len(date_stamps))
    
    #n_dates = 14
    f_ids = meta.f_ids

    n_females = meta.n_females
    egg_array= np.zeros([len(dates),n_females+1,3])

    for d in range(len(dates)):
        eggs_by_date = aviary_eggs[aviary_eggs[:,1] == dates[d]]

        ## First sum up totals
        egg_labels,egg_counts = np.unique(eggs_by_date[:,4],return_counts=True)

        F_count = np.sum(egg_counts[egg_labels == 'F'])
        N_count = np.sum(egg_counts[egg_labels=='NF'])
        Un_count = np.sum(egg_counts) - F_count - N_count

        # Sum up each individual bird
        for f in range(len(f_ids)):
            eggs_by_female = eggs_by_date[eggs_by_date[:,9] == f_ids[f]]
            egg_labels_f,egg_counts_f = np.unique(eggs_by_female[:,4],return_counts=True)

            F_count_f = np.sum(egg_counts_f[egg_labels_f == 'F'])
            N_count_f = np.sum(egg_counts_f[egg_labels_f =='NF'])
            Un_count_f = np.sum(egg_counts_f) - F_count_f - N_count_f

            #print(F_count_f,N_count_f, Un_count_f)
            egg_array[d,f] = [F_count_f,N_count_f,Un_count_f]
        # Subtract total from individual to get unknown birds. In many years, most birds are unknown!
        totes_fem = np.sum(egg_array[d],0)
        egg_array[d,-1] = np.array([F_count,N_count,Un_count]) - totes_fem

    return egg_array,date_stamps

## Defining metrics of cohesion, as well as functions for measuring possible confounds

## Function to identify all instances of countersinging
def find_countersong(sorted_data,meta):
    countersongs = np.zeros(len(sorted_data))
    for l in range(len(sorted_data)):
        line = sorted_data[l]
        singer,receiver = line[0],line[1]

        if singer in meta.m_ids and receiver in meta.m_ids:
            if singer == receiver:
                continue
            t = meta.timestamps[l]
            window = (meta.timestamps > t - 15) & (meta.timestamps < t + 15)
            for sub_line in sorted_data[window]:
                if sub_line[0] == receiver and sub_line[1] == singer:
                    countersongs[l] = 1
                    break

    return countersongs

## Slightly convoluted function that groups countersong into bouts. 
def define_countersong_bouts(sorted_data,meta):
    countersongs = find_countersong(sorted_data,meta)
    processed_songs = []
    bout_dict = {}
    for l in range(len(sorted_data)):
        if not countersongs[l]:
            continue
        elif l in processed_songs:
            continue
        else:
            participants = [sorted_data[l,0],sorted_data[l,1]]
            bout = [l]
            processed_songs.append(l)
            t = meta.timestamps[l]
            window = (meta.timestamps >= t) & (meta.timestamps < t + 15)
            indices = window.nonzero()[0]
            i = 0
            while i < len(indices): ## careful....
                sub_l = indices[i]
                if sub_l in processed_songs:
                    i += 1
                    continue
                sub_line = sorted_data[sub_l]
                if sub_line[0] in participants and sub_line[1] in participants: 
                    bout.append(sub_l)
                    processed_songs.append(sub_l)
                    t = meta.timestamps[sub_l]
                    window = (meta.timestamps >= t) & (meta.timestamps < t + 15)
                    indices = window.nonzero()[0]
                    i = 0
                i += 1

            bout_dict[l] = bout
    return countersongs,bout_dict

## Get the percent of female directed song in each bin
def percent_to_females(history,meta,sorted_data,window=100,shuffle=False,a_filter=None):
    if True:
        history_bins,[history_rate_bins,ts,window_indices]= sliding_bin_history(sorted_data,history,meta,window=window,a_filter=a_filter)
    else:
        history_bins,[history_rate_bins,ts,window_indices]= bin_history(sorted_data,history,meta,window=window,a_filter=a_filter)
    if shuffle == True:
        history_bins = shuffle_indy_bins(history_bins)
    elif shuffle == 'Day':
        history_bins = shuffle_day_bins(history_bins,ts,meta)
    #print(np.sum(history_bins[:10],(1,2)))
    n_females = meta.n_females
    f_songs = np.sum(history_bins[:,n_females:,:n_females],axis=(2))
    all_songs = np.sum(history_bins[:,n_females:],axis=(2))
    undirected_songs = np.array([np.diagonal(history_bins[b]) for b in range(len(history_bins))])
    undirected_songs = undirected_songs[:,n_females:]
    f_ratio = f_songs / (all_songs - undirected_songs)
    return f_ratio,[f_songs,all_songs]

## Correlate males by the degree of consistency in the ratio of f-directed song
def correlate_fsongs(history,meta,sorted_data,window=100,prune = False,shuffle=False,n_pruned =10,sman=True):
    f_ratio, [f_songs,all_songs] = percent_to_females(history,meta,sorted_data,window,shuffle)

    if sman:
        c_function = spearmanr
    else:
        c_function = pearsonr
    if prune: 
        f_ratio = f_ratio[:,np.sum(f_songs,0) >= n_pruned]

    n_males = f_ratio.shape[1]
    corr_matrix = np.zeros([n_males,n_males])
    xs,ys = [],[]
    for m in range(n_males):
        for n in range(n_males):
            m_vals = f_ratio[:,m]
            n_vals = f_ratio[:,n]

            m_vals_clean = m_vals[~np.isnan(m_vals) & ~np.isnan(n_vals)]
            n_vals_clean = n_vals[~np.isnan(m_vals) & ~np.isnan(n_vals)]
            if len(m_vals_clean) > 1 and len(n_vals_clean) > 1:
                r,p = spearmanr(m_vals_clean,n_vals_clean)
            else:
                r = np.nan
            if m != n:
                xs.extend(m_vals_clean)
                ys.extend(n_vals_clean)
                corr_matrix[m,n] = r
            if m == n:
                #corr_matrix[m,n] = r
                corr_matrix[m,n] = np.nan
    if False:
        fig,ax = plt.subplots()
        ax.scatter(xs,ys,alpha=.2)
        fig.show()
    return corr_matrix

## AS above, but uses only a subset of f_ratio
def correlate_subset(sub_set):
    n_males = sub_set.shape[1]
    corr_matrix = np.zeros([n_males,n_males])
    for m in range(n_males):
        for n in range(n_males):
            if m == n:
                corr_matrix[m,n] = np.nan
                continue
            m_vals = sub_set[:,m]
            n_vals = sub_set[:,n]

            m_vals_clean = m_vals[~np.isnan(m_vals) & ~np.isnan(n_vals)]
            n_vals_clean = n_vals[~np.isnan(m_vals) & ~np.isnan(n_vals)]
            if len(m_vals_clean) > 1 and len(n_vals_clean) > 1:
                r,p = spearmanr(m_vals_clean,n_vals_clean)
            else:
                r = np.nan
            corr_matrix[m,n] = r
    return corr_matrix

## Function to return correlation iterating by sampling males
def bootstrap_n_males(corr_matrix):
    n_males = np.shape(corr_matrix)[0]
    m_array = np.arange(n_males)
    subset_values = {}
    for r in range(2,n_males):
        subsets = itertools.combinations(m_array,r)
        subset_values[r] = []
        for subset in subsets:
            subset = list(subset)
            rows = corr_matrix[subset]
            sub_matrix = rows[:,subset]
            sub_corr = np.nanmean(sub_matrix)
            subset_values[r].append(sub_corr)
    return subset_values


def sex_ratio(meta):
    return meta.n_females / meta.n_males

## Calculate the participation by females
s_threshold = 20
def f_participation(history,meta):
    n_females = meta.n_females
    f_sums = np.sum(history[:,n_females:,:n_females],axis=(0,1))
    return np.sum(f_sums > s_threshold)

def f_pairbonded(history,meta):
    ## A female has a pairbond if she is participating, and receives > 60% of her songs from one male
    summed_songs = np.sum(history[:,n_females:,:n_females],axis=0)
    f_sums = np.sum(summed_songs,axis=0)

    song_ratios = np.transpose(summed_songs) / f_sums[:,None]

    pairbonds = np.max(song_ratios,axis=1) > .6
    participation = f_sums > s_threshold
    return np.sum(pairbonds * participation)

if __name__ == "__main__":
    fname2 = './AviaryDataFiles/BS2017.txt'
    test_data = np.genfromtxt(fname2,delimiter=',',dtype=str)

    reorder = [4,5,6,7,8,1,3]

    reorder_data = test_data[:,reorder]

    dar_data = reorder_data[reorder_data[:,6] == 'DARWIN']
    cop_data = reorder_data[reorder_data[:,6] == 'COOP']
    cop_sorted, cop_meta = sort_data(cop_data, aviary='ON1-2017',date_format = 'dmy')
    cop_history, cop_copulations = build_history(cop_sorted,cop_meta)
    meta=cop_meta
    sorted_data=cop_sorted
    history=cop_history
    history_bins,[history_rate_bins,ts,window_indices]=sliding_bin_history(sorted_data,history,meta,window=60)
    #print(shuffle_day_bins(history_bins,ts,meta).shape)
    f_ratio,_ = percent_to_females(history,meta,sorted_data)

    corr_matrix = correlate_fsongs(history,meta,sorted_data,prune=True)
    subset_values = bootstrap_n_males(corr_matrix)
    #print(subset_values)
    subset_means = []
    subset_rs = []
    for r in sorted(subset_values.keys()):
        print(np.nanmean(subset_values[r]))
        subset_means.append(np.nanmean(subset_values[r]))
        subset_rs.append(r)
        
    print(pearsonr(subset_rs,subset_means))


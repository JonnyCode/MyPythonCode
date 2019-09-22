#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 20:58:32 2019

@author: jcafaro
"""
# simple simulation of encoding and decoding entact/degraded measured neuron population
print('start')
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as linalg

# model parameters
sim_repeat_num = 10 ;
sample_rate = 1000 # (hz)
cell_num_tuned = 32
cell_num_untuned = 0
stim_range = (-10,10) # (au)
time_max = 10 # (sec)
ref_time = 0.002 # (sec) abs ref period
untuned_resp_prob_range = (0.01,.3) # prob response range for untuned cells
train_frac = 0.5 # fraction of time to train on (remaining fraction is test)
seed_num = 1 # random seed number (base seed)
set_spike_seed_flag = False
set_stim_seed_flag = False
stim_rand_uniform_flag = True
load_ole_weights_flag = False
save_flag = True

#save_path = '/media/tdy-share/Data/INI/SimulatedSpike'
save_path = '~/Desktop/Temp_out'

# degradation parameters
deg_set_num = cell_num_tuned # number of data sets with increasing degradation

ch_drop_flag = True
ch_drop_num = 1 # (num ch drop/set)

rate_drop_flag = True
rate_drop_prob = 0.1 # (prob drop/set) probability that a firing rate change happens
rate_drop = .1 # (rate_drop/set)

# from parameters
cell_num = cell_num_tuned + cell_num_untuned
time_pnts_num = int(time_max * sample_rate)
ref_num_pnts = int(ref_time * sample_rate)


spike_train = []
decoder_error= []

for rp in range(sim_repeat_num):
    
    # make stim that varies across stim range
    if stim_rand_uniform_flag:
        if set_stim_seed_flag:
            np.random.seed(seed_num) # set random seed
        else:
            np.random.seed(seed=None)  # set random seed
    
        stim = np.random.uniform(stim_range[0],stim_range[1],(1,time_pnts_num))
    else:
        stim = np.empty((1,time_pnts_num))
        stim[0] = np.array(np.arange(stim_range[0],stim_range[1],(stim_range[1]-stim_range[0])/time_pnts_num))
    
    # model responses (encoder)
    
    # make tuning curves
    np.random.seed(seed_num+1) # set random seed
    tc_mean = np.random.uniform(stim_range[0],stim_range[1],(1,cell_num_tuned))
    
    np.random.seed(seed_num+2) # set random seed
    tc_std = np.random.uniform(stim_range[0],stim_range[1],(1,cell_num_tuned))
    
    # make responses spike prob
    resp_prob = np.zeros((cell_num,time_pnts_num))
    
    for c in range(cell_num): # for each cell
        if c<cell_num_tuned: # if its tuned
            resp_prob[c,:] = np.exp(-(1/2)*((stim-tc_mean[0,c])/tc_std[0,c])**2) # gaussian
        else: # if its untuned
            resp_prob[c,:] = np.random.uniform(untuned_resp_prob_range[0],untuned_resp_prob_range[1])*np.ones((1,time_pnts_num))
    
    # poisson spike generator with refractory period
    resp = np.zeros((cell_num, time_pnts_num)) # default response mat
    for c in range(cell_num): # for each cell
        if set_spike_seed_flag:
            np.random.seed(seed_num)
        else:
            np.random.seed(np.random.seed(seed=None))
    
        temp = np.random.uniform(0,1,(1,time_pnts_num))
        ref=0
        for t in range(time_pnts_num): # for each time point
            if ref == 0: # if not refractory
                if resp_prob[c,t]>temp[0,t]:
                    resp[c,t]=1 # spike
                    ref = ref_num_pnts # enter ref period
            else:
                ref -=1
    
    # degrade response set
    resp_deg = np.empty((cell_num, time_pnts_num,deg_set_num))
    ch_drop_order = np.random.permutation(cell_num)
    
    if ch_drop_flag: # drop channels
        for s in range(deg_set_num): # for each degraded set
            if s==0:
                resp_deg[:,:,s]= resp
            else:
                drp_set = ch_drop_order[0:s*ch_drop_num]
                resp_deg[:,:,s]= resp
                resp_deg[drp_set,:,s] *=0
    
    # decoder (Optimal Linear Estimator)
    
    train_pnts_num = np.int(time_pnts_num*train_frac)
    test_pnts_num = time_pnts_num-train_pnts_num
    
    if load_ole_weights_flag:
        ole_weights = np.load(f"{save_path}/ole_wieghts_seed_{seed_num}.npy")
    else:
        ole_weights = linalg.lstsq(resp[:,0:train_pnts_num].T,stim[:,0:train_pnts_num].T) # least square weights trained on non-degraded signal
        #np.save(f"{save_path}/ole_wieghts_seed_{seed_num}.npy", ole_weights)
    
    
    stim_est = resp[:,train_pnts_num:].T@ole_weights[0]
    
    stim_est_deg = np.empty((test_pnts_num,deg_set_num))
    for s in range(deg_set_num):
        stim_est_deg[:,s] = np.ndarray.flatten(resp_deg[:,train_pnts_num:,s].T@ole_weights[0])
    
    # error
    mse = np.mean((stim[0,train_pnts_num:]-stim_est.T[0])**2)
    
    mse_deg = np.empty(deg_set_num)
    for s in range(deg_set_num):
        mse_deg[s] = np.mean((stim[0,train_pnts_num:]-stim_est_deg[:,s].T)**2)
    
    print('mse:',mse)
    print('mse degrasded:',mse_deg )
    
    spike_train.append(resp_deg)
    decoder_error.append(mse)
     
if save_flag: # save numpy array
    np.save(f"{save_path}/spike_train_.npy", spike_train)
    np.save(f"{save_path}/spike_train_.npy", decoder_error)
    np.save(f"{save_path}/spike_train_.npy", stim)
    

# %% figures
plt.figure(1) #true stim and est
plt.plot(stim[0,train_pnts_num:],'k')
plt.plot(stim_est.T[0],'r--')

plt.figure(2)
plt.imshow(resp,aspect='auto')
plt.colorbar()

plt.figure(3)
plt.imshow(resp_prob,aspect='auto')

plt.figure(4)
s =  np.arange(stim_range[0],stim_range[1],.1)
for c in range(cell_num):
    if c<cell_num_tuned:
        g = np.exp(-(1 / 2) * ((s - tc_mean[0, c]) / tc_std[0, c]) ** 2)  # gaussian
    else:
        g= resp_prob[c,0]*np.ones(s.shape)

    plt.plot(s,g)

plt.figure(5) #true stim and est for degraded sets
plt.plot(stim[0,train_pnts_num:],'k')
for s in range(deg_set_num):
    plt.plot(stim_est_deg[:,s])
    plt.title(str(s))
    plt.pause(1)

plt.figure(6) # degraded sets
for s in range(deg_set_num):
    plt.imshow(resp_deg[:,:,s], aspect='auto')
    plt.title(str(s))
    plt.pause(1)

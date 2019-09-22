# this script will generate a stimulus and run encoding and decoding function
# JC 2019-01-28

verbose = True # print/plot sanity checks
StimType = "StimWn"
EncoderType = "LnModel1"
DecoderType = "WarlandModel"

## import
import numpy as np
import matplotlib.pyplot as plt

## global parameters
GlobParams = dict() # make parameters a dictionary
GlobParams["Duration"] = 10 # (sec)
GlobParams["TimeStep"] = .001 # (sec)
GlobParams["Width"] = 100 # (pix) stim field
GlobParams["Hieght"] = 50 # (pix) stim field

## stimulus parameters
StimParams = dict() # make parameters a dictionary 
StimParams["StimWn"] = dict() # make parameters dictionary for wn
StimParams["StimWn"]["Mean"] = 256/2 # (au) mean
StimParams["StimWn"]["Std"] = 256/4 # (au) std

## encoder parameters
EncoderParams = dict() # make parameters a dictionary
EncoderParams["LnModel1"] = dict() # make parameters a dictionary for simple ln model 1
EncoderParams["LnModel1"]["srf_width"] = 20 # (um) width of gaussian srf
EncoderParams["LnModel1"]["trf_tau"] = 100 # (ms) tau of exponential trf
EncoderParams["LnModel1"]["trf_delay"] = 100 # (ms) delay of expontential trf
EncoderParams["LnModel1"]["trf_tauLength"] = 3 # number of tau that trf extends

## decoder parameters 

## global variables
T = np.int(GlobParams["Duration"]/GlobParams["TimeStep"]) # number time points

## generate stimulus
if StimType == "StimWn":
    Stim = np.random.normal(StimParams["StimWn"]["Mean"],\
                            StimParams["StimWn"]["Std"],\
                            (GlobParams["Width"],\
                             GlobParams["Hieght"],\
                             T)) # normal random (W,H,T)
    
    if verbose:
        plt.imshow(Stim[:,:,1])
        plt.show()
        
## encode stimulus

if EncoderType == "LnModel1":
    # (number) of cells in grid
    C = int(np.ceil(GlobParams["Width"]\
        /EncoderParams["LnModel1"]["srf_width"])\
            *np.ceil(GlobParams["Hieght"]\
        /EncoderParams["LnModel1"]["srf_width"])) 
    
    trf_l = EncoderParams["LnModel1"]["trf_tau"]\
    *EncoderParams["LnModel1"]["trf_tauLength"] # length of trf
    
    # make trf - decaying exponential with delay
    trf  = np.exp((np.arange(-trf_l,trf_l+1)-EncoderParams["LnModel1"]["trf_delay"])/-EncoderParams["LnModel1"]["trf_tau"]) 
    trf[0:trf_l+EncoderParams["LnModel1"]["trf_delay"]]=0
    
    # make srf - gaussian spread across stim field
    import gaussMaker2d as gm
    
    ax=np.arange(EncoderParams["LnModel1"]["srf_width"]/2,GlobParams["Width"],EncoderParams["LnModel1"]["srf_width"])
    ay=np.arange(EncoderParams["LnModel1"]["srf_width"]/2,GlobParams["Hieght"],EncoderParams["LnModel1"]["srf_width"])
    xx,yy = np.meshgrid(ax,ay)

    srf = np.empty((GlobParams["Hieght"],GlobParams["Width"],C)) # srf(W,H,cell)
    for ci in range(C): # for each cell
        srf_center = [xx.flat[ci],yy.flat[ci]]
        print(ci)
        srf[:,:,ci] = gm.gaussMaker2d((GlobParams["Width"],GlobParams["Hieght"]),srf_center,EncoderParams["LnModel1"]["srf_width"])
        print(ci)
        

    # dot product over space
    sdot = np.empty((C,T))
    for ci in range(C): # for each cell
        for ti in range(T): # for each time point in StimLong
            sdot[ci,ti] = np.sum(srf[:,:,ci]*Stim[:,:,ti])
            
    # convoluation over time 
    lp = np.empty((C,T))
    sdot2 = sdot - np.mean(sdot)
    for ci in range(C): # for each cell
        lp[ci,:] = np.convolve(sdot2[ci,:],trf,'same')
        
    # implement exponential non-linearity
    R = lp**2 ;
    
    
    if verbose:
        
        print("Number of cells =", str(C)) # print the number of cell
        
        # plot trf
        plt.plot(np.arange(-trf_l,trf_l+1),trf)
        plt.show()
        
        # plot trf pulse response
        temp=np.zeros(3000)
        temp[1000]=1
        temp2 = np.convolve(trf,temp,'same')

        plt.plot(temp2)
        plt.plot(temp)
        plt.show()
        
        # plot linear prediction
        plt.plot(lp[0,:])
        plt.show()
        
        # plot response
        plt.plot(R[0,:])
        plt.plot(R[1,:])
        plt.show()


## decode stimulus

## calculate error



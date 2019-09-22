import numpy as np
import h5py
arrays = {}
f = h5py.File('/Users/jcafaro/Dropbox/CNN Retina Project/BWDb1.mat')
for k, v in f.items():
    arrays[k] = np.array(v)
    
print('Variables in .mat: ', arrays.keys())
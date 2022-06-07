
from matplotlib import use
use('Agg') # no windows
import math
import numpy as np
import numpy.ma as npm
from sklearn.decomposition import non_negative_factorization, NMF, PCA, DictionaryLearning,MiniBatchDictionaryLearning
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import griddata
from scipy.ndimage import rotate, shift
import matplotlib.pyplot as plt
from matplotlib import cm
from os import path, mkdir
import pickle
from time import time as now
from scipy.spatial.transform import Rotation as R
import scipy.io as sio

TYPE  = 'flow'# 'vis'  # 'slow' 'both'
MODEL= 'rightwide'#'box' # 'wall'
PATH = 'expt127widevis'  # 'wall'
THRESHOLD = 90
DICT=100
REG='both'
ALG = 'mnf'
if 'wide' in PATH:
    VX=170
    VY=110
else:
    VX=80
    VY=80

#DICT=600
ITER = 2

FIGDIR=f"figp{PATH}m{MODEL}{DICT}{TYPE}{REG[:4]}{THRESHOLD}t{now()}"
TEXT = f'vf{MODEL}{TYPE}{REG[:4]}{DICT}.txt'

        
    
    
 
if __name__ == '__main__':
    for TYPE in ['flow']:
        for PATH in ['expt127wide','rightwide']:#,'wall']:
            for MODEL in ['rightwide']:#,'wall']:
                with open(f'{PATH}mazedat.npy','rb') as f:
                   visdat = np.load(f)
                   hprdat = np.load(f)
                   dhprdat = np.diff(hprdat,axis=0)
                   hprdat = hprdat[1:,:]
                   posdat = np.load(f)[1:,:]
                if TYPE == 'both':
                    POS=False
                    Nim = 5
                elif TYPE=='vis':
                    POS=True
                    visdat=visdat[:,:VX*VY]
                    Nim=1
                elif TYPE=='flow':
                    visdat=visdat[:,VX*VY:]
                    Nim=4
                elif TYPE=='slow':
                    visdat=visdat[:,VX*VY:]/10
                    Nim=4
                else:
                    raise('invalid model type')
                MODEL_FILE = f'p{PATH}m{MODEL}{TYPE}{REG[:4]}model{DICT}I{ITER}.pkl'
                print(MODEL_FILE)
                if path.exists(MODEL_FILE):
                    with open(MODEL_FILE,'rb') as fh:
                        model = pickle.load(fh)
                        W = model.transform(visdat)
                else:
                    raise "train model first"
                MATFILE = f"p{PATH}m{MODEL}{DICT}{TYPE}{THRESHOLD}tI{ITER}.mat"
                sio.savemat(MATFILE,dict(pos=posdat[:,0:2],md=hprdat[:,0],resp=W))
   # plt.close('all')

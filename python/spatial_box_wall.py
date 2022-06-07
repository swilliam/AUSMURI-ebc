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

TYPE = 'vis' #'flow'  # 'slow' 'both'
ALG = 'mnf'
VX=80
VY=80
MODEL='box2wall'
PATH='wall'

THRESHOLD = 90
DICT=100
#DICT=600
ITER = 100000

FIGDIR=f"figp{PATH}m{MODEL}{DICT}{TYPE}{THRESHOLD}t{now()}"
TEXT = f'vf{MODEL}{TYPE}{DICT}.txt'


def extr_visualise(c, pos, hpr, dhpr, data):
    fig = plt.figure(c+1, figsize=(9,8))
    b,MN, MX,t = np.percentile(data,[0,10,THRESHOLD,100])
    plt.title(f"min: {MN}, max: {MX}")
#    extr = (MN > data) | (data > MX)
    extr = (data > MX)
    s1 = plt.scatter(pos[~extr,0],pos[~extr,1],c=hpr[~extr],cmap=plt.get_cmap('hsv'))
    s2 = plt.scatter(pos[extr,0],pos[extr,1],c=hpr[extr], s=100,cmap=plt.get_cmap('hsv'),alpha=0.5)
    plt.colorbar(s1)
    plt.savefig(f'{FIGDIR}/FAC{TYPE}{c:03d}.png')
    plt.close()



def quiver_visualise(c, pos, hpr, dhpr, data):
    fig = plt.figure(c+1, figsize=(8,8))
    b,MN, MX,t = np.percentile(data,[0,10,THRESHOLD,100])
    print(f"min: {MN}, max: {MX}")
#    extr = (MN > data) | (data > MX)
    extr = (data > MX)
    scale_data = (data-MX)/(t-MX)+0.2
    s2 = plt.quiver(pos[extr,0],pos[extr,1],np.cos((90-hpr[extr])*math.pi/180) *
                    scale_data[extr],
                    np.sin((90-hpr[extr])*math.pi/180)*scale_data[extr],scale=10)
    # s2 = plt.quiver(pos[extr,0],pos[extr,1],np.cos(hpr[extr]*math.pi/180),
    #                 np.sin(hpr[extr]*math.pi/180),scale=10)
    plt.savefig(f'{FIGDIR}/QS{TYPE}{c:03d}.png')
    plt.close()

def angle_visualise(c, pos, hpr, dhpr, data):
    fig = plt.figure(c+1, figsize=(9,8))
    b,MN, MX,t = np.percentile(data,[0,10,THRESHOLD,100])
#    extr = (MN > data) | (data > MX)
    extr = (data > MX)
    scale_data = (data-MX)/(t-MX)+0.2
    s2 = plt.scatter(pos[extr,0],pos[extr,1],c=hpr[extr],s=5,cmap=plt.get_cmap('hsv'))
    # s2 = plt.quiver(pos[extr,0],pos[extr,1],np.cos(hpr[extr]*math.pi/180),
    #                 np.sin(hpr[extr]*math.pi/180),scale=10)
    plt.colorbar(s2)
    plt.savefig(f'{FIGDIR}/AC{TYPE}{c:03d}.png')
    plt.close()
    
def marginal_visualise(c,pos, hpr, dhpr, data):
    b,MN, MX,t = np.percentile(data,[0,10,THRESHOLD,100])
#    extr = (MN > data) | (data > MX)
    extr = (data > MX)
    fig = plt.figure(c+1, figsize=(24,18))
    ax1 = fig.add_subplot(221,xlabel='x',ylabel=r'$\theta$',title=f'Dict #{fig.number}')
    s1 = ax1.scatter(pos[~extr,0],hpr[~extr], s=5)
    s2 = ax1.scatter(pos[extr,0],hpr[extr])
    ax2 = fig.add_subplot(222,xlabel='y')
    s1 = ax2.scatter(pos[~extr,1],hpr[~extr],s=5)
    s2 = ax2.scatter(pos[extr,1],hpr[extr])
    ax3 = fig.add_subplot(223,ylim=(-50,50),xlabel='x',ylabel=r'$d\theta/dt$')
    s1 = ax3.scatter(pos[~extr,0],dhpr[~extr],s=5)
    s2 = ax3.scatter(pos[extr,0],dhpr[extr])
    ax4 = fig.add_subplot(224,ylim=(-50,50),xlabel='y')
    s1 = ax4.scatter(pos[~extr,1],dhpr[~extr],s=5)
    s2 = ax4.scatter(pos[extr,1],dhpr[extr])
    plt.savefig(f'{FIGDIR}/MV{TYPE}{c:03d}.png')
    plt.close()

    
def field_visualise(c,pos, hpr, dhpr, data):
    fig = plt.figure(c+1, figsize=(8,8))
    xi = np.linspace(0,500,100)
    yi = np.linspace(0,500,100)
    X,Y = np.meshgrid(xi,yi)
    Z = griddata((pos[:,0], pos[:,1]), data, (X, Y), method = 'linear')
    plt.imshow(Z,cmap=plt.get_cmap('jet'),aspect='equal')
    plt.savefig(f'{FIGDIR}/VF{TYPE}{c:03d}.png')
    plt.close()

Rebc = 248   
def wall_occupancy(pos, hpr):
    ebc = np.zeros((2*Rebc,2*Rebc))
    walls = np.zeros((2*500, 2*500))
    walls[250:750, 245:255] = np.ones((500,10))
    walls[250:750, 745:755] = np.ones((500,10))
    walls[245:255, 250:750] = np.ones((10,500))
    walls[745:755, 250:750:] = np.ones((10,500))
    for x, y, hd in zip(pos[:, 0], pos[:, 1], hpr):
        sw = shift(walls, [-x,-y])  # shift to centre
        rsw = rotate(sw, -hd)
        ebc += rsw[(500-248):(500+248),(500-248):(500+248)]
    return ebc
 
def ebr_visualise(c,pos, hpr, dhpr, data, woc):
    b,MN, MX,t = np.percentile(data,[0,10,THRESHOLD,100])
#    extr = (MN > data) | (data > MX)
    extr = (data > MX)
    fig = plt.figure(c+1, figsize=(8,8))
    ebc = np.zeros((2*Rebc,2*Rebc))
    walls = np.zeros((2*500, 2*500))
    walls[250:750, 245:255] = np.ones((500,10))
    walls[250:750, 745:755] = np.ones((500,10))
    walls[245:255, 250:750] = np.ones((10,500))
    walls[745:755, 250:750:] = np.ones((10,500))
    for x,y,hd,v in zip(pos[extr,0], pos[extr,1],hpr[extr],data[extr]):
        sw = shift(walls, [-x,-y])  # shift to centre
        rsw = rotate(sw, -hd)
        ebc += rsw[(500-248):(500+248),(500-248):(500+248)]*v

    with np.nditer(ebc,flags=['multi_index'],op_flags=['writeonly']) as ebci:
        for x in ebci:
            if (ebci.multi_index[0]-Rebc)**2 + (ebci.multi_index[1]-Rebc)**2 >= Rebc**2:
                ebc[ebci.multi_index[0],ebci.multi_index[1]] = npm.masked
            
    plt.imshow(ebc/woc, aspect='equal', interpolation='gaussian')
    plt.savefig(f'{FIGDIR}/EB{TYPE}{c:03d}.png')
    plt.close()
        
    
    
 
if __name__ == '__main__':
    with open(f'boxmazedat.npy','rb') as f:
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
    MODEL = f'box{TYPE}model{DICT}.pkl'
    if path.exists(MODEL):
        with open(MODEL,'rb') as fh:
            bmodel = pickle.load(fh)
        bW = bmodel.transform(visdat)
    pca = bmodel.named_steps['nmf']
    bH=pca.components_

    with open(f'wallmazedat.npy','rb') as f:
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
        # model = make_pipeline(MiniBatchDictionaryLearning(n_components=DICT,
        #                                          n_jobs=-1,fit_algorithm='cd',
        #                                          transform_algorithm='lasso_cd',
        #                                                   transform_max_iter=10000,
        #                                                   positive_code=POS,verbose=True,
        #                                                   batch_size=1000,n_iter=ITER
        # ))
    MODEL = f'{MODEL}{TYPE}model{DICT}.pkl'
    model = make_pipeline(NMF(n_components=DICT,max_iter=ITER,
                              alpha=1.0, init='custom',
                              verbose=True, l1_ratio=1.0))
    W=model.fit_transform(visdat,nmf__W=bW,nmf__H=bH)
    with open(MODEL,'wb') as fh:
            pickle.dump(model,fh)
    mkdir(FIGDIR)
    fig = plt.figure(0,figsize=(8,8))
    plt.plot(posdat[:,0],posdat[:,1])
    plt.savefig(f'{FIGDIR}path.png')
    plt.close()
    # # plt.plot(np.diff(hprdat[1:,0]-135,prepend=0))
    # # plt.show()
    pca = model.named_steps['nmf']
    H=pca.components_
    import sys
    np.savetxt(TEXT,np.hstack((posdat[:,0:2],W)))
    if path.exists(f'woc{Rebc}.txt'):
        print('loading woc')
        woc = np.loadtxt(f'woc{Rebc}.txt')
    else:
        print('creating woc')
        woc= wall_occupancy(posdat, hprdat[:,0])
        np.savetxt(f'woc{Rebc}.txt',woc)
    for c in range(H.shape[0]):
        print(f'd: {c}')
        plt.figure(0, figsize=(8,8*Nim))
        plt.imshow(H[c,:].reshape((Nim*VY,VX)))
        plt.colorbar()
        plt.savefig(f'{FIGDIR}/SC{c:03d}.png')
        plt.close()
        # ax =  fig.add_subplot(111,projection='3d')
        #ax.set_zlim(-50,50)
        # ax.plot3D(posdat[1:,0],posdat[1:,1],W[:,0])
        # scd = ax.scatter(posdat[:,0],posdat[:,1],hprdat[:,0],c=-W[:,c], s=100)
        #marginal_visualise(c,posdat,hprdat[:,0],dhprdat[:,0],W[:,c])
        extr_visualise(c,posdat,hprdat[:,0],dhprdat[:,0],W[:,c])
        angle_visualise(c,posdat,hprdat[:,0],dhprdat[:,0],W[:,c])
        quiver_visualise(c,posdat,hprdat[:,0],dhprdat[:,0],W[:,c])
        field_visualise(c,posdat,hprdat[:,0],dhprdat[:,0],W[:,c])
        # if TYPE not in ['slow','flow']:
        ebr_visualise(c,posdat,hprdat[:,0],dhprdat[:,0],W[:,c], woc)
        # plt.colorbar(scd, ax=ax)
        # plt.close(c)
    # plt.show(block=False)
    # input('?')
    # plt.close('all')

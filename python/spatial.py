from matplotlib import use
use('Agg') # no windows
import math
import numpy as np
import numpy.ma as npm
from sklearn.decomposition import non_negative_factorization, NMF, PCA, DictionaryLearning,MiniBatchDictionaryLearning
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import griddata
from scipy.ndimage import rotate, shift, gaussian_filter
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.image import NonUniformImage
from os import path, mkdir
import pickle
from time import time as now
from scipy.spatial.transform import Rotation as R
from itertools import chain

# CLI interface
import click


TYPE  = 'vis' #'slow'  # 'flow' 'both'
MODEL= 'gap'#'box' # 'wall'
PATH = 'gapwide' #'rightwide' #'expt127widevis' #'box'#'boxwide' # 'light_floor' #   # 'wall'
THRESHOLD = 90
DICT=100
REG=0.5
ALG = 'mnf'
if 'wide' in PATH:
    VX=170
    VY=110
else:
    VX=80
    VY=80
#DICT=600
ITER = 10000

FIGDIR=f"figp{PATH}m{MODEL}{DICT}{TYPE}r{REG}{THRESHOLD}t{now()}"
MODEL_FILE = f'p{PATH}m{MODEL}{TYPE}r{REG}model{DICT}I{ITER}.pkl'
TEXT = f'vf{MODEL}{TYPE}r{REG}{DICT}.txt'


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
    print(f"min: {np.amin(data)}, max: {np.amax(data)}, mean: {np.mean(data)}")
    Fig = plt.figure(c+1, figsize=(10,10))
    xi = np.linspace(-5,5)
    yi = np.linspace(-5,5)
    X,Y = np.meshgrid(xi,yi)
    Z,xe,ye = np.histogram2d(pos[:,0], pos[:,1],bins=125, weights=data,
                             density=False)
    N,xe,ye = np.histogram2d(pos[:,0], pos[:,1],bins=125, weights=np.ones_like(data),
                             density=False)
    R=gaussian_filter(Z,sigma=1,mode='constant')/gaussian_filter(N,sigma=1,mode='constant')
    np.place(R,np.isnan(R),[0])
    print(f"min: {np.amin(R)}, max: {np.amax(R)}, mean: {np.mean(R)}")
    plt.imshow(R, aspect='equal')
    plt.colorbar()
    plt.savefig(f'{FIGDIR}/VF{TYPE}{c:03d}.png')
    plt.close()
    
def field_clip_visualise(c,pos, hpr, dhpr, data):
    b,MN, MX,t = np.percentile(data,[0,10,THRESHOLD,100])
#    extr = (MN > data) | (data > MX)
    extr = (data < MX)
    print(f"min: {np.amin(data[extr])}, max: {np.amax(data[extr])}, mean: {np.mean(data[extr])}")
    fig = plt.figure(c+1, figsize=(10,10))
    Z,xe,ye = np.histogram2d(pos[extr,0], pos[extr,1],bins=50, weights=data[extr],
                             density=False)
    N,xe,ye = np.histogram2d(pos[extr,0], pos[extr,1],bins=50, weights=np.ones_like(data[extr]),
                             density=False)
    R=gaussian_filter(Z,sigma=5,mode='constant')/gaussian_filter(N,sigma=5,mode='constant')
    np.place(R,np.isnan(R),[0])
    print(f"min: {np.amin(R)}, max: {np.amax(R)}, mean: {np.mean(R)}")
    plt.imshow(R, cmap=plt.get_cmap('jet'), aspect='equal',
               interpolation='gaussian')
    plt.savefig(f'{FIGDIR}/VF{TYPE}{c:03d}.png')
    plt.close()

Rebc = 625
def wall_occupancy(pos, hpr, walls):
    W=walls.shape[0]
    ebc = np.zeros((2*Rebc,2*Rebc))
    for x, y, hd in zip(pos[:, 0], pos[:, 1], hpr):
        sw = shift(walls, [-x,-y])  # shift to centre
        rsw = rotate(sw, -hd)
        ebc += rsw[(W//2-Rebc):(W//2+Rebc),(W//2-Rebc):(W//2+Rebc)]
    return ebcimport

def polar_ebc(c,pos,hpr,data):
    b,MN, MX,t = np.percentile(data,[0,10,THRESHOLD,100])
#    extr = (MN > data) | (data > MX)
    extr = (data > MX)
    fig = plt.figure(c+1, figsize=(8,8))
    wall_data =[]
    for x,y,h,d  in zip(pos[extr, 0], pos[extr, 1],hpr[extr], data[extr]):
        # print(f"raw: ({x}, {y}) -> {h} | {d}")
        t1 = ((90 - math.atan2(-y,-x)) *180 /np.pi ) % 360
        t2 = ((90 - math.atan2(1250-y,-x)) *180 /np.pi ) % 360
        t3 = ((90 - math.atan2(1250-y,1250-x)) *180 /np.pi ) % 360
        t4 = ((90 - math.atan2(-y,1250-x)) *180 /np.pi ) % 360
        rt1 = t1 if t1< t2 else t1-360
        wt1 = np.linspace(rt1,t2,round(abs(t2-rt1)/3))
        rt2 = t2 if t2< t3 else t2-360
        wt2 = np.linspace(rt2,t3,round(abs(t3-rt2)/3))
        rt3 = t3 if t3< t4 else t4-360
        wt3 = np.linspace(rt3,t4,round(abs(t4-rt3)/3))
        rt4 = t4 if t4< t1 else t4-360
        wt4 = np.linspace(rt4,t1,round(abs(t1-rt4)/3))
        # print(f"theta i: {t1} {t2} {t3} {t4}")
        r1 = -x/np.cos((90-wt1)*np.pi/180)
        r2= (1250-y)/np.sin((90-wt2)*np.pi/180)
        r3= (1250-x)/np.cos((90-wt3)*np.pi/180)
        r4 = -y/np.sin((90-wt4)*np.pi/180)
        # print(f"theta: {wt2}\n r: {r2}")
        walls =  [np.array([(t-h) % 360,r,d])
                  for r,t in zip(r1,wt1) if r <= Rebc] + \
            [np.array([(t-h) % 360,r,d])
             for r,t in zip(r2,wt2) if r <= Rebc] + \
            [np.array([(t-h) % 360,r,d])
             for r,t in zip(r3,wt3) if r <= Rebc] + \
            [np.array([(t-h) % 360,r,d]) for r,t in zip(r4,wt4) if r <= Rebc]
        wall_data += walls
    # print(f"walls: {wall_data}")
    wda = np.vstack(wall_data)
    re = np.arange(0,626,5)
    te= np.arange(0,361,3)
    Z,xe,ye = np.histogram2d(wda[:,0],wda[:,1],bins=(te,re), weights = wda[:,2],
                       density=False)
    N,xe,ye = np.histogram2d(wda[:,0],wda[:,1],bins=(te,re),
                             weights = np.ones_like(wda[:,2]),density=False)
    E=gaussian_filter(Z,sigma=5,mode='constant')#/gaussian_filter(N,sigma=5,mode='constant')
    np.place(E,np.isnan(E),[0])
    ax = plt.subplot(111,projection='polar')
    R,T = np.meshgrid(re,te)
    # im = NonUniformImage(ax,interpolation='bilinear',extent=(0,625,0,360))
    rc = (re[:-1] + re[1:])/2
    tc = (te[:-1] + te[1:])/2
    # im.set_data(tc,rc,E.T)
    # ax.images.append(im)
    ax.pcolormesh(tc,rc,E.T)
    plt.savefig(f'{FIGDIR}/PE{TYPE}{c:03d}.png')
    plt.close()

def ebr_visualise(c,pos, hpr, dhpr, data, woc, walls):
    W=walls.shape[0]
    b,MN, MX,t = np.percentile(data,[0,10,THRESHOLD,100])
#    extr = (MN > data) | (data > MX)
    extr = (data > MX)
    fig = plt.figure(c+1, figsize=(8,8))
    ebc = np.zeros((2*Rebc,2*Rebc))
    for x,y,hd,v in zip(pos[extr,0], pos[extr,1],hpr[extr],data[extr]):
        sw = shift(walls, [-x,-y])  # shift to centre
        rsw = rotate(sw, -hd)
        ebc += rsw[(W//2-Rebc):(W//2+Rebc),(W//2-Rebc):(W//2+Rebc)]*v
    ebr = gaussian_filter(ebc,sigma=25.0,mode='constant')/gaussian_filter(woc,sigma=25.0,mode='constant')        
    with np.nditer(ebr,flags=['multi_index'],op_flags=['writeonly']) as ebci:
        for x in ebci:
            if (ebci.multi_index[0]-Rebc)**2 + (ebci.multi_index[1]-Rebc)**2 >= Rebc**2:
                ebr[ebci.multi_index[0],ebci.multi_index[1]] = npm.masked

    plt.imshow(ebr, aspect='equal')
    plt.savefig(f'{FIGDIR}/EB{TYPE}{c:03d}.png')
    plt.close()
        
    
    
NPoints = 30000
if __name__ == '__main__':
    with open(f'{PATH}mazedat.npy','rb') as f:
        visdat = np.load(f)[:NPoints,:]
        hprdat = np.load(f)[:NPoints,:]
        dhprdat = np.diff(hprdat,axis=0)
        #hprdat = hprdat[1:,:]
        posdat = np.load(f)[:NPoints,:]
    if TYPE == 'both':
        POS=False
        Nim = 5
    elif TYPE=='binocvis':
        POS=True
        VX=2*VX
        visdat=visdat[:,:VX*VY]
        Nim=1
    elif TYPE=='binocflow':
        VX=2*VX
        visdat=visdat[:,VX*VY:]
        Nim=4
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
    if path.exists(MODEL_FILE):
        with open(MODEL_FILE,'rb') as fh:
            model = pickle.load(fh)
        W = model.transform(visdat)
    else:
        # model = make_pipeline(MiniBatchDictionaryLearning(n_components=DICT,
        #                                          n_jobs=-1,fit_algorithm='cd',
        #                                          transform_algorithm='lasso_cd',
        #                                                   transform_max_iter=10000,
        #                                                   positive_code=POS,verbose=True,
        #                                                   batch_size=1000,n_iter=ITER
        # ))
        model = make_pipeline(NMF(n_components=DICT,max_iter=ITER,alpha_W=REG,
                                                         verbose=True, l1_ratio=0.75,
                                  init=None, tol=1e-3))
        W=model.fit_transform(visdat)
        with open(MODEL_FILE,'wb') as fh:
            pickle.dump(model,fh)
    mkdir(FIGDIR)
    fig,ax = plt.subplots(figsize=(8,8))
    ax.plot(posdat[:,0],posdat[:,1])
    ax.plot([0,0],[0,1250],color='black')
    ax.plot([0,1250],[1250,1250],color='black')
    ax.plot([1250,1250],[1250,0],color='black')
    ax.plot([0,1250],[0,0],color='black')
    ax.set_xticks([0,625, 1250])
    ax.set_xticklabels([0,62.5,125], fontsize=18)
    ax.set_yticks([0,625,1250])
    ax.set_yticklabels([0,62.5,125], fontsize=18,rotation=90)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    #plt.show()
    plt.savefig(f'{FIGDIR}path.png')
    plt.close()
    # # plt.plot(np.diff(hprdat[1:,0]-135,prepend=0))
    # # plt.show()
    pca = model.named_steps['nmf']
    H=pca.components_
    import sys
    np.savetxt(TEXT,np.hstack((posdat[:NPoints,0:2],W)))
    # walls = np.round(plt.imread(f"{PATH}.png")[:,:,0])  # binary maze
    # #walls = np.pad(maze,(1000-maze.shape[0])//2)      # padded walls   
    # if path.exists(f'woc{PATH}{Rebc}.txt'):
    #     print('loading woc')
    #     woc = np.loadtxt(f'woc{PATH}{Rebc}.txt')
    # else:
    #     print('creating woc')
    #     woc= wall_occupancy(posdat, hprdat[:,0], walls)
    #     np.savetxt(f'woc{PATH}{Rebc}.txt',woc)
    for c in range(H.shape[0]):
        print(f'd: {c}')
        plt.figure(0, figsize=(8*round(VX/VY),8*Nim))
        plt.imshow(H[c,:].reshape((Nim*VY,VX)))
        plt.colorbar()
        plt.savefig(f'{FIGDIR}/SC{c:03d}.png')
        plt.close()
        # ax =  fig.add_subplot(111,projection='3d')
        #ax.set_zlim(-50,50)
        # ax.plot3D(posdat[1:,0],posdat[1:,1],W[:,0])
        # scd = ax.scatter(posdat[:,0],posdat[:,1],hprdat[:,0],c=-W[:,c], s=100)
        #marginal_visualise(c,posdat,hprdat[:,0],dhprdat[:,0],W[:,c])
        #extr_visualise(c,posdat,hprdat[:,0],dhprdat[:,0],W[:,c])
        angle_visualise(c,posdat,hprdat[:,0],dhprdat[:,0],W[:,c])
        #quiver_visualise(c,posdat,hprdat[:,0],dhprdat[:,0],W[:,c])
        field_visualise(c,posdat,hprdat[:,0],dhprdat[:,0],W[:,c])
        #polar_ebc(c,posdat,hprdat[:,0],W[:,c])
        # if TYPE not in ['slow','flow','binocflow']:
        #     ebr_visualise(c,posdat,hprdat[:,0],dhprdat[:,0],W[:,c], woc, walls)
        # plt.colorbar(scd, ax=ax)
        # plt.close(c)
    # plt.show(block=False)
    # input('?')
    # plt.close('all')

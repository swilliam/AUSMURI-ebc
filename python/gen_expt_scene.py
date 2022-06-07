from direct.showbase.ShowBase import ShowBase
from direct.task import Task                      # import the bits of panda
from panda3d.core import GeoMipTerrain, loadPrcFileData, PointLight


from random import random
import math
import numpy as np
from scipy.stats import rayleigh, norm
from sklearn.decomposition import NMF, PCA
import matplotlib.pyplot as plt
import cv2
from rightmaze import B,N,H,maze, MAZE_ID
from itertools import chain
from scipy.io import loadmat

Z=200
P=15 #math.atan2(H,(N-2*B)/10)*180/math.pi
pxmm = 1

v_peak = 130
yaw_sd = 340

#Nsteps = 40000
VX, VY = 170,110

TRAIL_PLOT = False

loadPrcFileData('', 'win-size {} {}'.format(VX,VY))
traj=127
MAZE_OUT = f"expt{traj:03d}widevis"
tds = loadmat('trajData',squeeze_me=True)
x = tds['root']['x'][traj]*10
y = tds['root']['y'][traj]*10
hd= tds['root']['headdir'][traj]
nan_idx = np.isnan(x)| np.isnan(y) |np.isnan(hd)
x=x[~nan_idx]
y=y[~nan_idx]
print(f"x: min {np.min(x):4.2f}, max {np.max(x):4.2f}, diff {np.max(x)-np.min(x):4.2f}")
print(f"y: min {np.min(y):4.2f}, max {np.max(y):4.2f}, diff {np.max(y)-np.min(y):4.2f}")
print(f"mean: ({np.mean(x):4.2f},{np.mean(y):4.3f})")

hd =hd[~nan_idx] 
nx = (x-np.mean(x))+N/2
ny = (y-np.mean(y))+N/2
nhd= ((hd-np.pi/4)*180/np.pi) % 360
Nsteps = nx.shape[0]-1

class MyApp(ShowBase):                          # our 'class'
    def __init__(self):
        ShowBase.__init__(self)                        # initialise
        terrain = GeoMipTerrain("ratMaze")        # create a terrain
        terrain.setHeightfield(f"{MAZE_ID}.png")        # set the height map
        terrain.setColorMap(f"c{MAZE_ID}.png")           # set the colour map
        terrain.setBruteforce(True)
        #terrain.setRoughness(1.0) # level of detail
        root = terrain.getRoot()          # maximum height
        root.reparentTo(render)                        # render from root
        root.setSz(400)
        terrain.generate() # generate
        self.vismat = []
        self.camera.setPos(nx[0], ny[0], Z)
        self.camera.setHpr(nhd[0],30,0)
        self.camLens.setFov(VX,VY)
        self.camLens.setFar(2000)
        self.disableMouse()
        self.taskMgr.add(self.moveRat, 'moveRat')
        self.prev = np.zeros((VY,VX))
        self.hpr=[]
        self.pos=[]
        if TRAIL_PLOT:
            self.ax = plt.axes()
            # set limits
            plt.xlim(0,N) 
            plt.ylim(0,N)
        #self.movie(duration=10.0)

    def moveRat(self, task):
        print(f"N: {task.frame}")
        self.camera.setPos(nx[task.frame], ny[task.frame], Z)
        self.camera.setHpr(nhd[task.frame],P,0)
        self.pos.append(self.camera.getPos() )
        self.hpr.append(self.camera.getHpr())
        #self.screenshot()
        # plight = PointLight('plight')
        # plight.setColor((1, 1, 1,0.2))
        # plnp = render.attachNewNode(plight)
        # plnp.setPos(N//2, N//2, 1000)
        # render.setLight(plnp)
        sr = self.win.getScreenshot()
        data = sr.getRamImage()
        image = np.frombuffer(data, np.uint8)  # use data.get_data() instead of data in python 2
        image.shape = (sr.getYSize(), sr.getXSize(), sr.getNumComponents())
        image = np.flipud(image)
        if task.frame <Nsteps:
            next = image[:,:,2]
            # plt.imshow(next)
            # plt.show()
            # flow = cv2.calcOpticalFlowFarneback(self.prev, next, None, 0.5 ,3, 15, 3,
            #                                     5, 1.2, 0)
            # # print('Mean flow: {}'.format(np.max(np.abs(flow))))
            # fnext = next.flatten()
            # flow_sign = flow > 0.0
            # pflow = np.zeros(flow.shape)
            # nflow = np.zeros(flow.shape)
            # pflow[flow_sign] = flow[flow_sign]
            # nflow[~flow_sign] = -flow[~flow_sign]
            # data3 = np.concatenate((next.flatten(),10*pflow[:,:,0].flatten(),
            #                         10*pflow[:,:,1].flatten(),10*nflow[:,:,0].flatten(),
                     
            #    10*nflow[:,:,1].flatten()))
            self.vismat.append(next.flatten())
            self.prev=next
            t = Task.cont
        else:
            visdat = np.vstack(self.vismat)
            hprdat = np.vstack(self.hpr)
            posdat = np.vstack(self.pos)
            with open(f"{MAZE_OUT}mazedat.npy",'wb') as f:
                np.save(f,visdat)
                np.save(f,hprdat)
                np.save(f,posdat-[B,B,Z])
            t= None
        return t
        
if __name__ == '__main__':
    app = MyApp()                                   # our 'object'
    app.run()                                       # away we go!

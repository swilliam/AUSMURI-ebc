from direct.showbase.ShowBase import ShowBase
from direct.task import Task                      # import the bits of panda
from panda3d.core import GeoMipTerrain, loadPrcFileData, PointLight


VX, VY =3*80,3*60
loadPrcFileData('', 'win-size {} {}'.format(VX,VY))
from random import random
import math
import numpy as np
from scipy.stats import rayleigh, norm
from sklearn.decomposition import NMF, PCA
import matplotlib.pyplot as plt
import cv2
from genmaze import B,N,H
Z=75
P=15 #math.atan2(H,(N-2*B)/10)*180/math.pi
pxcm = (N-2*B)/50

v_peak = 13*pxcm
yaw_sd = 340

Nsteps = 330
def edge(x,y,h,v,dh):
    print(x,y,h,v,dh,pxcm)
    if x <B+15 and h >180:     # close to west edge
        print('west')
        v -= (v-5*pxcm)/2
        h = (h+90 if h > 270 else h-90) % 360
    elif y > N-B-15 and (h >270 or h <90):  # north
        print('north')
        v -= (v-5*pxcm)/2
        h = (h+ 90 if h <90 else h-90) % 360
    elif x > N-B-15 and h<180:  # east
        print('east')
        v -= (v-5*pxcm)/2
        h = (h+ 90 if h >90 else h-90) % 360
    elif y < B+15  and (h <270 and h>90): # south
        print('south')
        v -= (v-5*pxcm)/2
        h = (h+ 90 if h >180 else h-90) % 360
    dt = globalClock.getDt()
    print(f'deltat: {dt}')
    if dt > 0.2:
        dt=0.03
    nh=(h+dh*dt)%360
    nx=x+math.cos((90-nh)/180*math.pi)*v*dt
    ny=y+math.sin((90-nh)/180*math.pi)*v*dt
    #print(nx,ny,nh)
    return nx,ny,nh
        

class MyApp(ShowBase):                          # our 'class'
    def __init__(self):
        ShowBase.__init__(self)                        # initialise
        terrain = GeoMipTerrain("ratMaze")        # create a terrain
        terrain.setHeightfield("box.png")        # set the height map
        terrain.setColorMap("cbox.png")           # set the colour map
        terrain.setBruteforce(True)
        #terrain.setRoughness(1.0) # level of detail
        root = terrain.getRoot()          # maximum height
        root.reparentTo(render)                        # render from root
        root.setSz(200)
        terrain.generate() # generate
        self.vismat = []
        self.camera.setPos(N//2, N//2, Z)
        self.camera.setHpr(0,P,0)
        self.camLens.setFov(90)
        self.disableMouse()
        self.taskMgr.add(self.moveRat, 'moveRat')
        self.prev = np.zeros((VY,VX))
        self.hpr=[]
        self.pos=[]
        #self.movie(namePrefix='ratmovie/movie',duration=10.0)

    def moveRat(self, task):
        camX = self.camera.getX()
        camY = self.camera.getY()
        camH = self.camera.getH()
        delH = norm.rvs(loc=0,scale=yaw_sd)
        new_v = rayleigh.rvs(scale=v_peak)
        V_rat = new_v if new_v >5*pxcm else 5.0*pxcm
        # print(camX)
        newX, newY, newH = edge(camX,camY,camH,V_rat,delH)
        self.camera.setPos(newX, newY, Z)
        self.camera.setHpr(newH,P,0)
        self.pos.append(self.camera.getPos())
        self.hpr.append(self.camera.getHpr())
        #self.screenshot()
        plight = PointLight('plight')
        plight.setColor((1, 1, 1,0.2))
        plnp = render.attachNewNode(plight)
        plnp.setPos(N//2, N//2, 1000)
        render.setLight(plnp)
        sr = self.win.getScreenshot()
        data = sr.getRamImage()
        image = np.frombuffer(data, np.uint8)  # use data.get_data() instead of data in python 2
        image.shape = (sr.getYSize(), sr.getXSize(), sr.getNumComponents())
        image = np.flipud(image)
        if task.frame <Nsteps:
            next = image[:,:,2]
            # plt.imshow(next)
            # plt.show()
            flow = cv2.calcOpticalFlowFarneback(self.prev, next, None, 0.5 ,3, 15, 3,
                                                5, 1.2, 0)
            # print('Mean flow: {}'.format(np.max(np.abs(flow))))
            fnext = next.flatten()
            flow_sign = flow > 0.0
            pflow = np.zeros(flow.shape)
            nflow = np.zeros(flow.shape)
            pflow[flow_sign] = flow[flow_sign]
            nflow[~flow_sign] = -flow[~flow_sign]
            print('flow')
            plt.figure(1,figsize=(16,12))
            plt.imshow(np.vstack((np.hstack((pflow[:,:,0],pflow[:,:,1])),
                                  np.hstack((nflow[:,:,0],nflow[:,:,1])))))
            plt.savefig(f"ratmovie/pflow{task.frame:03d}.png")
            plt.close()
            data3 = np.concatenate((next.flatten(),10*pflow[:,:,0].flatten(),
                                    10*pflow[:,:,1].flatten(),10*nflow[:,:,0].flatten(),
                     
               10*nflow[:,:,1].flatten()))
            self.vismat.append(data3)
            self.prev=next
            t = Task.cont
        else:
            visdat = np.vstack(self.vismat)
            hprdat = np.vstack(self.hpr)
            posdat = np.vstack(self.pos)
            with open('visnndat.npy','wb') as f:
                np.save(f,visdat)
                np.save(f,hprdat)
                np.save(f,posdat)
            t= None
        return t
        
if __name__ == '__main__':
    app = MyApp()                                   # our 'object'
    app.run()                                       # away we go!

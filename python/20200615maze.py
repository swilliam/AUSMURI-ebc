from direct.showbase.ShowBase import ShowBase
from direct.task import Task                      # import the bits of panda
from panda3d.core import GeoMipTerrain          # that we need
from panda3d.core import loadPrcFileData 
VX, VY =160, 120
loadPrcFileData('', 'win-size {} {}'.format(VX,VY))
from random import random
import math
import numpy as np
from sklearn.decomposition import NMF, PCA
import matplotlib.pyplot as plt
import cv2
from genmaze import B,N,H
Z=H//2
P=5
class MyApp(ShowBase):                          # our 'class'
    def __init__(self):
        ShowBase.__init__(self)                        # initialise
        terrain = GeoMipTerrain("ratMaze")        # create a terrain
        terrain.setHeightfield("box.png")        # set the height map
        terrain.setColorMap("box.png")           # set the colour map
        terrain.setBruteforce(True)                    # level of detail
        root = terrain.getRoot()                       # capture root
        root.reparentTo(render)                        # render from root
        root.setSz(60)                                 # maximum height
        terrain.generate() # generate
        self.dc =0
        self.vismat = []
        self.camera.setPos(N//2, N//2, Z)
        self.camera.setHpr(135,P,0)
        self.disableMouse()
        self.taskMgr.add(self.moveRat, 'moveRat')
        self.prev = np.zeros((VY,VX))
        self.hpr=[]
        self.pos=[]
        #self.movie(duration=10.0)

    def moveRat(self, task):
        camX = self.camera.getX()
        camY = self.camera.getY()
        delT = math.pi*random()/4 - math.pi/8
        delX = camX + 2*(random()-0.5)
        delY = camY + 2*(random()-0.5)
        # print(camX)
        self.camera.setPos(delX, delY, Z)
        self.camera.setHpr(self.camera.getHpr()[0]+delT*180/math.pi,P,0)
        self.pos.append(self.camera.getPos())
        self.hpr.append(self.camera.getHpr())
        self.screenshot()
        sr = self.win.getScreenshot()
        data = sr.getRamImage()
        image = np.frombuffer(data, np.uint8)  # use data.get_data() instead of data in python 2
        image.shape = (sr.getYSize(), sr.getXSize(), sr.getNumComponents())
        image = np.flipud(image)
        if self.dc<1000:
            next = image[:,:,3]
            flow = cv2.calcOpticalFlowFarneback(self.prev, next, None, 0.5 ,3, 15, 3,
                                                5, 1.2, 0)
            # print('Mean flow: {}'.format(np.max(np.abs(flow))))
            data3 = np.concatenate((next.flatten(),10*flow[:,:,0].flatten(),
                                    10*flow[:,:,1].flatten()))
            self.vismat.append(data3)
            self.dc +=1
            self.prev=next
        if task.time <40:
            t = Task.cont
        else:
            visdat = np.vstack(self.vismat)
            hprdat = np.vstack(self.hpr)
            posdat = np.vstack(self.pos)
            with open('visdat.npy','wb') as f:
                np.save(f,visdat)
                np.save(f,hprdat)
                np.save(f,posdat)
            t= None
        return t
        

app = MyApp()                                   # our 'object'
app.run()                                       # away we go!

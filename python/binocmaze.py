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
from boxmaze import B,N,H,maze, MAZE_ID
from itertools import chain, zip_longest


Z=15
P=15 #math.atan2(H,(N-2*B)/10)*180/math.pi
pxmm = (N-2*B)/500

v_peak = 130
yaw_sd = 340

Nsteps = 40000
VX, VY = 160, 80
TRAIL_PLOT = False #True
PREVIEW = False #True

loadPrcFileData('', 'win-size {} {}'.format(VX,VY))
loadPrcFileData('', 'side-by-side-stereo 1')
loadPrcFileData('', 'default-iod 5')
#loadPrcFileData('', 'default-converge 10')

def near_wall_dir(ix, iy, dx, dy):
    sdx = math.copysign(1,dx)
    sdy = math.copysign(1,dy)
    ptwx =  [(round(ix + i * sdx), iy) for i in range(30)
         if all([ix + i * sdx > 0, ix + i * sdx < N])]
    ptwy = [(ix, round(iy + i * sdy)) for i in range(30)
         if all([ iy + i * sdy > 0, iy + i * sdy < N])]
    path_to_wall = [x for x in chain.from_iterable(zip_longest( ptwx, ptwy))
                                                   if x is not None]
    #print(list(path_to_wall))
    #print(len(list(path_to_wall)))
    for x,y in path_to_wall:
        iswall = False
        wdir = np.nan #outward normal

        if maze[x,y] >=1.0:
            #print(maze[x-1:x+2,y-1:y+2])
            if maze[x+1,y] <1.0:
                wdir = 270   #west
            elif maze[x-1,y] <1.0:
                wdir = 90  #east
            elif maze[x,y-1] <1.0:
                wdir = 0  #north
            elif maze[x,y+1] <1.0:
                wdir = 180     #south
            else:
                raise 'no wall'
            iswall = True
            break
    return iswall, wdir


def map_edge(x, y, h, v, dh):
    dt = 0.03
    # nh = (h + dh * dt) % 360
    # nx = x + math.cos((90 - nh) / 180 * math.pi) * v * dt
    # ny = y + math.sin((90 - nh) / 180 * math.pi) * v * dt
    # print(f"NEW coord: ({nx},{ny}) mm, dir: {nh} v: {v}")
    ix = math.floor(x / pxmm + B)
    iy = math.floor(y / pxmm + B)
    dy = math.cos(h * math.pi / 180)
    dx = -math.sin(h * math.pi / 180)
    print(f"coord: ({x},{y}) mm, dir: {h} dh: {dh} v: {v}")
    print(f"coord: ({ix},{iy}) px, wall dir:({dx},{dy})")
    is_wall, wdir = near_wall_dir(ix,iy,dx,dy)
    if is_wall:  # wall within 2cm
        print(f"WALL! {wdir}")
        v -= (v - 50) / 2  # reduce speed
        if wdir == 0:
            hs = -1 if (h+90) % 360 < 90 else 1
        else:
            hs = (h-wdir)/abs(h-wdir)
        dh = hs * 90 / dt

    nh = (h + dh * dt) % 360
    nx = x - math.sin(nh / 180 * math.pi) * v * dt
    ny = y + math.cos(nh / 180 * math.pi) * v * dt
    print(f"FINAL coord: ({nx},{ny}) mm, dir: {nh} v: {v}")
    if any([nx < 0, nx > 500, ny < 0, ny > 500]):
        raise "escaped maze"
    print("\n")
    return nx, ny, nh

def edge(x,y,h,v,dh):
    #print(x,y,h,v,dh,pxcm)
    if x <B+15 and h >180:     # close to west edge
        print('west')
        v -= (v-50*pxmm)/2
        h = (h+90 if h > 270 else h-90) % 360
    elif y > N-B-15 and (h >270 or h <90):  # north
        print('north')
        v -= (v-50*pxmm)/2
        h = (h+ 90 if h <90 else h-90) % 360
    elif x > N-B-15 and h<180:  # east
        print('east')
        v -= (v-50*pxmm)/2
        h = (h+ 90 if h >90 else h-90) % 360
    elif y < B+15  and (h <270 and h>90): # south
        print('south')
        v -= (v-50*pxmm)/2
        h = (h+ 90 if h >180 else h-90) % 360
    nh=(h+dh*dt)%360
    nx=x-math.sin((nh)/180*math.pi)*v*dt
    ny=y+math.cos((nh)/180*math.pi)*v*dt
    #print(nx,ny,nh)
    return nx, ny, nh
        

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
        root.setSz(50)
        terrain.generate() # generate
        self.vismat = []
        leftcam = self.makeCamera(self.win,displayRegion=(0,1,0,1),
                                  camName='left',useCamera=self.cam)
        leftcam.setHpr(320,0,0)
        leftcam.setPos(-10,0,Z)
        rightcam= self.makeCamera(self.win,displayRegion=(0,1,0,1),
                                  camName='right',useCamera=None)
        rightcam.setHpr(40,0,0)
        rightcam.setPos(10,0,0)
        print(self.camList)
        leftcam.node().getLens().setFov(90)
        rightcam.node().getLens().setFov(90)
        self.camera.setPos(N//2,N//2,Z)
        self.camera.setHpr(0,P,0)
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
        camX = self.camera.getX()
        camY = self.camera.getY()
        camH = self.camera.getH()
        delH = norm.rvs(loc=0,scale=yaw_sd)
        new_v = rayleigh.rvs(scale=v_peak)
        new_v = new_v if new_v > 50 else 50
        V_rat = new_v if new_v < 350 else 350
        # print(camX)
        newX, newY, newH = map_edge(camX,camY,camH,V_rat,delH)
        if TRAIL_PLOT:
            self.ax.scatter([newX], [newY])
            plt.draw()
            plt.pause(0.5)
            if task.frame >100:
                self.ax.collections[0].remove()
        # map_edge(camX,camY,camH,V_rat,delH)
        self.camera.setPos(newX, newY, Z)
        self.camera.setHpr(newH,P,0)
        self.pos.append(self.camera.getPos() )
        self.hpr.append(self.camera.getHpr())
        #self.screenshot()
        plight = PointLight('plight')
        plight.setColor((1, 1, 1,0.2))
        plnp = render.attachNewNode(plight)
        plnp.setPos(N//2, N//2, 1000)
        render.setLight(plnp)
        # # left eye
        # self.camera.setHpr((newH-40) % 360,P,0)

        # lsr = self.win.getScreenshot()
        # ldata = lsr.getRamImage()
        # limage = np.frombuffer(ldata, np.uint8)  # use data.get_data() instead of data in python 2
        # limage.shape = (lsr.getYSize(), lsr.getXSize(), lsr.getNumComponents())
        # limage = np.flipud(limage)
        # # right eye
        # self.camera.setHpr((newH+40) % 360,P,0)

        rsr = self.win.getScreenshot()
        rdata = rsr.getRamImage()
        rimage = np.frombuffer(rdata, np.uint8)  # use data.get_data() instead of data in python 2
        rimage.shape = (VY,VX, rsr.getNumComponents())
        rimage = np.flipud(rimage)
        self.camera.setHpr(newH,P,0)
        if task.frame <Nsteps:
            next = rimage[:,:,2]
            if PREVIEW:
                ax= plt.axes()
                ax.imshow(next)
                plt.show()
            flow = cv2.calcOpticalFlowFarneback(self.prev, next, None, 0.5 ,3, 15, 3,
                                                5, 1.2, 0)
            # print('Mean flow: {}'.format(np.max(np.abs(flow))))
            fnext = next.flatten()
            flow_sign = flow > 0.0
            pflow = np.zeros(flow.shape)
            nflow = np.zeros(flow.shape)
            pflow[flow_sign] = flow[flow_sign]
            nflow[~flow_sign] = -flow[~flow_sign]
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
            with open(f"binoc{MAZE_ID}mazedat.npy",'wb') as f:
                np.save(f,visdat)
                np.save(f,hprdat)
                np.save(f,posdat)
            t= None
        return t
        
if __name__ == '__main__':
    app = MyApp()                                   # our 'object'
    app.run()                                       # away we go!


import numpy as np
import matplotlib.pyplot as plt
from  math import sqrt


MAZE_ID = "circle"
N=2049                           # size
B = 400                     # border
H = 400 # height of wall
F= 0 # height of floor
maze = np.zeros((N,N))
for x in range(N):
    for y in range(N):
        if sqrt((x-1024)**2 + (y-1024)**2) <627:
            maze[x,y] = H+F
        if sqrt((x-1024)**2 + (y-1024)**2) <625:
            maze[x,y] = F

plt.imsave(f'{MAZE_ID}.png',maze)
cmaze = np.zeros((N,N,3))
for x in range(N):
    for y in range(N):
        if sqrt((x-1024)**2 + (y-1024)**2) <627:
            cmaze[x,y] = 0.05*np.ones((1,3))
        if sqrt((x-1024)**2 + (y-1024)**2) <625:
            cmaze[x,y] = 0.2*np.ones((1,3))

plt.imsave(f'c{MAZE_ID}.png',cmaze)

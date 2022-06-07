
import numpy as np
import matplotlib.pyplot as plt


MAZE_ID = "nowalls"
N=2049                           # size
B = 400                     # border
H = 400 # height of wall
F= 200 # height of floor
maze = np.zeros((N,N))
maze[B+2:N-B-2,B+2:N-B-2] = F*np.ones((N-2*(B+2),N-2*(B+2)))
plt.imsave(f'{MAZE_ID}.png',maze)
cmaze = np.zeros((N,N,3))
cmaze[B+2:N-B-2,B+2:N-B-2,:] = 0.2*np.ones((N-2*(B+2),N-2*(B+2),3))
plt.imsave(f'c{MAZE_ID}.png',cmaze)

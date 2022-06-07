
import numpy as np
import matplotlib.pyplot as plt


MAZE_ID = "big"
N=2049                           # size
B = 400                     # border
H = 400                           # height
maze = np.zeros((N,N))
maze[B:N-B,B] = H*np.ones((1,N-2*B))
maze[B,B:N-B] = H*np.ones((1,N-2*B))
maze[N-B-1,B:N-B] = H*np.ones((1,N-2*B))
maze[B:N-B,N-B-1] = H*np.ones((1,N-2*B))
maze[B+1:N-B-1,B+1:N-B-1] = H*np.ones((N-2*(B+1),N-2*(B+1)))
maze[B:N-B,B+1] = H*np.ones((1,N-2*B))
maze[B+1,B:N-B] = H*np.ones((1,N-2*B))
maze[N-B-2,B:N-B] = H*np.ones((1,N-2*B))
maze[B:N-B,N-B-2] = H*np.ones((1,N-2*B))
maze[B+2:N-B-2,B+2:N-B-2] = np.zeros((N-2*(B+2),N-2*(B+2)))
plt.imsave(f'{MAZE_ID}.png',maze)
cmaze = np.zeros((N,N,3))
cmaze[B:N-B,B,:] = np.ones((1,N-2*B,3))
cmaze[B,B:N-B,:] = 0.05*np.ones((1,N-2*B,3))
cmaze[N-B-1,B:N-B,:] = 0.1*np.ones((1,N-2*B,3))
cmaze[B:N-B,N-B-1,:] = np.zeros((1,N-2*B,3))
cmaze[B:N-B,B+1,:] = np.ones((1,N-2*B,3))
cmaze[B+1,B:N-B,:] = 0.05*np.ones((1,N-2*B,3))
cmaze[N-B-2,B:N-B,:] = 0.05*np.ones((1,N-2*B,3))
cmaze[B:N-B,N-B-2,:] = np.zeros((1,N-2*B,3))
cmaze[B+2:N-B-2,B+2:N-B-2,:] = 0.2*np.ones((N-2*(B+2),N-2*(B+2),3))
plt.imsave(f'c{MAZE_ID}.png',cmaze)

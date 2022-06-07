import numpy as np
import matplotlib.pyplot as plt


MAZE_ID = "wall"
N=513                           # size
B = 5                         # border
H = 50                           # height
maze = np.zeros((N,N))
maze[B:N-B,B] = H*np.ones((1,N-2*B))
maze[B,B:N-B] = H*np.ones((1,N-2*B))
maze[N-B-1,B:N-B] = H*np.ones((1,N-2*B))
maze[B:N-B,N-B-1] = H*np.ones((1,N-2*B))
maze[B+1:N-B-1,B+1:N-B-1] = np.ones((N-2*(B+1),N-2*(B+1)))
maze[B:N-B,B+1] = H*np.ones((1,N-2*B))
maze[B+1,B:N-B] = H*np.ones((1,N-2*B))
maze[N-B-2,B:N-B] = H*np.ones((1,N-2*B))
maze[B:N-B,N-B-2] = H*np.ones((1,N-2*B))
maze[B+2:N-B-2,B+2:N-B-2] = np.ones((N-2*(B+2),N-2*(B+2)))/200
#wall
maze[B:(N//2-B),B+N//2] = H*np.ones((1,N//2-2*B))
maze[B:(N//2-B),B+1+N//2] = H*np.ones((1,N//2-2*B))
plt.imsave('wall.png',maze)
cmaze = np.zeros((N,N,3))
cmaze[B:N-B,B,:] = np.ones((1,N-2*B,3))
cmaze[B,B:N-B,:] = 0.05*np.ones((1,N-2*B,3))
cmaze[N-B-1,B:N-B,:] = 0.1*np.ones((1,N-2*B,3))
cmaze[B:N-B,N-B-1,:] = np.zeros((1,N-2*B,3))
cmaze[B:N-B,B+1,:] = np.ones((1,N-2*B,3))
cmaze[B+1,B:N-B,:] = 0.05*np.ones((1,N-2*B,3))
cmaze[N-B-2,B:N-B,:] = 0.1*np.ones((1,N-2*B,3))
cmaze[B:N-B,N-B-2,:] = np.zeros((1,N-2*B,3))
cmaze[B+2:N-B-2,B+2:N-B-2,:] = 0.2*np.ones((N-2*(B+2),N-2*(B+2),3))
cmaze[B:(N//2-B),B+N//2,:] = np.zeros((1,N//2-2*B,3))
cmaze[B:(N//2-B),B+N//2+1,:] = np.zeros((1,N//2-2*B,3))
plt.imsave('cwall.png',cmaze)

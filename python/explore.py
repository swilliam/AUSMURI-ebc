import math
import numpy as np
from sklearn.decomposition import NMF, PCA, DictionaryLearning
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
VX, VY =160, 120

if __name__ == '__main__':
    with open('visdat.npy','rb') as f:
        visdat = np.load(f)
        hprdat = np.load(f)
        posdat = np.load(f)
    NMFmodel = make_pipeline(DictionaryLearning(n_components=10,n_jobs=-1))
    print(NMFmodel)
    W=NMFmodel.fit_transform(visdat[1:,:])
    # plt.figure(0)
    # plt.plot(W[:,3])
    # plt.plot(np.diff(hprdat[1:,0]-135,prepend=0))
    # plt.show()
    pca = NMFmodel.named_steps['dictionarylearning']
    H=pca.components_
    for c in range(H.shape[0]):
        plt.figure(c+1)
        plt.imshow(H[c,:].reshape((3*VY,VX)))
        plt.savefig(f'SC{c:03d}.png')
    plt.show(block=False)
    input('?')
    plt.close('all')

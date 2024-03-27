from sklearn.datasets import make_swiss_roll
from scipy.spatial.distance import directed_hausdorff
import numpy as np
import torch
import scipy
import matplotlib.pyplot as plt
from tqdm import trange
std=0.2
orig,t=make_swiss_roll(5000,noise=0,random_state=0)
orig=orig[:,[0,2]]
plt.scatter(orig[:,0],orig[:,1])
plt.savefig("orig.png")
plt.close()
data=orig+std*np.random.randn(*orig.shape)
plt.scatter(data[:,0],data[:,1])
plt.savefig("noised.png")
plt.close()
np.save("orig.npy", orig)
np.save("noise.npy",data)
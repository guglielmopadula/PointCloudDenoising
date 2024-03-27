#Paper: Total Denoising: Unsupervised Learning of 3D Point Cloud Cleaning
from sklearn.datasets import make_swiss_roll
from scipy.spatial.distance import directed_hausdorff, cdist
import numpy as np
import torch
import scipy
from tqdm import trange
import matplotlib.pyplot as plt
from tqdm import trange
import torch_geometric

def k(x,r):
    return torch.exp(-torch.linalg.norm(1/r*x)**2/(2*(1/torch.sqrt(2*torch.pi))))


def sample_radius(x,y,r):
    all=torch_geometric.nn.pool.radius(x,y,r)
    return all[1]



data=np.load("noise.npy")
data=data.reshape(5000,2)
data_torch=torch.tensor(data).float()

mean_torch=torch.zeros_like(data_torch)
for i in trange(len(mean_torch)):
    tmp=data_torch[sample_radius(data_torch,data_torch[i].unsqueeze(0),0.5)]
    mean_torch[i]=torch.mean(tmp,dim=0)
with torch.no_grad():
    plt.scatter(mean_torch[:,0],mean_torch[:,1])
    plt.savefig("l2_circle_averaging.png")
    plt.close()

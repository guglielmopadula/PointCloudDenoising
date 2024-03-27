#Paper: Recorrupted-to-Recorrupted: Unsupervised Deep Learning for Image Denoising
from sklearn.datasets import make_swiss_roll
from scipy.spatial.distance import directed_hausdorff, cdist
import numpy as np
import torch
import scipy
import matplotlib.pyplot as plt
from tqdm import trange
std=0.2
orig=np.load("orig.npy")
data=np.load("noise.npy")
data=data.reshape(5000,-1,2)
matrix=std*np.eye(2)
z=np.random.randn(5000,50,2)#
alpha=1
z=z.reshape(-1,2)
z=z@matrix
z=z.reshape(-1,50,2)
data_1=data+alpha*z
data_2=data-(1/alpha)*z
dataset=torch.utils.data.TensorDataset(torch.tensor(data_1.reshape(-1,2)).float(),torch.tensor(data_2.reshape(-1,2)).float())
dataloader=torch.utils.data.DataLoader(dataset,batch_size=500)
model=torch.nn.Sequential(torch.nn.Linear(2,100),
                          torch.nn.ReLU(),
                          torch.nn.Linear(100,100),
                          torch.nn.ReLU(),
                          torch.nn.Linear(100,100),
                          torch.nn.ReLU(),
                          torch.nn.Linear(100,100),
                          torch.nn.ReLU(),
                          torch.nn.Linear(100,2))
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
for i in trange(100):
    tot_loss=0
    for data_ in dataloader:
        x,y=data_
        optimizer.zero_grad()
        pred=model(x)
        loss=torch.linalg.norm(pred-y)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            tot_loss=tot_loss+(torch.linalg.norm(pred-y).item()/torch.linalg.norm(pred).item())/len(dataloader)
    print(tot_loss)
z=np.random.randn(5000,500,2).reshape(-1,2)
z=z@matrix
z=z.reshape(5000,500,2)
z=torch.tensor(z).float()
with torch.no_grad():
    denoised=torch.mean(model(torch.tensor(data).float().reshape(5000,-1,2)+alpha*z),axis=1).numpy()
    plt.scatter(denoised[:,0],denoised[:,1])
    plt.savefig("r2r.png")
    plt.close()

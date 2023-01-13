# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 16:11:38 2022

@author: hayata
"""
# %%
import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import matplotlib.pyplot as plt

# %%
class Generator(nn.Module):
    def __init__(self,z_len):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(z_len, z_len*2),
            nn.ReLU(),
            nn.Linear(z_len*2,z_len*4),
            nn.ReLU(),
            nn.Linear(z_len*4,z_len),
            )
        self.tanh = nn.Tanh()
    def forward(self, z,):
        x = self.model(z)
        return x
    
class Discriminator(nn.Module):
    def __init__(self,z_len):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(z_len, z_len//2),
            nn.LeakyReLU(),
            nn.Linear(z_len//2, z_len//4),
            nn.LeakyReLU(),
            nn.Linear(z_len//4,1),
            nn.Sigmoid())
           
    def forward(self, z):
        x = self.model(z)
        return x
    
#%%
batch_size = 1

num_epochs = 1000
#2値交差エントロピー(バイナリクロスエントロピー)
#ylogx + (1-y)log(1-x)
criterion = torch.nn.BCELoss()
#入力ノイズの次元数
z_len = 100

netG = Generator(z_len)

netD = Discriminator(z_len)
#本物のラベル
real_label = torch.ones(batch_size,1)
#偽物のラベル
fake_label = 

optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=[0.5, 0.999])
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=[0.5, 0.999])

#%%
Y_real = torch.rand(z_len,).detach().numpy()
Y_fake = netG(torch.randn((z_len,)),).detach().numpy()
plt.figure()
plt.hist(Y_real,density=True,alpha=0.5)
plt.hist(Y_fake,density=True,alpha=0.5)
#%%
for epoch in range(num_epochs):
    #入力ノイズの分布(標準正規分布)に　#調べる
    z = 
    #本物の分布（一様分布)に従う乱数生成
    real = torch.rand(batch_size, z_len)
    
    ###D###
    #output = D(x)
    output = 
    
    #D(x) → 1 ･･･(1)
    errD_real = criterion(output, real_label)   
    #fake = G(z)
    fake = 
    #Gの固定
    fake_detach = fake.detach()
    #output = D(G(z))
    output = netD(fake_detach)
    #D(G(z)) → 0 ･･･ (2)
    errD_fake =   
    
    #(1) + (2)
    errD = 
    
    optimizerD.zero_grad()
    errD.backward()
    optimizerD.step()
    
    ###G###
    #output = D(G(z))
    output = 
    #D(G(z)) → 1
    errG = 
    
    optimizerG.zero_grad()
    errG.backward()
    optimizerG.step()
    
    print('\t epoch {} \t errD_real {:.2f} \t errD_fake {:.2f} \t errG {:.2f}'
          .format(epoch, errD_real.item(), errD_fake.item(), errG.item()))

#%%          
netG.eval()
Y_real = torch.rand(z_len,).detach().numpy()
Y_fake = netG(torch.randn((z_len,)),).detach().numpy()
plt.figure()
plt.hist(Y_real,density=True,alpha=0.5)
plt.hist(Y_fake,density=True,alpha=0.5)
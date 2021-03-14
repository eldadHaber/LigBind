import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math
import time

from src import utils

# Generate random
X = []
E = []
C = []

for i in range(32):
    N = torch.randint(20,35,(1,))
    coords   = torch.randn(N,3)
    X.append(coords)
    ind      = torch.randint(0,5,(N,))
    elements = torch.zeros(N,5)
    for j in range(N):
        elements[j,ind[j]] = 1
    E.append(elements)
    charge   = torch.randn(N)
    C.append(charge)

#input = utils.getData(coords, elements, charge)
input, Mask = utils.getBatchData(X, E, C)
output = torch.randn(32)

nNin   = 13
nopen  = 32
nhid   = 32
nlayer = 18
model = utils.resNet(nNin,nopen,nhid,nlayer)

total_params = sum(p.numel() for p in model.parameters())
print('Number of parameters ', total_params)

score = model(input)

## Start optimization
lrK = 1.0e-3
lrG = 1.0e-3
lrO = 1.0e-6
lrW = 1.0e-6


optimizer = optim.Adam([{'params': model.Kopen, 'lr': lrO},
                        {'params': model.K, 'lr': lrK},
                        {'params': model.G, 'lr': lrG},
                        {'params': model.W, 'lr': lrW}])

epochs = 10

ndata = 1 #n_data_total

for j in range(epochs):
    # Prepare the data
    aloss = 0.0

    for i in range(ndata):

        #input = utils.getData(coords, elements, charge)
        input, Mask = utils.getBatchData(X, E, C)
        optimizer.zero_grad()
        score = model(input, Mask)
        loss  = F.mse_loss(score,output)
        loss.backward()
        nk = torch.norm(model.K.grad).item()
        nw = torch.norm(model.W.grad).item()
        ng = torch.norm(model.G.grad).item()
        no = torch.norm(model.Kopen.grad).item()
        optimizer.step()

        nprnt = 1
        if i % nprnt == 0:
            print("%2d.%1d   %10.3E   %10.3E   %10.3E   %10.3E   %10.3E" % (j, i, loss.item(), nk, ng, no, nw))

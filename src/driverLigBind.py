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

# Generate fake training data
ndata = 2048
for i in range(ndata):
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

input, Mask = utils.getBatchData(X, E, C)
output = torch.log(torch.sum(input,dim=[1,2,3]))
output = output-output.mean()
output=output/torch.std(output)

# Generate fake validation data
XV = []
EV = []
CV = []
nValdata = 128
for i in range(nValdata):
    N = torch.randint(20,35,(1,))
    coords   = torch.randn(N,3)
    XV.append(coords)
    ind      = torch.randint(0,5,(N,))
    elements = torch.zeros(N,5)
    for j in range(N):
        elements[j,ind[j]] = 1
    EV.append(elements)
    charge   = torch.randn(N)
    CV.append(charge)

#input = utils.getData(coords, elements, charge)
inputV, MaskV = utils.getBatchData(XV, EV, CV)
outputV = torch.log(torch.sum(inputV,dim=[1,2,3]))
outputV = outputV-outputV.mean()
outputV =outputV/torch.std(outputV)

nNin   = 13
nopen  = 32
nhid   = 32
nlayer = 18
model = utils.resNet(nNin,nopen,nhid,nlayer)

total_params = sum(p.numel() for p in model.parameters())
print('Number of parameters ', total_params)

score = model(input)

## Start optimization
lrK = 1.0e-4
lrG = 1.0e-4
lrO = 1.0e-5
lrW = 1.0e-5


optimizer = optim.Adam([{'params': model.Kopen, 'lr': lrO},
                        {'params': model.K, 'lr': lrK},
                        {'params': model.G, 'lr': lrG},
                        {'params': model.W, 'lr': lrW}])

epochs = 100
bs     = 64
#ndata = 1 #n_data_total

for j in range(epochs):
    # Prepare the data
    aloss = 0.0

    for i in range(ndata//bs):

        #input = utils.getData(coords, elements, charge)
        b = i*bs
        input, Mask = utils.getBatchData(X[b:b+bs], E[b:b+bs], C[b:b+bs])
        scoreObs   = output[b:b+bs]
        optimizer.zero_grad()
        score = model(input, Mask)
        loss  = F.mse_loss(score,scoreObs)
        loss.backward()
        nk = torch.norm(model.K.grad).item()
        nw = torch.norm(model.W.grad).item()
        ng = torch.norm(model.G.grad).item()
        no = torch.norm(model.Kopen.grad).item()
        optimizer.step()

        nprnt = 1
        if i % nprnt == 0:
            print("%2d.%1d   %10.3E   %10.3E   %10.3E   %10.3E   %10.3E" % (j, i, loss.item(), nk, ng, no, nw))

    # Validation error
    inputV, MaskV = utils.getBatchData(XV, EV, CV)
    scoreObsV = outputV
    scoreV = model(inputV, MaskV)
    lossV = F.mse_loss(scoreV, scoreObsV)

    print(" ======== %2d   %10.3E ============== " % (j, lossV.item()))

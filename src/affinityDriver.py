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

# Generate fake leg data
ndata = 128
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

# Generate fake res data
ndata = 128
XR = []
ER = []
for i in range(ndata):
    N = torch.randint(70,80,(1,))
    coords   = torch.randn(N,3)
    XR.append(coords)
    ind      = torch.randint(0,10,(N,))
    elements = torch.zeros(N,10)
    for j in range(N):
        elements[j,ind[j]] = 1
    ER.append(elements)


inputL, MaskL = utils.getBatchData(X, E, C)
inputR, MaskR = utils.getBatchData(XR, ER)

output = torch.log(torch.sum(inputL,dim=[1,2,3]))
output = output-output.mean()
output=output/torch.std(output)



nNinLeg = 13
nNinRes = 21
nopen   = 25
nhid    = 25
nout    = 16
nlayer  = 4
modelLeg = utils.resNet(nNinLeg,nopen,nhid,nout, nlayer)
modelRes = utils.resNet(nNinRes,nopen,nhid,nout, nlayer)

total_params = sum(p.numel() for p in modelLeg.parameters())
print('Number of parameters ', total_params)

#fLeg = modelLeg(inputL)
#fRes = modelRes(inputR)
#score = torch.diag(fLeg.t()@fRes)


## Start optimization
lrK = 1.0e-4
lrG = 1.0e-4
lrO = 1.0e-5
lrW = 1.0e-5


optimizer = optim.Adam([{'params': modelLeg.Kopen, 'lr': lrO},
                        {'params': modelLeg.K, 'lr': lrK},
                        {'params': modelLeg.G, 'lr': lrG},
                        {'params': modelLeg.W, 'lr': lrW},
                        {'params': modelRes.Kopen, 'lr': lrO},
                        {'params': modelRes.K, 'lr': lrK},
                        {'params': modelRes.G, 'lr': lrG},
                        {'params': modelRes.W, 'lr': lrW}])

epochs = 100
bs     = 4
#ndata = 1 #n_data_total

for j in range(epochs):
    # Prepare the data

    for i in range(ndata//bs):

        #input = utils.getData(coords, elements, charge)
        b = i*bs
        inputL, MaskL = utils.getBatchData(X[b:b+bs], E[b:b+bs], C[b:b+bs])
        inputR, MaskR = utils.getBatchData(XR[b:b+bs], ER[b:b+bs])
        scoreObs   = output[b:b+bs]

        optimizer.zero_grad()
        # run the networks forward
        fLeg = modelLeg(inputL)
        fRes = modelRes(inputR)
        score = torch.diag(fLeg@fRes.t())

        loss  = F.mse_loss(score,scoreObs)
        loss.backward()
        nk = torch.norm(modelRes.K.grad).item()
        nw = torch.norm(modelRes.W.grad).item()
        ng = torch.norm(modelRes.G.grad).item()
        no = torch.norm(modelRes.Kopen.grad).item()
        optimizer.step()

        nprnt = 1
        if i % nprnt == 0:
            print("%2d.%1d   %10.3E   %10.3E   %10.3E   %10.3E   %10.3E" % (j, i, loss.item(), nk, ng, no, nw))

import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math
import torch.autograd.profiler as profiler

from src import graphOps as GO
from src import utils
from src import graphNet as GN

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

lig = torch.load('../sampleData/training_ligand_score.pt')
pro = torch.load('../sampleData/training_pocket_score.pt')

def getLigData(lig,i):
    I = (lig['bonds'][i][:,0]-1).long()
    J = (lig['bonds'][i][:,1]-1).long()

    #xe = torch.tensor(lig['bonds'][i][:,2], dtype=torch.long)
    xe = lig['bonds'][i][:, 2].long()
    xe = F.one_hot(xe,8)
    xe = xe.t().unsqueeze(0)

    xn = lig['atom_types'][i]
    cn = lig['charge'][i].unsqueeze(1)

    xn = torch.cat((xn,50*cn),dim=1)
    xn = xn.t().unsqueeze(0)

    return I, J, xn, xe.float()

def getPocketData(P,i):

    score = torch.tensor([float(P['scores'][i])])

    X = P['coords'][i]
    D = utils.distanceMap(X)
    D = torch.exp(-D/torch.max(D))
    D[D<0.9] = 0
    I, J = torch.nonzero(D,  as_tuple=True)

    xn = P['atom_types'][i]
    xn = xn.t()

    xe = D[I, J]
    return I, J, xn.unsqueeze(0), xe.unsqueeze(0).unsqueeze(0), score




# Setup the network for ligand and its parameters
nNin = 55
nEin = 8
nNopen = 16
nEopen = 16
nEhid = 16
nNclose = 16
nEclose = 1
nlayer = 18

modelL = GN.graphNetwork(nNin, nEin, nNopen, nEhid, nNclose, nlayer, h=.1)
modelL.to(device)

total_params = sum(p.numel() for p in modelL.parameters())
print('Number of parameters  for ligand', total_params)

IL, JL, xnL, xeL = getLigData(lig,5)
nNodesL = xnL.shape[2]
GL = GO.graph(IL, JL, nNodesL)

xnOutL, xeOutL = modelL(xnL, xeL, GL)


# network for the protein
# Setup the network for ligand and its parameters
nNin = 43
nEin = 1
nNopen = 16
nEopen = 16
nEhid = 16
nNclose = 16
nEclose = 1
nlayer = 6


modelP = GN.graphNetwork(nNin, nEin, nNopen, nEhid, nNclose, nlayer, h=.1)
modelP.to(device)

total_params = sum(p.numel() for p in modelP.parameters())
print('Number of parameters  for pocket', total_params)

IP, JP, xnP, xeP, score = getPocketData(pro,5)
nNodesP = xnP.shape[2]
GP = GO.graph(IP, JP, nNodesP)

xnOutP, xeOutP = modelP(xnP, xeP, GP)

bindingScore = torch.dot(torch.mean(xnOutP,dim=2).squeeze(),torch.mean(xnOutL,dim=2).squeeze())


#### Start Training ####
lrO = 1e-3
lrC = 1e-3
lrN = 1e-3
lrE1 = 1e-3
lrE2 = 1e-3

optimizer = optim.Adam([{'params': modelP.K1Nopen, 'lr': lrO},
                        {'params': modelP.K2Nopen, 'lr': lrC},
                        {'params': modelP.K1Eopen, 'lr': lrO},
                        {'params': modelP.K2Eopen, 'lr': lrC},
                        {'params': modelP.KE1, 'lr': lrE1},
                        {'params': modelP.KE2, 'lr': lrE2},
                        {'params': modelP.KN1, 'lr': lrE1},
                        {'params': modelP.KN2, 'lr': lrE2},
                        {'params': modelP.KNclose, 'lr': lrE2},
                        {'params': modelL.K1Nopen, 'lr': lrO},
                        {'params': modelL.K2Nopen, 'lr': lrC},
                        {'params': modelL.K1Eopen, 'lr': lrO},
                        {'params': modelL.K2Eopen, 'lr': lrC},
                        {'params': modelL.KE1, 'lr': lrE1},
                        {'params': modelL.KE2, 'lr': lrE2},
                        {'params': modelL.KN1, 'lr': lrE1},
                        {'params': modelL.KN2, 'lr': lrE2},
                        {'params': modelL.KNclose, 'lr': lrE2}])


epochs = 5

ndata = 4
hist = torch.zeros(epochs)

for j in range(epochs):
    # Prepare the data
    aloss = 0.0
    for i in range(ndata):

        # Get the lig data
        IL, JL, xnL, xeL = getLigData(lig, i)
        nNodesL = xnL.shape[2]
        GL = GO.graph(IL, JL, nNodesL)
        # Get the pro data
        IP, JP, xnP, xeP, trueScore = getPocketData(pro, i)
        nNodesP = xnP.shape[2]
        GP = GO.graph(IP, JP, nNodesP)

        optimizer.zero_grad()
        xnOutL, xeOutL = modelL(xnL, xeL, GL)
        xnOutP, xeOutP = modelP(xnP, xeP, GP)

        predScore = torch.dot(torch.mean(xnOutP, dim=2).squeeze(), torch.mean(xnOutL, dim=2).squeeze())

        loss = F.mse_loss(predScore.unsqueeze(0), trueScore)

        optimizer.zero_grad()
        loss.backward()

        aloss += loss.detach()
        optimizer.step()
        # scheduler.step()
        nprnt = 1
        if (i + 1) % nprnt == 0:
            aloss = aloss / nprnt
            print("%2d.%1d   %10.3E" % (j, i, aloss), flush=True)
            aloss = 0.0

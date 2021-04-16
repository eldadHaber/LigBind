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

def getLigData(lig,IND):

    n = len(IND)
    II = torch.zeros(0)
    JJ = torch.zeros(0)
    XE = torch.zeros(0,8)
    XN = torch.zeros(0,55)
    NL = torch.zeros(n, dtype=torch.long)
    cnt = 0
    for j in range(n):

        i = IND[j]
        I = (lig['bonds'][i][:,0]-1).long()
        J = (lig['bonds'][i][:,1]-1).long()
        II = torch.cat((II,I+cnt))
        JJ = torch.cat((JJ,J+cnt))

        #xe = torch.tensor(lig['bonds'][i][:,2], dtype=torch.long)
        xe = lig['bonds'][i][:, 2].long()
        xe = F.one_hot(xe,8)
        XE = torch.cat((XE,xe),dim=0)

        xn = lig['atom_types'][i]
        cn = lig['charge'][i].unsqueeze(1)

        xn = torch.cat((xn,50*cn),dim=1)
        XN = torch.cat((XN, xn), dim=0)
        NL[j] = xn.shape[0]
        cnt = cnt + xn.shape[0]

    XE = XE.t().unsqueeze(0)
    XN = XN.t().unsqueeze(0)

    return II, JJ, XN, XE.float(), NL

def getPocketData(P,IND):

    n = len(IND)
    II = torch.zeros(0)
    JJ = torch.zeros(0)
    XE = torch.zeros(0)
    XN = torch.zeros(0,43)

    score = torch.zeros(n)
    NP    = torch.zeros(n, dtype=torch.long)
    cnt = 0
    for j in range(n):
        i = IND[j]
        score[j] = torch.tensor([float(P['scores'][i])])

        X = P['coords'][i]
        D = utils.distanceMap(X)
        D = D / D.std()
        D = torch.exp(-2 * D)

        # Choose k=6 neiboughrs
        nsparse = 6
        vals, indices = torch.topk(D, k=min(nsparse, D.shape[0]), dim=1)
        nd = D.shape[0]
        I = torch.ger(torch.arange(nd), torch.ones(nsparse, dtype=torch.long))
        I = I.view(-1)
        J = indices.view(-1).type(torch.LongTensor)

        II = torch.cat((II,I+cnt))
        JJ = torch.cat((JJ,J+cnt))

        xn = P['atom_types'][i]
        XN = torch.cat((XN,xn),dim=0)
        xe = D[I, J]
        XE = torch.cat((XE,xe))

        NP[j] = xn.shape[0]
        cnt = cnt + xn.shape[0]

    XN = XN.t()

    return II, JJ, XN.unsqueeze(0), XE.unsqueeze(0).unsqueeze(0), NP, score

def computeScore(XNOutP, XNOutL, NP, NL):

    cnt = 0
    n = len(NL)
    compScore = torch.zeros(n)
    for i in range(n):
        xnOutP = XNOutP[:,:,cnt:cnt+NP[i]]
        xnOutL = XNOutL[:,:,cnt:cnt+NL[i]]
        bindingScore = torch.dot(torch.mean(xnOutP, dim=2).squeeze(), torch.mean(xnOutL, dim=2).squeeze())
        compScore[i] = bindingScore

    return compScore

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

#IL, JL, xnL, xeL = getLigData(lig,5)
#nNodesL = xnL.shape[2]
#GL = GO.graph(IL, JL, nNodesL)
#xnOutL, xeOutL = modelL(xnL, xeL, GL)


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

#IP, JP, xnP, xeP, score = getPocketData(pro,5)
#nNodesP = xnP.shape[2]
#GP = GO.graph(IP, JP, nNodesP)
#xnOutP, xeOutP = modelP(xnP, xeP, GP)
#bindingScore = torch.dot(torch.mean(xnOutP,dim=2).squeeze(),torch.mean(xnOutL,dim=2).squeeze())


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

ndata = 256
hist = torch.zeros(epochs)

batchSize = 64
for j in range(epochs):
    # Prepare the data
    aloss = 0.0
    for i in range(ndata//batchSize):

        IND = torch.arange(i*batchSize,(i+1)*batchSize)
        # Get the lig data
        IL, JL, xnL, xeL, NL = getLigData(lig, IND)
        nNodesL = xnL.shape[2]
        GL = GO.graph(IL, JL, nNodesL)
        # Get the pro data
        IP, JP, xnP, xeP, NP, trueScore = getPocketData(pro, IND)
        nNodesP = xnP.shape[2]
        GP = GO.graph(IP, JP, nNodesP)

        optimizer.zero_grad()
        xnOutL, xeOutL = modelL(xnL, xeL, GL)
        xnOutP, xeOutP = modelP(xnP, xeP, GP)

        #predScore = torch.dot(torch.mean(xnOutP, dim=2).squeeze(), torch.mean(xnOutL, dim=2).squeeze())
        predScore = computeScore(xnOutP, xnOutL, NP, NL)
        loss = F.mse_loss(predScore, trueScore)

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

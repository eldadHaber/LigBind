import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math
import torch.autograd.profiler as profiler

import graphOps as GO
import utils
import graphNet as GN
#from src import graphOps as GO
#from src import utils
#from src import graphNet as GN

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

lig = torch.load('../../../data/mpro_docking_data/mpro_lig_select.pt')
pro = torch.load('../../../data/mpro_docking_data/mpro_rec_select.pt')


def getLigData(lig,IND):

    #X      = lig['coords']
    #atype  = lig['atom_types']
    #charge = lig['charge']
    #score  = lig['score']
    #dof    = lig['torsdof']
    #atomc  = lig['atom_connect']
    #btype  = lig['bond_type']

    n = len(IND)
    II = torch.zeros(0).long()
    JJ = torch.zeros(0).long()
    XE = torch.zeros(0,6)
    XN = torch.zeros(0,19)
    NL = torch.zeros(n, dtype=torch.long)
    score = torch.zeros(n)
    cnt = 0
    for j in range(n):

        i = IND[j]
        score[j] = torch.tensor([float(lig['scores'][i])])

        I = (lig['atom_connect'][i][:,0]-1).long()
        J = (lig['atom_connect'][i][:,1]-1).long()
        #I = torch.cat((I,J))
        #J = torch.cat((J,I))
        #N = len(lig['coords'][i])
        #temp = torch.arange(N)
        #I = torch.cat((I, temp))
        #J = torch.cat((J, temp))
        kk = torch.tensor([J.max(),I.max()]).max()+1
        S  = torch.eye(kk,kk)
        S[I,J] = 1
        S = S+S.t()
        I,J = torch.nonzero(S,as_tuple=True)

        II = torch.cat((II,I+cnt))
        JJ = torch.cat((JJ,J+cnt))

        #xe = torch.tensor(lig['bonds'][i][:,2], dtype=torch.long)
        #xe = lig['bond_type'][i].long()
        #oo = torch.zeros(N,6)
        #xe = torch.cat((xe,xe,oo),dim=0)
        n  = I.shape[0]
        xe = torch.zeros(n,6)

        XE = torch.cat((XE,xe),dim=0)

        xn = lig['atom_types'][i]
        cn = lig['charges'][i].unsqueeze(1)

        xn = torch.cat((xn,5*cn),dim=1)
        XN = torch.cat((XN, xn), dim=0)
        NL[j] = xn.shape[0]
        cnt = cnt + xn.shape[0]

    XE = XE.t().unsqueeze(0)
    XN = XN.t().unsqueeze(0)

    return II, JJ, XN, XE.float(), score, NL

#II, JJ, XN, XE, score, NL = getLigData(lig,[1])


def getPocketData(P,IND):

    #X      = lig['coords']
    #atype  = lig['atom_types']
    #charge = lig['charge']
    #atomc  = lig['atom_connect']
    #btype  = lig['bond_type']
    #
    n = len(IND)
    II = torch.zeros(0)
    JJ = torch.zeros(0)
    XE = torch.zeros(0)
    XN = torch.zeros(0,18)

    score = torch.zeros(n)
    NP    = torch.zeros(n, dtype=torch.long)
    cnt = 0
    for j in range(n):
        i = IND[j]

        #I = (lig['atom_connect'][i][:,0]-1).long()
        #J = (lig['atom_connect'][i][:,1]-1).long()
        #II = torch.cat((II,I+cnt))
        #JJ = torch.cat((JJ,J+cnt))

        X = P['coords'][i]
        D = utils.distanceMap(X)
        D = D / D.std()
        D = torch.exp(-2 * D)

        # Choose k=11 neiboughrs
        nsparse = 32
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

        cn = P['charges'][i].unsqueeze(1)
        xn = torch.cat((xn, 5*cn), dim=1)

        NP[j] = xn.shape[0]
        cnt = cnt + xn.shape[0]

    XN = XN.t()

    IJ1 = torch.cat((II, JJ)).unsqueeze(1)
    IJ2 = torch.cat((JJ, II)).unsqueeze(1)
    IJ = torch.cat((IJ1, IJ2), dim=1)
    IJ = IJ.unique(dim=0)
    II = IJ[:,0]
    JJ = IJ[:,1]
    return II.long(), JJ.long(), XN.unsqueeze(0), XE.unsqueeze(0).unsqueeze(0), NP

#II, JJ, XN, XE, NP = getPocketData(pro,[1])


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

nNin    = 19
nopen   = 32
nhid    = 32
nNclose = 32
nlayer  = 12

modelL = GN.graphNetwork(nNin, nopen, nNclose, nlayer)
modelL.to(device)


total_params = sum(p.numel() for p in modelL.parameters())
print('Number of parameters  for ligand', total_params)


IL, JL, xnL, xeL, score, NL = getLigData(lig,[1,2])
nNodesL = xnL.shape[2]
GL = GO.graph(IL, JL, nNodesL)
xnOutL = modelL(xnL, xeL, GL)


# network for the protein
# Setup the network for ligand and its parameters
nNin = 18
nopen = 32
nhid  = 32
nNclose = 32
nlayer = 6

modelP = GN.graphNetwork(nNin, nopen, nNclose, nlayer)
modelP.to(device)

total_params = sum(p.numel() for p in modelP.parameters())
print('Number of parameters  for pocket', total_params)

IP, JP, xnP, xeP, NP = getPocketData(pro,[0,1])
nNodesP = xnP.shape[2]
GP = GO.graph(IP, JP, nNodesP)
xnOutP = modelP(xnP, xeP, GP)

s = computeScore(xnOutP, xnOutL, NP, NL)

#### Start Training ####
lrO = 1e-3
lrC = 1e-3
lrN = 1e-3
lrE1 = 1e-3
lrE2 = 1e-3

optimizer = optim.Adam([{'params': modelP.K1Nopen, 'lr': lrO},
                        {'params': modelP.K2Nopen, 'lr': lrC},
                        {'params': modelP.KE1, 'lr': lrE1},
                        {'params': modelP.KE2, 'lr': lrE2},
                        {'params': modelP.KNclose, 'lr': lrE2},
                        {'params': modelP.Kw, 'lr': lrE2},
                        {'params': modelL.K1Nopen, 'lr': lrO},
                        {'params': modelL.K2Nopen, 'lr': lrC},
                        {'params': modelL.KE1, 'lr': lrE1},
                        {'params': modelL.KE2, 'lr': lrE2},
                        {'params': modelL.KNclose, 'lr': lrE2},
                        {'params': modelL.Kw, 'lr': lrE2}])


epochs = 50

ndata = 20
hist = torch.zeros(epochs)

batchSize = 4
for j in range(epochs):
    # Prepare the data
    aloss = 0.0
    for i in range(ndata//batchSize):

        IND = torch.arange(i*batchSize,(i+1)*batchSize)
        # Get the lig data
        IL, JL, xnL, xeL, truescore,  NL = getLigData(lig, IND)
        nNodesL = xnL.shape[2]
        GL = GO.graph(IL, JL, nNodesL)
        # Get the pro data
        IP, JP, xnP, xeP, NP = getPocketData(pro, IND)
        nNodesP = xnP.shape[2]
        GP = GO.graph(IP, JP, nNodesP)

        optimizer.zero_grad()
        xnOutL = modelL(xnL, xeL, GL)
        xnOutP = modelP(xnP, xeP, GP)

        #predScore = torch.dot(torch.mean(xnOutP, dim=2).squeeze(), torch.mean(xnOutL, dim=2).squeeze())
        predScore = computeScore(xnOutP, xnOutL, NP, NL)
        loss = F.mse_loss(predScore, truescore)/F.mse_loss(truescore*0, truescore)

        optimizer.zero_grad()
        loss.backward()

        gC  = modelP.KNclose.grad.norm().item()
        gE1 = modelP.KE1.grad.norm().item()
        gE2 = modelP.KE2.grad.norm().item()
        gO1 = modelP.K1Nopen.grad.norm().item()
        gO2 = modelP.K2Nopen.grad.norm().item()
        gw  = modelP.Kw.grad.norm().item()


        aloss += loss.detach()
        optimizer.step()
        # scheduler.step()
        nprnt = 1
        if (i + 1) % nprnt == 0:
            aloss = aloss / nprnt
            print("%2d.%1d   %10.3E   %10.3E   %10.3E   %10.3E   %10.3E   %10.3E   %10.3E"
                  % (j, i, aloss, gC, gE1, gE2, gO1, gO2, gw), flush=True)
            aloss = 0.0

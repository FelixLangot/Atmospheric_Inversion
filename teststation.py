#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 14:13:47 2020

@author: felixlangot
"""

import numpy as np
import matplotlib.pyplot as plt 

# Import data

BRWobs = np.loadtxt('BRW_obs.txt')
MHDobs = np.loadtxt('BRW_obs.txt')
MLOobs = np.loadtxt('BRW_obs.txt')
NWRobs = np.loadtxt('BRW_obs.txt')
SMOobs = np.loadtxt('BRW_obs.txt')
SPOobs = np.loadtxt('BRW_obs.txt')

ame_obs       = np.loadtxt('cfc11.ame.txt')
afr_obs       = np.loadtxt('cfc11.afr.txt')
tropasia_obs  = np.loadtxt('cfc11.tropasia.txt')
northasia_obs = np.loadtxt('cfc11.northasia.txt')
eur_obs       = np.loadtxt('cfc11.eur.txt')
strato_obs    = np.loadtxt('cfc11.strato.txt')

BRWres = np.loadtxt('BRW_mod.txt')
MHDres = np.loadtxt('MHD_mod.txt')
MLOres = np.loadtxt('MLO_mod.txt')
NWRres = np.loadtxt('NWR_mod.txt')
SMOres = np.loadtxt('SMO_mod.txt')
SPOres = np.loadtxt('SPO_mod.txt')

# Create y0

y0 = np.zeros((1,168))

BRW_conc = BRWobs[:,1]
MHD_conc = MHDobs[:,1]
MLO_conc = MLOobs[:,1]
NWR_conc = NWRobs[:,1]
SMO_conc = SMOobs[:,1]
SPO_conc = SPOobs[:,1]

y0 = np.array([[BRW_conc], [MHD_conc], [MLO_conc], [NWR_conc], [SMO_conc], [SPO_conc]])
y0 = np.ravel(y0)

# Create R

BRW_err = BRWobs[:,2]
MHD_err = MHDobs[:,2]
MLO_err = MLOobs[:,2]
NWR_err = NWRobs[:,2]
SMO_err = SMOobs[:,2]
SPO_err = SPOobs[:,2]

y_err = np.array([[BRW_err], [MHD_err], [MLO_err], [NWR_err], [SMO_err], [SPO_err]])
y_err = np.ravel(y_err)

R = np.identity(168)*y_err
# R = 2*R                                                Test chgt d'erreur obs
R = R*R

# Create x_b

C0     = 140

ame_fx = ame_obs[:,1]
afr_fx = afr_obs[:,1]
tropasia_fx = tropasia_obs[:,1]
northasia_fx = northasia_obs[:,1]
eur_fx = eur_obs[:,1]
strato_fx = strato_obs[:,1]

x_b = np.array([[ame_fx], [afr_fx], [tropasia_fx],
               [northasia_fx], [eur_fx], [strato_fx]])
x_b = np.ravel(x_b)

x_b = np.append(x_b,C0)


# Create B

C0_err = 50

ame_err = ame_obs[:, 2]
afr_err = afr_obs[:, 2]
tropasia_err = tropasia_obs[:, 2]
northasia_err = northasia_obs[:, 2]
eur_err = eur_obs[:, 2]
strato_err = strato_obs[:, 2]

xb_err = np.array([[ame_err], [afr_err], [tropasia_err],
                  [northasia_err], [eur_err], [strato_err]])
xb_err = np.ravel(xb_err)
xb_err = np.append(xb_err, C0_err)

B = np.identity(169)*xb_err
# B = 2*B                                           Test chgt d'erreur a priori
B = B*B

# Create H

BRWresi = BRWres[:, 1:7]
MHDresi = MHDres[:, 1:7]
MLOresi = MLOres[:, 1:7]
NWRresi = NWRres[:, 1:7]
SMOresi = SMOres[:, 1:7]
SPOresi = SPOres[:, 1:7]

h0 = np.array([[BRWresi], [MHDresi], [MLOresi],
                       [NWRresi], [SMOresi], [SPOresi]])
h0 = np.ravel(h0)
h0 = np.reshape(h0,(168,6))


# Build H
'''
h1 = np.zeros((28,28))
h1[:,0] = h0[0:28,0]
np.ravel(h1)
print(np.shape(h1))
print(h1)
for i in np.arange(0,28):
    print(i)
    for j in np.arange(0,28):
        if i == j:
            h1[i,j] = h1[0,0]

  

print(h1)

'''
region=6
nyr=28
station=6

H = np.zeros((station*nyr,region*nyr+1), np.float)
 
for ireg in range(region):
   for iyr in range(nyr):
       for istat in range(station):
           deb = istat*28+iyr
           fin = (istat+1)*28
           H[deb:fin,ireg*28+iyr] = h0[istat*28:istat*28+28-iyr,ireg]
H[:,region*nyr]=1
       

H_trans = np.transpose(H)
R_inverse = np.linalg.inv(R)
B_inverse = np.linalg.inv(B)    
         
A1 = np.matmul(H_trans,R_inverse)
A2 = np.matmul(A1,H)
A = np.linalg.inv(A2+B_inverse)

xa1 = y0-np.matmul(H,x_b)
xa2 = np.matmul(A,H_trans)
xa3 = np.matmul(xa2,R_inverse)
xa4 = np.matmul(xa3,xa1)
x_a = x_b+xa4


# Emissions globales

GlobEma = np.zeros(nyr)

for i in range(nyr):
    Globyrsuma = 0
    Emyra = []
    for j in np.arange(i,(station-1)*nyr,nyr):
        Emyra.append(x_a[j])
    Globyra = sum(Emyra)
    GlobEma[i] = Globyra
    
GlobEmb = np.zeros(nyr)

for i in range(nyr):
    Globyrsumb = 0
    Emyrb = []
    for j in np.arange(i,(station-1)*nyr,nyr):
        Emyrb.append(x_b[j])
    Globyrb = sum(Emyrb)
    GlobEmb[i] = Globyrb
    
yrs = np.zeros(28)  
for j in np.arange(0,28,1):
  yrs[j]=j
        
yrs = yrs + 1979
    
plt.scatter(yrs,GlobEma)
plt.scatter(yrs,GlobEmb)
plt.show()

# 3 Emissions par regions
PriorAmerique = x_b[0:28]
PriorAfrique = x_b[28:56]
PriorAsieTrop = x_b[56:84]
PriorAsieNord = x_b[84:112]
PriorEurope = x_b[112:140]

PostAmerique = x_a[0:28]
PostAfrique = x_a[28:56]
PostAsieTrop = x_a[56:84]
PostAsieNord = x_a[84:112]
PostEurope = x_a[112:140]

plt.plot(yrs,PriorAmerique)
plt.plot(yrs,PriorAfrique)
plt.plot(yrs,PriorAsieTrop)
plt.plot(yrs,PriorAsieNord)
plt.plot(yrs,PriorEurope)
plt.show()

plt.plot(yrs,PostAmerique)
plt.plot(yrs,PostAfrique)
plt.plot(yrs,PostAsieTrop)
plt.plot(yrs,PostAsieNord)
plt.plot(yrs,PostEurope)
plt.show()

# 4 test erreurs
errprior = np.sqrt(np.diag(B))
errpost = np.sqrt(np.diag(A))

print(np.mean(errpost/errprior))

# 5 tests de config 

# 2x err err /2=>change R&B
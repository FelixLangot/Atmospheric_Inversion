# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 16:41:22 2020

@author: anouck
"""

import numpy as np
import matplotlib.pyplot as plt
#import pandas as pd
import sklearn.metrics


#création du vecteur d'observation y0
BRW_obs = np.loadtxt('BRW_obs.txt')
MHD_obs = np.loadtxt('MHD_obs.txt')
NWR_obs = np.loadtxt('NWR_obs.txt')
SMO_obs = np.loadtxt('SMO_obs.txt')
SPO_obs = np.loadtxt('SPO_obs.txt')

BRW_obs_conc = BRW_obs[:,1]
MHD_obs_conc = MHD_obs[:,1]
NWR_obs_conc = NWR_obs[:,1]
SMO_obs_conc = SMO_obs[:,1]
SPO_obs_conc = SPO_obs[:,1]


y0 = np.array([[BRW_obs_conc],
               [MHD_obs_conc], 
               [NWR_obs_conc], 
               [SMO_obs_conc], 
               [SPO_obs_conc]])

yo = np.ravel(y0)

#création de la matrice R

BRW_obs_err = BRW_obs[:,2]
MHD_obs_err = MHD_obs[:,2]
NWR_obs_err = NWR_obs[:,2]
SMO_obs_err = SMO_obs[:,2]
SPO_obs_err = SPO_obs[:,2]

y_erreur = np.array([[BRW_obs_err],  
               [MHD_obs_err], 
               [NWR_obs_err], 
               [SMO_obs_err], 
               [SPO_obs_err]])

y_erreur = np.ravel(y_erreur)

R = np.eye(140)

R = R*y_erreur
#R = R*2 
#R = R/2    
R = R*R

#création du vecteur d'état xb

ame = np.loadtxt('cfc11.ame.txt')
afr = np.loadtxt('cfc11.afr.txt')
tropasia = np.loadtxt('cfc11.tropasia.txt')
northasia = np.loadtxt('cfc11.northasia.txt')
eur = np.loadtxt('cfc11.eur.txt')
strato = np.loadtxt('cfc11.strato.txt')

ame_flux = ame[:,1]
afr_flux = afr[:,1]
tropasia_flux = tropasia[:,1]
northasia_flux = northasia[:,1]
eur_flux = eur[:,1]
strato_flux = strato[:,1]

xb = np.array([[ame_flux], 
               [afr_flux], 
               [tropasia_flux], 
               [northasia_flux], 
               [eur_flux], 
               [strato_flux]])
xb = np.ravel(xb)
xb = np.append(xb,140)

#création du vecteur B

ame_flux_err = ame[:,2]
afr_flux_err = afr[:,2]
tropasia_flux_err = tropasia[:,2]
northasia_flux_err = northasia[:,2]
eur_flux_err = eur[:,2]
strato_flux_err = strato[:,2]

xb_erreur = np.array([[ame_flux_err], 
               [afr_flux_err], 
               [tropasia_flux_err], 
               [northasia_flux_err], 
               [eur_flux_err], 
               [strato_flux_err]])

xb_erreur=np.ravel(xb_erreur)
xb_erreur = np.append(xb_erreur,50)

B = np.eye(169)
B =  xb_erreur*B
#B = B*2
B = B/2
B = B*B

#création de H

BRW_mod = np.loadtxt('BRW_mod.txt')
MHD_mod = np.loadtxt('MHD_mod.txt')
NWR_mod = np.loadtxt('NWR_mod.txt')
SMO_mod = np.loadtxt('SMO_mod.txt')
SPO_mod = np.loadtxt('SPO_mod.txt')

BRW_mod_rep = BRW_mod[:,1:7]
MHD_mod_rep = MHD_mod[:,1:7]
NWR_mod_rep = NWR_mod[:,1:7]
SMO_mod_rep = SMO_mod[:,1:7]
SPO_mod_rep = SPO_mod[:,1:7]

ho = np.array([[BRW_mod_rep], 
               [MHD_mod_rep], 
               [NWR_mod_rep], 
               [SMO_mod_rep], 
               [SPO_mod_rep]])

ho = np.ravel(ho)
ho = np.reshape(ho,(140,6))

nyr = 28
nstat = 5
nreg = 6

H = np.zeros((nstat*nyr, nreg*nyr+1), np.float)

for ireg in range(nreg):
    for iyr in range(nyr):
        for istat in range(nstat):
             deb = istat*28 + iyr
             fin = (istat+1)* 28   #indices des lignes à remplir dans H
             H[deb:fin, 28*ireg + iyr] = ho[28*istat:28*istat+28-iyr, ireg]
            
    #28 = indice pour mon numéro de colonnes (pour les blocs après amérique, 
    #il faut rajouter 28 car on est décalé) 
H[:,nreg*nyr]=1

#Calculer A et xA 

H_trans = np.transpose(H)
R_inverse = np.linalg.inv(R)
B_inverse = np.linalg.inv(B)

A1 = np.matmul(H_trans,R_inverse)
A2 = np.matmul(A1,H)
A = np.linalg.inv(A2 + B_inverse)

xA1 = yo - np.matmul(H,xb)
xA2 = np.matmul(A,H_trans)
xA3 = np.matmul(xA2,R_inverse)
xA4 = np.matmul(xA3,xA1)
xA = xb + xA4


#Récupérer les émissions mondiales


erreurs_prior = np.sqrt(np.diag(B))
erreurs_post = np.sqrt(np.diag(A))

#a posterior

glob_ij_a = np.zeros(28)
err_glob_a = np.zeros(nyr)


for i in range(28):
    glob_annee_sum_a = 0
    glob_annee_a = []
    err = []
    for j in np.arange(i,(nreg-1)*nyr,nyr):
    
        glob_annee_a.append(xA[j])
        err.append(erreurs_post[j])
    glob_annee_sum_a = np.sum(glob_annee_a)
    errsum = np.sum(err)
    glob_ij_a[i] = glob_annee_sum_a
    
#a prior

glob_ij_b = np.zeros(28)


for i in range(28):
    glob_annee_sum_b = 0
    glob_annee_b = []
    for j in np.arange(i,(nreg-1)*nyr,nyr):
    
        glob_annee_b.append(xb[j])
    glob_annee_sum_b = np.sum(glob_annee_b)
    glob_ij_b[i] = glob_annee_sum_b 
    
#graphe

yrs = np.zeros(28)
for j in np.arange(0,28,1):
    yrs[j]=j

yrs = yrs + 1979

plt.errorbar(yrs,glob_ij_a, yerr=errsum, fmt='d')
plt.title('Evolution des émissions globales de CFC-11 entre 1979 et 2006')
plt.xlabel('Années')
plt.ylabel('Emissions globales en kt/an')
#plt.plot(yrs,glob_ij_b)
plt.show()

#par régions

em_Amérique_a = xA[0:28]
em_Af_a = xA[28:56]
em_AsieT_a = xA[56:84]
em_AsN_a = xA[84:112]
em_eu_a = xA[112:140]

em_Amérique_b = xb[0:28]
em_Af_b = xb[28:56]
em_AsieT_b = xb[56:84]
em_AsN_b = xb[84:112]
em_eu_b = xb[112:140]

err_post_Am = erreurs_post[0:28]
err_post_Af = erreurs_post[28:56]
err_post_AsieT = erreurs_post[56:84]
err_post_AsN = erreurs_post[84:112]
err_post_eu = erreurs_post[112:140]

err_prior_Am = erreurs_prior[0:28]
err_prior_Af = erreurs_prior[28:56]
err_prior_AsieT = erreurs_prior[56:84]
err_prior_AsN = erreurs_prior[84:112]
err_prior_eu = erreurs_prior[112:140]

reduc_err_Am = np.mean(err_post_Am/err_prior_Am)
reduc_err_Af = np.mean(err_post_Af/err_prior_Af)
reduc_err_AsT = np.mean(err_post_AsieT/err_prior_AsieT)
reduc_err_AsN = np.mean(err_post_AsN/err_prior_AsN)
reduc_err_Eur = np.mean(err_post_eu/err_prior_eu)

plt.errorbar(yrs,em_Amérique_a, yerr = err_post_Am, fmt='d', markersize=4, color='blue')
plt.errorbar(yrs,em_Af_a, yerr = err_post_Af, fmt='d', markersize=4, color='green')
plt.errorbar(yrs,em_AsieT_a, yerr = err_post_AsieT, fmt='d', markersize=4, color='yellow')
plt.errorbar(yrs,em_AsN_a, yerr = err_post_AsN, fmt='d', markersize=4, color='red')
plt.errorbar(yrs,em_eu_a, yerr = err_post_eu, fmt='d', markersize=4, color='purple')
plt.legend(['Amérique','Afrique','Asie trop','Asie Nord','Europe'])
#plt.title('Evolution des émissions régionales de CFC-11 entre 1979 et 2006')
plt.xlabel('Années')
plt.ylabel('Emissions de CFC par région en kt/an')
plt.savefig('RegEmMLO.eps', dpi=200)
plt.show()

plt.plot(yrs,em_Amérique_b)
plt.plot(yrs,em_Af_b)
plt.plot(yrs,em_AsieT_b)
plt.plot(yrs,em_AsN_b)
plt.plot(yrs,em_eu_b)
plt.show()

#réduction des erreurs

erreurs_prior = np.sqrt(np.diag(B))
erreurs_post = np.sqrt(np.diag(A))

reduc_erreur = np.mean(erreurs_post/erreurs_prior)

#question 2

inventaire1 = np.loadtxt('/Users/felixlangot/Google Drive (felixlangot@gmail.com)/UVSQ/Modélisation/TD4-Inversion/TP_INV/inventaire1_cfc11.txt')
inventaire2 = np.loadtxt('/Users/felixlangot/Google Drive (felixlangot@gmail.com)/UVSQ/Modélisation/TD4-Inversion/TP_INV/inventaire2_cfc11.txt')

annee1 = inventaire1[:,0]
annee2 = inventaire2[:,0]
inventaire1 = inventaire1[:,1]
inventaire2 = inventaire2[:,1]

plt.errorbar(yrs,glob_ij_a, yerr=errsum, fmt='d')
plt.scatter(annee1,inventaire1, color = 'red')
plt.scatter(annee2,inventaire2, color = 'green')
#plt.title('Evolution des émissions globales de CFC-11 entre 1979 et 2006')
plt.xlabel('Années')
plt.ylabel('Emissions globales en kt/an')
plt.legend(['inventaire1','inventaire2','inversion'])
#plt.plot(yrs,glob_ij_b)
plt.savefig('GlobEmMLO.eps', dpi=200)
plt.show()

inventaire1 = inventaire1[:22]
inventaire = (inventaire1 + inventaire2)/2

glob_ij_a_trunc = glob_ij_a[:22]

mse = sklearn.metrics.mean_squared_error(glob_ij_a_trunc,inventaire)
rmse = np.sqrt(mse)

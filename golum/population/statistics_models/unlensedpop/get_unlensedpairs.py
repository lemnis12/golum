import numpy as np
import sys
np.random.seed(23423)
import pylab as plt

## O3

def get_tdmag(det,nn, Ntot,batchtime=3.1536e7):
   td,dist= np.loadtxt("../unlensed/all_bbh_unlensprop_%s.txt"%(det),usecols=(7,8),unpack=1)
   
   idx = (td>=(nn)*batchtime) & (td<=(nn+1)*batchtime)
   td1 = td[idx]
   dist1 = dist [idx]
    
   
   rand1=np.random.randint(0,td1.size,Ntot)
   rand2=np.random.randint(0,td1.size,Ntot)
  
   ## time delays in days 
   td_batch=np.abs((td[rand1]-td[rand2]))/3600./24.
   ## magnification ratio Dlum propto 1/\sqrt(\mu)
   mag_batch=(dist[rand1]/dist[rand2])**2 
   
   return td_batch, mag_batch

delt=3.1536e7/2. ## 6 months for O3a observing run
det=np.array(["O3","AL"])
batchnum=np.array([1,3])
batchtime=np.array([delt,delt])
Ntot=21000
tdcomb=np.zeros(Ntot)
magcomb=np.zeros(Ntot)
for ii in range(det.size-1):
   batchsize=np.int(Ntot/batchnum[ii])
   for jj in range(batchnum[ii]):
      mini=jj*batchsize 
      maxi=(jj+1)*batchsize 
      tdcomb[mini:maxi],magcomb[mini:maxi]=get_tdmag(det[ii],batchnum[ii],batchsize,batchtime[ii])
  
   
   np.savetxt("unlensedpairs_tdmag_%s.txt"%(det[ii]),np.transpose([tdcomb,magcomb])) 

from __future__ import annotations
import numpy as np

def energy_score(samples,target):
    samples=np.asarray(samples,dtype=float);target=np.asarray(target,dtype=float)
    first=np.mean(np.linalg.norm(samples-target[None,...],axis=-1))
    if len(samples)<2:return float(first)
    pairwise=np.linalg.norm(samples[:,None,:]-samples[None,:,:],axis=-1)
    return float(first-.5*np.mean(pairwise))

def crps_ensemble(samples,target):
    samples=np.asarray(samples,dtype=float);target=np.asarray(target,dtype=float)
    return float(np.mean(np.abs(samples-target[None,...]))-.5*np.mean(np.abs(samples[:,None,:]-samples[None,:,:])))

def empirical_coverage(samples,target,level=.90):
    samples=np.asarray(samples,dtype=float);target=np.asarray(target,dtype=float)
    tail=(1-level)/2
    lo=np.quantile(samples,tail,axis=0);hi=np.quantile(samples,1-tail,axis=0)
    return float(np.mean((target>=lo)&(target<=hi)))

def mae(pred,target):return float(np.mean(np.abs(np.asarray(pred)-np.asarray(target))))

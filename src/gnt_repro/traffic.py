from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class PCARepresentation:
    mean:np.ndarray
    components:np.ndarray
    dmin:float=1e-4
    def encode(self,demand):
        x=np.log(np.maximum(np.asarray(demand),self.dmin))
        return (x-self.mean)@self.components.T
    def decode(self,latent):return np.exp(np.asarray(latent)@self.components+self.mean)

def fit_log_pca(demand,rank,dmin=1e-4):
    x=np.log(np.maximum(np.asarray(demand),dmin));mean=x.mean(axis=0)
    _,_,vt=np.linalg.svd(x-mean,full_matrices=False)
    return PCARepresentation(mean,vt[:min(rank,len(vt))],dmin)

def _base_matrix(nodes,rng):
    pop=rng.lognormal(0.,.55,size=nodes)
    gravity=np.outer(pop,pop);gravity/=max(float(gravity.mean()),1e-8)
    np.fill_diagonal(gravity,0.);return gravity

def generate_traffic(network,steps,rng,regime='train',reference_base=None):
    n,p=network.nodes,network.demands
    base=_base_matrix(n,rng) if reference_base is None else reference_base.copy()
    base*=.055/max(float(base.mean()),1e-8)
    base_vec=np.array([base[i,j] for i,j in network.pairs])
    latent=np.zeros(3);out=np.zeros((steps,p))
    for t in range(steps):
        phase=2*np.pi*t/24
        seasonal=1+.18*np.sin(phase)+.08*np.cos(phase/2)
        latent=.94*latent+rng.normal(0,.045,size=3)
        factors=1+.06*latent[0]+.04*np.sin(np.linspace(0,2*np.pi,p,endpoint=False)+latent[1])
        d=base_vec*seasonal*factors*np.exp(rng.normal(0,.035,size=p))
        if regime=='S1':d[::max(1,p//8)]*=1+5*np.exp(-((t-.45*steps)/max(.08*steps,1))**2)
        elif regime=='S2':
            d*=np.exp(rng.normal(0,.10,size=p));d=np.roll(d,p//7)
        elif regime=='S3':
            sink_pairs=[idx for idx,(_,j) in enumerate(network.pairs) if j in (4,9)]
            d[sink_pairs]*=1+7*np.exp(-((t-.55*steps)/max(.10*steps,1))**2)
        elif regime=='S4':d*=1.35
        elif regime=='S5':d*=1.05
        out[t]=np.maximum(d,1e-5)
    return out

def make_datasets(network,cfg,rng):
    base=_base_matrix(network.nodes,rng)
    train=np.concatenate([generate_traffic(network,cfg.steps_per_week,rng,'train',base) for _ in range(cfg.train_weeks)])
    validation=np.concatenate([generate_traffic(network,cfg.steps_per_week,rng,'train',base) for _ in range(cfg.validation_weeks)])
    scenarios={'In-d.':generate_traffic(network,cfg.test_steps,rng,'train',base)}
    for s in ('S1','S2','S3','S4','S5'):
        scenarios[s]=generate_traffic(network,cfg.test_steps,rng,s,base)
    return {'train':train,'validation':validation,**scenarios}

def make_sequences(latent,window):
    return (np.asarray([latent[t-window:t] for t in range(window,len(latent))]),
            np.asarray([latent[t] for t in range(window,len(latent))]))

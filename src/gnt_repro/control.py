from __future__ import annotations
import numpy as np
from .topology import evaluate_network

class DriftGate:
    def __init__(self,beta=.15,tau=.12,threshold=.25):
        self.beta,self.tau,self.threshold=beta,tau,threshold
        self.rho=0.
    def update(self,predicted,measured):
        error=np.sum(np.abs(measured-predicted))/(np.sum(np.abs(measured))+1e-8)
        self.rho=(1-self.beta)*self.rho+self.beta*error
        gate=0. if self.rho>self.threshold else float(np.exp(-self.rho/max(self.tau,1e-8)))
        return float(self.rho),gate

def risk_weighted_reference(network,weights,forecast_ensemble,alpha=.90):
    costs=np.asarray([evaluate_network(network,weights,d)['cost'] for d in forecast_ensemble])
    n=max(1,int(np.ceil((1-alpha)*len(forecast_ensemble))))
    tail=forecast_ensemble[np.argsort(costs)[-n:]]
    return .5*np.mean(forecast_ensemble,axis=0)+.5*np.mean(tail,axis=0)

def randomized_local_search(network,initial_weights,demand_reference,rng,proposals=32):
    best=np.asarray(initial_weights,dtype=float).copy()
    best_cost=float(evaluate_network(network,best,demand_reference)['cost'])
    for _ in range(proposals):
        candidate=np.clip(best*np.exp(rng.normal(0,.10,size=len(best))),.20,4.)
        score=float(evaluate_network(network,candidate,demand_reference)['cost'])
        if score<best_cost:best,best_cost=candidate,score
    return best,best_cost

def static_ospf(network):return np.ones(network.links,dtype=float)

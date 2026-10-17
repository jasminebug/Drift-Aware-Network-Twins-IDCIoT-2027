"""Small PPO-style sanity benchmark, NOT a full network-routing DRL experiment."""
import numpy as np
class GaussianPolicy:
    def __init__(self,seed=7,std=.35):
        rng=np.random.default_rng(seed);self.weight=float(rng.normal(0,.05));self.bias=float(rng.normal(0,.05));self.std=std
    def mean(self,state):return np.tanh(self.weight*np.asarray(state)+self.bias)
    def sample(self,state,rng):
        mu=self.mean(state);a=mu+self.std*rng.normal(size=np.asarray(state).shape)
        logp=-.5*((a-mu)/self.std)**2-np.log(self.std*np.sqrt(2*np.pi))
        return a,logp

def ppo_update(policy,state,action,old_log_prob,advantage,learning_rate=.05,clip_ratio=.2):
    mu=policy.mean(state)
    lp=-.5*((action-mu)/policy.std)**2-np.log(policy.std*np.sqrt(2*np.pi))
    ratio=np.exp(np.clip(lp-old_log_prob,-10,10));un=ratio*advantage
    cl=np.clip(ratio,1-clip_ratio,1+clip_ratio)*advantage
    keep=((un<=cl)|(advantage<0)).astype(float)
    v=keep*ratio*advantage*(action-mu)/(policy.std**2)*(1-mu*mu)
    policy.weight+=learning_rate*float(np.mean(v*state))
    policy.bias+=learning_rate*float(np.mean(v))
    return float(np.mean(np.minimum(un,cl)))

def run_ppo_sanity(seed=7,episodes=40):
    rng=np.random.default_rng(seed);policy=GaussianPolicy(seed);init=[];final=[]
    for i in range(episodes):
        state=rng.uniform(-1,1,size=64);action,old=policy.sample(state,rng)
        reward=-(action-state)**2;init.append(float(np.mean(reward)))
        adv=(reward-np.mean(reward))/(np.std(reward)+1e-8)
        ppo_update(policy,state,action,old,adv)
        if i>=episodes-5:
            probe=np.linspace(-1,1,64);final.append(float(np.mean(-(policy.mean(probe)-probe)**2)))
    return {'initial_mean_reward':float(np.mean(init[:5])),
            'final_probe_reward':float(np.mean(final)),'learned_weight':float(policy.weight)}

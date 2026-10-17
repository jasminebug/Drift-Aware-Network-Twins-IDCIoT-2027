from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .utils import sigmoid

@dataclass
class LSTMCache:
    x:np.ndarray;h_prev:np.ndarray;c_prev:np.ndarray
    i:np.ndarray;f:np.ndarray;o:np.ndarray;g:np.ndarray
    c:np.ndarray;h:np.ndarray

class ConditionalLSTM:
    """One-step residual LSTM with explicit backpropagation through time."""
    def __init__(self,latent_dim,hidden_dim,noise_dim,seed=7):
        self.latent_dim=latent_dim;self.hidden_dim=hidden_dim;self.noise_dim=noise_dim
        self.rng=np.random.default_rng(seed)
        din=latent_dim+noise_dim;scale=1/np.sqrt(din+hidden_dim)
        self.W=self.rng.normal(0,scale,size=(hidden_dim+din,4*hidden_dim))
        self.b=np.zeros(4*hidden_dim)
        self.Wo=self.rng.normal(0,scale,size=(hidden_dim,latent_dim))*.1
        self.bo=np.zeros(latent_dim)
    def _forward(self,context,noise,cache=False):
        h=np.zeros(self.hidden_dim);c=np.zeros(self.hidden_dim);history=[]
        for t,z in enumerate(context):
            n=noise if t==len(context)-1 else np.zeros(self.noise_dim)
            x=np.concatenate([z,n]);a=np.concatenate([h,x]);gates=a@self.W+self.b
            i,f,o,g=np.split(gates,4);i,f,o=sigmoid(i),sigmoid(f),sigmoid(o);g=np.tanh(g)
            hp,cp=h,c;c=f*c+i*g;h=o*np.tanh(c)
            if cache:history.append(LSTMCache(x,hp,cp,i,f,o,g,c,h))
        y=context[-1]+h@self.Wo+self.bo
        return (y,history) if cache else y
    def predict(self,context,noise=None):
        if noise is None:noise=np.zeros(self.noise_dim)
        return self._forward(np.asarray(context),np.asarray(noise),False)
    def sample(self,context,count,rng=None):
        rng=self.rng if rng is None else rng
        return np.asarray([self.predict(context,rng.normal(size=self.noise_dim)) for _ in range(count)])
    def backward_one(self,context,noise,d_prediction):
        _,hist=self._forward(context,noise,True)
        dW=np.zeros_like(self.W);db=np.zeros_like(self.b)
        dWo=np.outer(hist[-1].h,d_prediction);dbo=d_prediction.copy()
        dh=d_prediction@self.Wo.T;dc=np.zeros(self.hidden_dim)
        for item in reversed(hist):
            tc=np.tanh(item.c);do=dh*tc;dc=dc+dh*item.o*(1-tc*tc)
            df=dc*item.c_prev;di=dc*item.g;dg=dc*item.i;dc=dc*item.f
            da=np.concatenate([di*item.i*(1-item.i),df*item.f*(1-item.f),
                              do*item.o*(1-item.o),dg*(1-item.g*item.g)])
            ah=np.concatenate([item.h_prev,item.x]);dW+=np.outer(ah,da);db+=da
            dh=da@self.W[:self.hidden_dim].T
        return {'W':dW,'b':db,'Wo':dWo,'bo':dbo}
    def apply_gradients(self,grads,learning_rate,clip=5.):
        for k in ('W','b','Wo','bo'):
            setattr(self,k,getattr(self,k)-learning_rate*np.clip(grads[k],-clip,clip))
    def train_batch(self,contexts,targets,rng,energy_weight=1.,mse_weight=.25,learning_rate=.001):
        grads={k:np.zeros_like(getattr(self,k)) for k in ('W','b','Wo','bo')}
        total=0.
        for context,target in zip(contexts,targets):
            na=rng.normal(size=self.noise_dim);nb=rng.normal(size=self.noise_dim)
            a=self.predict(context,na);b=self.predict(context,nb)
            da=a-target;db=b-target
            norm_a=np.linalg.norm(da)+1e-8;norm_b=np.linalg.norm(db)+1e-8
            norm_ab=np.linalg.norm(a-b)+1e-8
            es=.5*(norm_a+norm_b)-.5*norm_ab
            mse=.5*(np.mean(da*da)+np.mean(db*db))
            total+=energy_weight*es+mse_weight*mse
            ga=energy_weight*(.5*da/norm_a-.5*(a-b)/norm_ab)+mse_weight*da/self.latent_dim
            gb=energy_weight*(.5*db/norm_b+.5*(a-b)/norm_ab)+mse_weight*db/self.latent_dim
            grads_a=self.backward_one(context,na,ga)
            grads_b=self.backward_one(context,nb,gb)
            for k in grads:grads[k]+=grads_a[k]+grads_b[k]
        scale=1/max(len(contexts),1)
        self.apply_gradients({k:v*scale for k,v in grads.items()},learning_rate)
        return float(total*scale)

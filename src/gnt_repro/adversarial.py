"""Optional discriminator utility; not trained in the default experimental pipeline."""
import numpy as np
from .utils import sigmoid
class LinearDiscriminator:
    def __init__(self,feature_dim,seed=7):
        rng=np.random.default_rng(seed);self.weight=rng.normal(0,.05,size=feature_dim);self.bias=0.
    def probability(self,features):return sigmoid(np.asarray(features)@self.weight+self.bias)
    def train_step(self,real,fake,learning_rate=.01):
        real=np.atleast_2d(real);fake=np.atleast_2d(fake)
        p_real=self.probability(real);p_fake=self.probability(fake)
        gr=p_real-1.;gf=p_fake
        self.weight-=learning_rate*(real.T@gr+fake.T@gf)/max(len(real),1)
        self.bias-=learning_rate*float(np.mean(np.concatenate([gr,gf])))
        return float(-np.mean(np.log(p_real+1e-8))-np.mean(np.log(1-p_fake+1e-8)))
    def generator_gradient(self,fake):
        fake=np.atleast_2d(fake);p=np.atleast_1d(self.probability(fake))
        return -(1-p)[:,None]*self.weight[None,:]

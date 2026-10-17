from __future__ import annotations
from dataclasses import dataclass,asdict
from pathlib import Path

@dataclass
class ExperimentConfig:
    mode:str='quick'
    seed:int=7
    nodes:int=14
    steps_per_week:int=60
    train_weeks:int=4
    validation_weeks:int=1
    test_steps:int=48
    window:int=8
    latent_rank:int=8
    hidden_dim:int=32
    noise_dim:int=4
    forecast_rollouts:int=8
    train_iterations:int=40
    batch_size:int=8
    learning_rate:float=.002
    energy_weight:float=1.
    mse_weight:float=.25
    drift_beta:float=.15
    drift_tau:float=.12
    drift_threshold:float=.25
    risk_alpha:float=.90
    proposals:int=6
    output_dir:Path=Path('outputs')
    def to_jsonable(self):
        d=asdict(self);d['output_dir']=str(d['output_dir']);return d

def make_config(mode,output_dir,seed=None):
    if mode=='paper':
        cfg=ExperimentConfig(mode='paper',seed=7 if seed is None else seed,steps_per_week=400,train_weeks=40,validation_weeks=4,test_steps=400,window=12,latent_rank=16,hidden_dim=64,noise_dim=8,forecast_rollouts=100,train_iterations=900,batch_size=32,learning_rate=.001,proposals=80)
    elif mode=='quick':cfg=ExperimentConfig(seed=7 if seed is None else seed)
    else:raise ValueError(f'Unknown mode: {mode}')
    cfg.output_dir=Path(output_dir)
    return cfg

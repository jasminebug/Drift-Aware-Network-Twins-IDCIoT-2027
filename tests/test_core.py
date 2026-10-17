import numpy as np
from gnt_repro.topology import make_network
from gnt_repro.model import ConditionalLSTM
from gnt_repro.control import DriftGate
from gnt_repro.ppo import run_ppo_sanity
from gnt_repro.losses import energy_score,empirical_coverage

def test_network_topology():
    net=make_network(7)
    assert net.links==42 and net.demands==182
    A=net.path_matrix(np.ones(net.links))
    assert A.shape==(42,182) and np.all(A.sum(axis=0)>0)

def test_lstm_shapes_finite():
    m=ConditionalLSTM(5,8,3,seed=7)
    sample=m.sample(np.zeros((4,5)),6,np.random.default_rng(3))
    assert sample.shape==(6,5) and np.isfinite(sample).all()

def test_energy_coverage():
    sample=np.random.default_rng(1).normal(size=(32,4))
    assert np.isfinite(energy_score(sample,np.zeros(4)))
    assert 0<=empirical_coverage(sample,np.zeros(4))<=1

def test_gate_shutdown():
    gate=DriftGate(beta=1,threshold=.2)
    assert gate.update(np.ones(10),np.ones(10))[1]>0
    assert gate.update(np.ones(10),10*np.ones(10))[1]==0

def test_ppo_diagnostic():
    assert all(np.isfinite(x) for x in run_ppo_sanity(seed=4,episodes=8).values())

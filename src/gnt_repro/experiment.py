"""Compact diagnostic experiment. Does not recreate all paper ablations."""
from __future__ import annotations
import csv,platform
from pathlib import Path
import numpy as np
from .config import ExperimentConfig
from .control import DriftGate,randomized_local_search,risk_weighted_reference,static_ospf
from .losses import crps_ensemble,empirical_coverage,energy_score,mae
from .model import ConditionalLSTM
from .plotting import save_figures
from .ppo import run_ppo_sanity
from .topology import evaluate_network,make_network
from .traffic import fit_log_pca,make_datasets,make_sequences
from .utils import relative_l1,write_json

def _write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)

def _fit_forecaster(latent,cfg,seed):
    contexts,targets=make_sequences(latent,cfg.window)
    model=ConditionalLSTM(latent.shape[1],cfg.hidden_dim,cfg.noise_dim,seed=seed)
    rng=np.random.default_rng(seed+100)
    for step in range(cfg.train_iterations):
        idx=rng.integers(0,len(contexts),size=min(cfg.batch_size,len(contexts)))
        lr=cfg.learning_rate*(.35+.65*np.exp(-step/max(cfg.train_iterations,1)))
        model.train_batch(contexts[idx],targets[idx],rng,cfg.energy_weight,cfg.mse_weight,lr)
    return model

def _forecast_metrics(samples,targets,rep):
    pred=rep.decode(samples.mean(axis=1));truth=rep.decode(targets)
    decoded=rep.decode(samples.reshape(-1,samples.shape[-1])).reshape(samples.shape[0],samples.shape[1],-1)
    crps=float(np.mean([crps_ensemble(s,y) for s,y in zip(decoded,truth)]))
    cov=float(np.mean([empirical_coverage(s,y,.90) for s,y in zip(decoded,truth)]))
    energy=float(np.mean([energy_score(s,y) for s,y in zip(decoded,truth)]))
    return {'mae':mae(pred,truth),'crps':crps,'cov90':cov,'energy':energy,'relative_l1':relative_l1(pred,truth)}

def _controller_eval(network,model,rep,scenario,cfg,seed):
    rng=np.random.default_rng(seed);weights=static_ospf(network)
    gate=DriftGate(cfg.drift_beta,cfg.drift_tau,cfg.drift_threshold)
    names=('reactive','raw_forecast','ensemble_mean','risk_ungated','gated','oracle','static')
    history={k:[] for k in names};rho_trace=[];gate_trace=[];ue=[];ge=[]
    previous_forecast=scenario[cfg.window-1]
    for t in range(cfg.window,len(scenario)-1):
        context=rep.encode(scenario[t-cfg.window:t]);ensemble=rep.decode(model.sample(context,cfg.forecast_rollouts,rng))
        raw=ensemble[0];mean_ref=np.mean(ensemble,axis=0)
        risk_ref=risk_weighted_reference(network,weights,ensemble,cfg.risk_alpha)
        rho,g=gate.update(previous_forecast,scenario[t]);previous_forecast=raw
        gated_ref=g*risk_ref+(1-g)*scenario[t]
        refs={'reactive':scenario[t],'raw_forecast':raw,'ensemble_mean':mean_ref,
              'risk_ungated':risk_ref,'gated':gated_ref,'oracle':scenario[t+1],'static':scenario[t]}
        for k,ref in refs.items():
            cand=weights if k=='static' else randomized_local_search(network,weights,ref,rng,cfg.proposals)[0]
            history[k].append(float(evaluate_network(network,cand,scenario[t+1])['delay_ms']))
        rho_trace.append(rho);gate_trace.append(g)
        ue.append(relative_l1(risk_ref,scenario[t+1])*100)
        ge.append(relative_l1(gated_ref,scenario[t+1])*100)
    result={k:float(np.mean(v)) for k,v in history.items()}
    result.update({'rho':np.asarray(rho_trace),'gate':np.asarray(gate_trace),'ungated_error':np.asarray(ue),'gated_error':np.asarray(ge)})
    return result

def run_experiment(cfg:ExperimentConfig):
    out=Path(cfg.output_dir);out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(cfg.seed);network=make_network(cfg.seed)
    datasets=make_datasets(network,cfg,rng)
    rep=fit_log_pca(datasets['train'],cfg.latent_rank)
    model=_fit_forecaster(rep.encode(datasets['train']),cfg,cfg.seed)
    in_latent=rep.encode(datasets['In-d.']);contexts,targets=make_sequences(in_latent,cfg.window)
    srng=np.random.default_rng(cfg.seed+10)
    samples=np.asarray([model.sample(context,cfg.forecast_rollouts,srng) for context in contexts])
    forecast=_forecast_metrics(samples,targets,rep)
    persistence=np.repeat(in_latent[cfg.window-1:-1,None,:],cfg.forecast_rollouts,axis=1)
    twin_rows=[{'variant':'Persistence',**_forecast_metrics(persistence,targets,rep)},
               {'variant':'Conditional LSTM + energy score',**forecast}]
    _write_csv(out/'twin_variants.csv',twin_rows)
    scenarios=['In-d.','S1','S2','S3','S4','S5']
    controller={};delay_rows=[]
    for i,s in enumerate(scenarios):
        r=_controller_eval(network,model,rep,datasets[s],cfg,cfg.seed+50+i)
        controller[s]=r
        for k in ('reactive','raw_forecast','ensemble_mean','risk_ungated','gated','oracle','static'):
            delay_rows.append({'scenario':s,'method':k,'mean_delay_ms':r[k]})
    _write_csv(out/'controller_delay.csv',delay_rows)
    drift=100*float(np.mean([relative_l1(datasets['In-d.'][t],datasets['In-d.'][t-1]) for t in range(1,len(datasets['In-d.']))]))
    raw_errors=[];gated_errors=[];criterion=[]
    for s in scenarios:
        r=controller[s];raw=float(np.mean(r['ungated_error']));gated=float(np.mean(r['gated_error']))
        raw_errors.append(raw);gated_errors.append(gated)
        criterion.append({'scenario':s,'drift_percent':drift,'ungated_reference_error_percent':raw,'gated_reference_error_percent':gated})
    _write_csv(out/'criterion.csv',criterion)
    decoded=rep.decode(samples.reshape(-1,samples.shape[-1])).reshape(samples.shape[0],samples.shape[1],-1)
    truth=rep.decode(targets);levels=np.asarray([.5,.8,.9,.95])
    curves=[('LSTM + ES',[float(np.mean([empirical_coverage(s,y,level) for s,y in zip(decoded,truth)])) for level in levels],None)]
    names=['static OSPF','reactive TE','raw proactive','risk ungated','gated GNT-TE']
    values=[[controller[s][k] for k in ('static','reactive','raw_forecast','risk_ungated','gated')] for s in scenarios]
    save_figures(out,{'scenarios':scenarios,'drift':drift,'criterion_raw':raw_errors,
        'criterion_gated':gated_errors,'gate_trace':{k:controller['S3'][k] for k in ('rho','gate','ungated_error','gated_error')},
        'threshold':cfg.drift_threshold,'controller_names':names,'controller_values':values,
        'coverage_levels':levels,'coverage_curves':curves})
    ppo=run_ppo_sanity(cfg.seed+900,episodes=40 if cfg.mode=='paper' else 12)
    write_json(out/'ppo_sanity.json',ppo)
    summary={'forecast':{'full_cov90':forecast['cov90'],**forecast},
       'controller':{'gated_mean_delay_ms':float(np.mean([controller[s]['gated'] for s in scenarios])),
       'scenarios':{s:{k:v for k,v in controller[s].items() if np.isscalar(v)} for s in scenarios}},
       'ppo_sanity':ppo,'configuration':cfg.to_jsonable()}
    write_json(out/'metrics.json',summary)
    write_json(out/'manifest.json',{'configuration':cfg.to_jsonable(),
      'python':platform.python_version(),'numpy':np.__version__,'platform':platform.platform(),
      'notes':'Locally generated diagnostic run; not claimed as full manuscript reproduction.'})
    return summary

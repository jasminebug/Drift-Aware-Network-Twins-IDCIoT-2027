#!/usr/bin/env python3
"""Diagnostic runnable companion for accepted IDCIoT 2027 paper; not full result reproduction."""
from pathlib import Path
import sys,argparse
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from gnt_repro.config import make_config
from gnt_repro.experiment import run_experiment

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('quick','paper'),default='quick')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/local')
    parser.add_argument('--seed',type=int,default=None)
    args=parser.parse_args()
    cfg=make_config(args.mode,args.output_dir,args.seed)
    result=run_experiment(cfg)
    print('Finished diagnostic run:',cfg.output_dir.resolve())
    print('Gated delay:',result['controller']['gated_mean_delay_ms'])
    print('90% empirical coverage:',result['forecast']['cov90'])
    print('IMPORTANT: These are diagnostic numbers, NOT the accepted-paper tables.')

if __name__=='__main__':main()

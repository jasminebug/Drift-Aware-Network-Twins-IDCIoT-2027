from __future__ import annotations
import json
from pathlib import Path
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-np.clip(x,-30,30)))

def write_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,sort_keys=True),encoding='utf-8')

def relative_l1(pred,truth,eps=1e-8):
    pred=np.asarray(pred);truth=np.asarray(truth)
    return float(np.mean(np.sum(np.abs(pred-truth),axis=-1)/(np.sum(np.abs(truth),axis=-1)+eps)))

def set_global_style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.labelsize':10,'figure.dpi':140,'savefig.dpi':220})

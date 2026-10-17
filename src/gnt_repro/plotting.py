from pathlib import Path
import numpy as np
from .utils import set_global_style

def save_figures(out,artifacts):
    import matplotlib.pyplot as plt
    set_global_style();out=Path(out);out.mkdir(parents=True,exist_ok=True)
    scenarios=artifacts['scenarios'];x=np.arange(len(scenarios))
    # Use the complete architecture diagram under assets/manuscript_figure_1.png.
    # Do not generate a mock architecture chart for diagnostic runs.
    levels=artifacts['coverage_levels']
    fig,ax=plt.subplots(figsize=(5.5,3.4))
    ax.plot(levels*100,levels*100,'--',label='nominal')
    for label,vals,_color in artifacts['coverage_curves']:
        ax.plot(levels*100,np.asarray(vals)*100,'-o',label=label)
    ax.set(xlabel='Nominal coverage (%)',ylabel='Empirical coverage (%)')
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'coverage.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,3.5))
    ax.axhline(artifacts['drift'],linestyle='--',label='traffic drift')
    ax.bar(x-.18,artifacts['criterion_raw'],width=.36,label='ungated risk reference')
    ax.bar(x+.18,artifacts['criterion_gated'],width=.36,label='gated reference')
    ax.set_xticks(x,scenarios);ax.set_ylabel('Relative error (%)');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'criterion.png');plt.close(fig)
    tr=artifacts['gate_trace']
    fig,axes=plt.subplots(2,1,figsize=(6,4.8),sharex=True)
    axes[0].plot(tr['ungated_error'],label='ungated');axes[0].plot(tr['gated_error'],label='gated');axes[0].legend(fontsize=7)
    axes[1].plot(tr['rho'],label='rho');axes[1].plot(tr['gate'],label='gate')
    axes[1].axhline(artifacts['threshold'],linestyle=':',label='threshold');axes[1].legend(fontsize=7)
    axes[1].set_xlabel('Control interval');fig.tight_layout();fig.savefig(out/'gate_trace.png');plt.close(fig)
    vals=np.asarray(artifacts['controller_values']);names=artifacts['controller_names']
    fig,ax=plt.subplots(figsize=(7,3.6));width=.8/len(names)
    for i,name in enumerate(names):ax.bar(x+(i-(len(names)-1)/2)*width,vals[:,i],width=width,label=name)
    ax.set_xticks(x,scenarios);ax.set_ylabel('Mean delay (ms)');ax.legend(fontsize=7,ncol=3)
    fig.tight_layout();fig.savefig(out/'controller_comparison.png');plt.close(fig)

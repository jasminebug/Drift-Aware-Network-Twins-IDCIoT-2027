\
"""Make manuscript-transcription illustrations. No model training or reproduction claim."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path(__file__).resolve().parents[1]
DATA=BASE/'paper'/'reported'
OUT=BASE/'assets'/'reported'
OUT.mkdir(parents=True,exist_ok=True)
with (DATA/'table_i_forecasting.csv').open(encoding='utf8',newline='') as f:
    d={r['variant']:r for r in csv.DictReader(f)}
labels=['LSTM (mse)','T1 (mse+adv)','T2 (mse+es)','GAN-LSTM']
values=[float(d[x]['coverage_at_nominal_90'].replace('%','').strip()) for x in labels]
fig,ax=plt.subplots(figsize=(8,3.4))
bars=ax.bar(['MSE','MSE + adv','MSE + ES','MSE + adv + ES'], values)
ax.axhline(90,ls='--',color='0.3',lw=1,label='Nominal 90%')
ax.set(ylabel='Empirical coverage (%)', ylim=(0,100), title='Reported 90% predictive-interval coverage')
for b,v in zip(bars,values):ax.text(b.get_x()+b.get_width()/2,v+1,f'{v:.1f}%',ha='center',fontsize=9)
ax.legend(frameon=False,loc='upper left'); fig.tight_layout()
for suffix in ['png','svg']: fig.savefig(OUT/f'coverage_table_i.{suffix}',dpi=210)
plt.close(fig)
with (DATA/'table_ii_delay_ms.csv').open(encoding='utf8',newline='') as f:
    d={r['reference_matrix']:r for r in csv.DictReader(f)}
sc=['in_distribution','S1','S2','S3','S4','S5']
fig,ax=plt.subplots(figsize=(8.8,3.8))
for row,label in [('(reactive)','Reactive TE'),('Raw forecast','Raw forecast'),('Risk-weighted, ungated','Risk-weighted (ungated)'),('+ drift gate','With drift gate'),('True  (oracle)','Oracle')]:
    if row not in d:
        opts={r.strip():r for r in d}
        row=opts.get(row.strip(),row)
    vals=[float(d[row][s]) for s in sc]
    ax.plot(['In-d.','S1','S2','S3','S4','S5'],vals,marker='o',label=label)
ax.set(ylabel='Mean delay (ms)',title='Reported routing delay by scenario (Table II)')
ax.legend(ncol=3,frameon=False,fontsize=8,loc='upper left')
ax.grid(alpha=.18);fig.tight_layout()
for suffix in ['png','svg']:fig.savefig(OUT/f'delay_table_ii.{suffix}',dpi=210)
plt.close(fig)
print('Rebuilt Table I and Table II visualizations from transcribed values. Not an experimental rerun.')

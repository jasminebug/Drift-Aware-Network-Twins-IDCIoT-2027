\
"""Verify manuscript-table transcription arithmetic and warn about known gaps."""
import csv, json
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
with (BASE/'paper/reported/table_i_forecasting.csv').open(newline='',encoding='utf8') as f:
    forecasting={x['variant']:x for x in csv.DictReader(f)}
a=forecasting['LSTM (mse)'];b=forecasting['GAN-LSTM']
crps=100*(float(a['crps'])-float(b['crps']))/float(a['crps'])
assert abs(crps-25.9)<0.1,(crps,'Expected manuscript-rounded 25.9%')
assert b['coverage_at_nominal_90'].replace(' ','').replace('%','')=='74.7'
with (BASE/'paper/reported/table_ii_delay_ms.csv').open(newline='',encoding='utf8') as f:
    delays={x['reference_matrix'].strip():x for x in csv.DictReader(f)}
reactive=float(delays['(reactive)']['in_distribution']);gated=float(delays['+ drift gate']['in_distribution']);oracle=float(delays['True  (oracle)']['in_distribution'])
oracle_gap_closure=100*(reactive-gated)/(reactive-oracle)
assert abs(oracle_gap_closure-90.9)<0.2,(oracle_gap_closure,'Expected manuscript-rounded 90.9%')
s3_gain=100*(float(delays['Risk-weighted, ungated']['S3'])-float(delays['+ drift gate']['S3']))/float(delays['Risk-weighted, ungated']['S3'])
assert abs(s3_gain-15.6)<0.2
meta=json.loads((BASE/'paper/reported/provenance.json').read_text(encoding='utf8'))
assert meta['independently_reproduced_by_repo'] is False
print('PASS: Table I CRPS improvement {:.2f}%; coverage 74.7%; Table II oracle-gap closure {:.2f}%; S3 improvement {:.2f}%'.format(crps,oracle_gap_closure,s3_gain))
print('NOTE: Mathematical consistency checks are not scientific reproduction of manuscript experiments.')
print('NOTE: Table III has incomplete cells; S5 is mismatched in existing simulator; GAN/DRL ablations are not fully reproduced.')

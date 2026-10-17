\
"""Tests check transcription arithmetic, not the reported scientific validity."""
from pathlib import Path
import csv

ROOT=Path(__file__).resolve().parents[1]

def load(fname,key):
    with (ROOT/'paper/reported'/fname).open(newline='',encoding='utf8') as f:
        return {row[key].strip():row for row in csv.DictReader(f)}

def test_manuscript_crps_and_coverage():
    d=load('table_i_forecasting.csv','variant')
    assert abs((float(d['LSTM (mse)']['crps'])-float(d['GAN-LSTM']['crps']))/float(d['LSTM (mse)']['crps'])*100-25.9)<0.1
    assert d['GAN-LSTM']['coverage_at_nominal_90'].replace(' ','').startswith('74.7')

def test_delay_manuscript_oracle_gap():
    d=load('table_ii_delay_ms.csv','reference_matrix')
    gap=(float(d['(reactive)']['in_distribution'])-float(d['+ drift gate']['in_distribution']))/(float(d['(reactive)']['in_distribution'])-float(d['True  (oracle)']['in_distribution']))*100
    assert abs(gap-90.9)<0.2

def test_incomplete_table_iii_is_not_filled():
    p=(ROOT/'paper/reported/table_iii_as_printed.csv').read_text(encoding='utf8')
    assert 'printed_row_1' in p

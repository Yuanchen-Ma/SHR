"""Check that outcome columns and future query rows cannot change SHR inputs."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[n]='1'
import numpy as np
import pandas as pd
import joblib
from shr.io import key, read, write
from shr.paths import EXPERIMENTS, model_dir, prediction_dir
from shr.risk.features import BASE
from shr.risk.training import validation_inputs

from . import DIAGNOSTICS, SELECTION
from .scoring import fit, score

def main():
    selected=[]
    for arch in ['tcn','transformer']:
        for cohort in ['FD001','ds01_03']:
            selected.append(next(j for j in read(EXPERIMENTS/'design.json')['jobs'] if j['architecture']==arch and j['cohort']==cohort and j['repetition']==11 and j['protocol']=='matched'))
    checks=[]
    for j in selected:
        aux=joblib.load(model_dir(j)/'auxiliary.joblib')
        folder=prediction_dir(j)/'evaluation'
        p=next(p for p in sorted(folder.glob('*/*/test.parquet')) if p.parent.name=='matched')
        f=pd.read_parquet(p)
        unit=f.unit_id.iloc[0];corr=f[f.unit_id.eq(unit)].corruption_id.iloc[0]
        f=f[f.unit_id.eq(unit)&f.corruption_id.eq(corr)].reset_index(drop=True)
        full=aux.features(f)
        drop=[c for c in f if c in ['rul_true','rul_raw','error','late_loss','corruption_onset_cycle','corruption_type','corruption_severity'] or c.startswith('score_')]
        visible=f.drop(columns=drop)
        without=aux.features(visible)
        np.testing.assert_allclose(full[BASE].to_numpy(float),without[BASE].to_numpy(float),rtol=0,atol=1e-10)
        cut=float(np.median(f.cycle));take=f.cycle.le(cut).to_numpy()
        partial=aux.features(visible.loc[take].reset_index(drop=True))
        np.testing.assert_allclose(full.loc[take,BASE].to_numpy(float),partial[BASE].to_numpy(float),rtol=0,atol=1e-10)
        h,val,meta=validation_inputs(j);choice=read(SELECTION/key(j)/'shr_xgboost.json')['best']
        model=fit(j,h,meta,'shr','xgboost',choice)
        s0=score(model,full,'shr');s1=score(model,without,'shr');sp=score(model,partial,'shr')
        np.testing.assert_allclose(s0,s1,rtol=0,atol=1e-8)
        np.testing.assert_allclose(s0[take],sp,rtol=0,atol=1e-8)
        checks.append(dict(job=j,queries=len(f),prefix_queries=int(take.sum()),outcome_free_inputs=True,prefix_invariant_inputs=True,outcome_free_score=True,prefix_invariant_score=True))
        print('CAUSALITY PASS',j,flush=True)
    write(DIAGNOSTICS/'causality_verification.json',checks)

if __name__=='__main__':main()

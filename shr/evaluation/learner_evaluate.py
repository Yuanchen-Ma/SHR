"""Use the complete-score validation choices for the risk learner comparison."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[n]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse,time,traceback,gc
from concurrent.futures import ProcessPoolExecutor,as_completed
import pandas as pd
import numpy as np
from threadpoolctl import threadpool_limits
from shr.io import key, now, read, write
from shr.paths import EXPERIMENTS
from shr.risk import learners as lm
from shr.risk.training import outer_features, validation_inputs

from . import LEARNERS, SELECTION
from .metrics import evaluate
from .scoring import fit, score, validation_value

def one(task):
    j,learner=task;dest=LEARNERS/key(j)/learner
    if (dest/'complete.json').exists():return read(dest/'complete.json')
    begin=time.perf_counter()
    with threadpool_limits(limits=4,user_api='openmp'),threadpool_limits(limits=1,user_api='blas'):
        path=SELECTION/key(j)/f'shr_{learner}.json'
        chosen=read(path)['best']
        lm.CACHE_NEIGHBORS=learner=='knn';lm.NEIGHBOR_CACHE.clear()
        h,val,meta=validation_inputs(j);model=fit(j,h,meta,'shr',learner,chosen)
        np.testing.assert_allclose(validation_value(val,score(model,val,'shr')),chosen['validation_area'],rtol=0,atol=1e-8)
        rows=[]
        for ds, scene, frame in outer_features(j, meta):
            scores = score(model, frame, 'shr')
            rows.append(dict(**j, dataset_id=ds, scenario=scene, method='shr',
                             learner=learner, **evaluate(frame, scores, 20)))
        dest.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_parquet(dest/'metrics.parquet',index=False)
        report=dict(job=j,learner=learner,seconds=time.perf_counter()-begin,finished_at=now(),rows=len(rows))
        write(dest/'complete.json',report)
        del model;lm.NEIGHBOR_CACHE.clear();lm.CACHE_NEIGHBORS=False;gc.collect()
        return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=2);p.add_argument('--limit',type=int);a=p.parse_args()
    tasks=[(j,l) for j in read(EXPERIMENTS/'design.json')['jobs'] if j['architecture']=='tcn' for l in ['rf','logistic','knn']]
    tasks=[t for t in tasks if not (LEARNERS/key(t[0])/t[1]/'complete.json').exists()]
    if a.limit:tasks=tasks[:a.limit]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        fs={pool.submit(one,t):t for t in tasks}
        for n,f in enumerate(as_completed(fs),1):
            try:r=f.result()
            except Exception:write(EXPERIMENTS/'learner_evaluation_error.json',dict(task=fs[f],traceback=traceback.format_exc()));raise
            write(EXPERIMENTS/'learner_evaluation_status.json',dict(complete=n,total=len(tasks),last=r))
            print('LEARNER',n,'/',len(tasks),r,flush=True)

if __name__=='__main__':main()

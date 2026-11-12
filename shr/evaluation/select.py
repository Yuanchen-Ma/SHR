"""Select risk-model hyperparameters on disjoint validation engines."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[n]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse,time,gc,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from shr.io import key, now, read, write
from shr.paths import EXPERIMENTS, PRIOR_DIR, ROOT
from shr.risk import learners as lm
from shr.risk.training import validation_inputs

from . import SELECTION
from .scoring import fit, score, validation_value

def one(task):
    j,method,learner=task;k=key(j);dest=SELECTION/k
    out=dest/f'{method}_{learner}.json'
    if out.exists():return dict(cached=True,job=j,method=method,learner=learner)
    begin=time.perf_counter()
    with threadpool_limits(limits=4,user_api='openmp'),threadpool_limits(limits=1,user_api='blas'):
        h,val,meta=validation_inputs(j)
        # Baseline learner selections were computed on the same validation engines.
        prior=PRIOR_DIR/'jobs'/k/f'{method}_{learner}_selected.json'
        if method!='shr' and prior.exists():
            chosen=read(prior);write(out,dict(**chosen,reused_selection=prior.relative_to(ROOT).as_posix(),criterion_unchanged=True,finished_at=now()))
            return dict(job=j,method=method,learner=learner,reused=True)
        design=read(EXPERIMENTS/'design.json');records=[]
        lm.CACHE_NEIGHBORS=learner=='knn';lm.NEIGHBOR_CACHE.clear()
        for candidate in design['candidates'][learner]:
            cp=dest/'trials'/f'{method}_{candidate["id"]}.json'
            if cp.exists():records.append(read(cp));continue
            start=time.perf_counter();item=dict(**candidate)
            if learner=='knn':
                pos=(h.rul_pred>h.rul_true).to_numpy();sizes=[len(h),int(pos.sum()),int((~pos).sum())]
                if candidate['parameters']['n_neighbors']>min(x for x in sizes if x):
                    item.update(status='invalid',reason='k exceeds nonempty head sample count');write(cp,item);records.append(item);continue
            model=fit(j,h,meta,method,learner,candidate)
            ss=score(model,val,method)
            item.update(status='ok',validation_area=validation_value(val,ss),seconds=time.perf_counter()-start)
            write(cp,item);records.append(item)
            del model;gc.collect()
        lm.NEIGHBOR_CACHE.clear();lm.CACHE_NEIGHBORS=False
        best=min([r for r in records if r['status']=='ok'],key=lambda r:(r['validation_area'],r['id']))
        write(out,dict(best=best,validation_units=meta['validation_units'],complete_moment_constraints=True,finished_at=now(),seconds=time.perf_counter()-begin))
    return dict(job=j,method=method,learner=learner,best=best['id'],seconds=round(time.perf_counter()-begin,2))

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=3);p.add_argument('--limit',type=int);p.add_argument('--architecture',default='all');p.add_argument('--learner',default='all');a=p.parse_args()
    d=read(EXPERIMENTS/'design.json');tasks=[]
    for j in d['jobs']:
        if a.architecture!='all' and j['architecture']!=a.architecture:continue
        pairs=[('shr',l) for l in (['xgboost','rf','logistic','knn'] if j['architecture']=='tcn' else ['xgboost'])]
        pairs += [('epsilon','xgboost'),('rcr','xgboost'),('ccr','rf')]
        for method,learner in pairs:
            if a.learner!='all' and learner!=a.learner:continue
            if not (SELECTION/key(j)/f'{method}_{learner}.json').exists():tasks.append((j,method,learner))
    # Complete all primary-method selections before the alternative SHR learners.
    tasks.sort(key=lambda t: (t[1]=='shr' and t[2]!='xgboost',t[0]['architecture']=='transformer'))
    if a.limit:tasks=tasks[:a.limit]
    write(EXPERIMENTS/'selection_status.json',dict(started_at=now(),pending=len(tasks),workers=a.workers,coordinator_pid=os.getpid()))
    start=time.perf_counter();done=0
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        fs={pool.submit(one,t):t for t in tasks}
        for f in as_completed(fs):
            try:r=f.result()
            except Exception:
                write(EXPERIMENTS/'selection_error.json',dict(task=fs[f],traceback=traceback.format_exc()));raise
            done+=1;print('SELECT',done,'/',len(tasks),r,flush=True)
            write(EXPERIMENTS/'selection_status.json',dict(complete=done,total=len(tasks),elapsed_seconds=time.perf_counter()-start,workers=a.workers,coordinator_pid=os.getpid(),last=r))
    print('SELECTION FINISHED',done,flush=True)

if __name__=='__main__':main()

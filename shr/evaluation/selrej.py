"""Validation-select native SelRej training settings with a fixed architecture/loss."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[n]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import time,argparse,copy,gc,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
import torch
from torch import nn
import joblib
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from shr.io import key, now, read, write
from shr.paths import EXPERIMENTS, PRIOR_DIR, model_dir, official_path, partition_path, prediction_dir
from shr.randomness import derive_seed
from shr.risk.features import AGE, HISTORY, STATIC
from shr.risk.training import validation_inputs

from . import SELREJ
from .metrics import RiskCurve, evaluate
from .policy import allocate, stream_groups, sufficient
from .scoring import validation_value

def predict(net,scaler,h):
    x=scaler.transform(h[STATIC+HISTORY+AGE]).astype(np.float32)
    with torch.no_grad():return -net(torch.tensor(x)).squeeze(1).numpy().astype(float)

def fit_select(j,h,val,design,dest):
    scaler=StandardScaler().fit(h[STATIC+HISTORY+AGE]);x=scaler.transform(h[STATIC+HISTORY+AGE]).astype(np.float32)
    e=(h.rul_pred-h.rul_true).to_numpy();w=h.fit_weight.to_numpy();target=e*e
    xx=torch.tensor(x);ww=torch.tensor(w/w.sum(),dtype=torch.float32)
    yy=torch.tensor(target/max(np.average(target,weights=w),1),dtype=torch.float32)
    seed=derive_seed(j['seed'],'v10_crossfit',j['protocol']);best=None;beststate=None;trials=[];canonical_error=None
    for k,lr in enumerate(design['learning_rates']):
        torch.manual_seed(seed);net=nn.Sequential(nn.Linear(x.shape[1],64),nn.ReLU(),nn.Linear(64,1))
        opt=torch.optim.Adam(net.parameters(),lr=lr)
        for epoch in range(1,max(design['steps'])+1):
            opt.zero_grad();g=torch.sigmoid(net(xx).squeeze(1));coverage=(ww*g).sum()
            loss=(ww*yy*g).sum()/coverage.clamp_min(1e-6)+design['coverage_penalty']*torch.relu(design['target_coverage']-coverage)**2
            loss.backward();opt.step()
            if epoch not in design['steps']:continue
            ss=predict(net,scaler,val);v=validation_value(val,ss)
            row=dict(id=f'selrej_{k}_{epoch:03d}',learning_rate=lr,steps=epoch,validation_area=v)
            trials.append(row)
            if best is None or (v,row['id'])<(best['validation_area'],best['id']):best=row;beststate=copy.deepcopy(net.state_dict())
            if lr==.003 and epoch==150:
                native=joblib.load(model_dir(j)/'loss_rejection.joblib')
                oldx=native.scaler.transform(val[native.cols]).astype(np.float32)
                with torch.no_grad():old=-native.gates[0](torch.tensor(oldx)).squeeze(1).numpy().astype(float)
                canonical_error=float(np.max(np.abs(ss-old)))
                np.testing.assert_allclose(ss,old,rtol=1e-5,atol=1e-5)
    net=nn.Sequential(nn.Linear(x.shape[1],64),nn.ReLU(),nn.Linear(64,1));net.load_state_dict(beststate);net.eval()
    np.testing.assert_allclose(validation_value(val,predict(net,scaler,val)),best['validation_area'],rtol=0,atol=1e-8)
    dest.mkdir(parents=True,exist_ok=True);joblib.dump(dict(state=beststate,scaler=scaler,input_dim=x.shape[1]),dest/'selected_model.joblib',compress=3)
    write(dest/'selection.json',dict(job=j,best=best,trials=trials,canonical_replay_max_error=canonical_error))
    return net,scaler

def one(j):
    dest=SELREJ/key(j)
    if (dest/'complete.json').exists():return read(dest/'complete.json')
    begin=time.perf_counter();design=read(EXPERIMENTS/'selrej_design.json');torch.set_num_threads(4)
    with threadpool_limits(limits=4,user_api='openmp'),threadpool_limits(limits=1,user_api='blas'):
        h,val,_=validation_inputs(j);mp,rp=model_dir(j),prediction_dir(j)
        if (dest/'selection.json').exists():
            d=joblib.load(dest/'selected_model.joblib');scaler=d['scaler'];net=nn.Sequential(nn.Linear(d['input_dim'],64),nn.ReLU(),nn.Linear(64,1));net.load_state_dict(d['state']);net.eval()
        else:net,scaler=fit_select(j,h,val,design,dest)
        aux=joblib.load(mp/'auxiliary.joblib')
        cal=pd.read_parquet(rp/'calibration.parquet');hcpath=EXPERIMENTS/'features'/key(j)/'calibration.parquet'
        hc=pd.read_parquet(hcpath) if hcpath.exists() else aux.features(cal)
        selection=read(partition_path(j))
        policy=cal.unit_id.isin(selection['policy_units']).to_numpy();cs=predict(net,scaler,hc)[policy]
        np.savez_compressed(dest/'calibration_scores.npz',scores=cs)
        folder = rp / 'evaluation'
        files = sorted(folder.glob('*/*/test.parquet'))
        if not files:
            raise FileNotFoundError(f'No held-out predictions in {folder}')
        offline=[];online=[];official=[];curves=[]
        for path in files:
            ds,scene=path.parent.relative_to(folder).parts;f=pd.read_parquet(path)
            cached=PRIOR_DIR/'features'/key(j)/'outer'/ds/scene/'test.parquet'
            hf=pd.read_parquet(cached) if cached.exists() else aux.features(f)
            ss=predict(net,scaler,hf);core=scene!='clean' and not scene.endswith('_high')
            md=dict(**j,dataset_id=ds,scenario=scene,core=core,variant='selrej')
            offline.append(dict(**md,**evaluate(f,ss,20)))
            sf=f[['unit_id','cycle','corruption_id','rul_pred','rul_true']].copy();sf['score_selrej']=ss
            dd=dest/'scores'/ds/scene;dd.mkdir(parents=True,exist_ok=True);sf.to_parquet(dd/'queries.parquet',index=False)
            if not core:continue
            u=np.searchsorted(np.sort(cs),ss,side='right')/len(cs);rej,forced=allocate(stream_groups(f),u,.2)
            v,_=sufficient(f,~rej,20,{})
            online.append(dict(**md,forced_fraction=float(forced.mean()),fixed_rejection=float((ss>np.quantile(cs,.8)).mean()),**v))
            if j['architecture']=='tcn':
                e=(f.rul_pred-f.rul_true).to_numpy()
                for metric,loss in [('positive',np.maximum(e,0)),('mse',e*e)]:
                    curve=RiskCurve(loss,ss)
                    for c in np.linspace(.5,1,51):curves.append(dict(**md,metric=metric,coverage=c,risk=float(curve.risk(c))))
        if j['architecture']=='tcn' and j['benchmark']=='cmapss' and j['protocol']=='matched':
            f=pd.read_parquet(official_path(j));ss=predict(net,scaler,aux.features(f))
            for scene,take in [('clean',f.corruption_type.eq('clean').to_numpy()),('matched',np.ones(len(f),bool))]:
                take=take&f.is_official_cutoff.to_numpy(bool)
                official.append(dict(**j,scenario=scene,variant='selrej',**evaluate(f.loc[take],ss[take],20)))
        for name,rows in [('offline',offline),('online',online),('official',official),('curves',curves)]:
            if rows:pd.DataFrame(rows).to_parquet(dest/(name+'.parquet'),index=False)
        result=dict(job=j,seconds=time.perf_counter()-begin,finished_at=now())
        write(dest/'complete.json',result);del net,aux;gc.collect();return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=1);p.add_argument('--limit',type=int);a=p.parse_args()
    js=[j for j in read(EXPERIMENTS/'design.json')['jobs'] if not (SELREJ/key(j)/'complete.json').exists()]
    if a.limit:js=js[:a.limit]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        fs={pool.submit(one,j):j for j in js}
        for n,f in enumerate(as_completed(fs),1):
            try:r=f.result()
            except Exception:write(EXPERIMENTS/'selrej_error.json',dict(job=fs[f],traceback=traceback.format_exc()));raise
            write(EXPERIMENTS/'selrej_status.json',dict(complete=n,total=len(js),last=r));print('SELREJ',n,'/',len(js),r,flush=True)

if __name__=='__main__':main()

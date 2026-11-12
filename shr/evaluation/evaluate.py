"""Re-evaluate selected risk models without retraining the fixed RUL predictors."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[n]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse,gc,time,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
import joblib
import torch
from xgboost import XGBRegressor
from threadpoolctl import threadpool_limits
from shr.io import key, now, read, write
from shr.paths import EXPERIMENTS, PRIOR_DIR, model_dir, official_path, partition_path, prediction_dir
from shr.risk.features import BASE, PAIR, pair_data
from shr.risk.methods import RiskMethod
from shr.risk.moments import HEAD_INPUTS, anchor_fit, anchor_predict, current_features, reference
from shr.risk.settings import role_weights, settings
from shr.risk.training import fit_seed, validation_inputs

from . import PRIMARY, SELECTION
from .metrics import RiskCurve, evaluate
from .policy import allocate, stream_groups, sufficient
from .scoring import components, fit, score, validation_value

NAMES=['epsilon','lossrej_linear','selrej','rcr','ccr','shr']
GRID=[0,5,10,20,30,40]

def loss_rejection_scores(model,h):
    x=model.scaler.transform(h[model.cols]).astype(np.float32)
    linear=np.maximum(model.linear.predict(x)[:,0],0)
    with torch.no_grad():sel=-model.gates[0](torch.tensor(x)).squeeze(1).numpy().astype(float)
    return {'lossrej_linear':linear,'selrej':sel}

def query_frame(f,scores,**extra):
    cols=[c for c in ['unit_id','cycle','corruption_id','rul_pred','rul_true'] if c in f]
    out=f[cols].copy()
    for name,value in scores.items():out['score_'+name]=value
    for name,value in extra.items():out[name]=value
    return out

def new_heads(h,j,meta,candidate,cols=BASE):
    model=RiskMethod('shr','xgboost',candidate['parameters'],fit_seed(j,'shr'),meta['kind'])
    model.cols=list(cols)
    return model.fit(h)

def current_only(h,j,aux,candidate):
    """Fit the current-only ablation with the selected final-head settings."""
    h0=current_features(h);features=h0.copy();z=np.full((len(h),2),np.nan)
    rp = prediction_dir(j)
    adapt_units=pd.read_parquet(rp/'adapt.parquet',columns=['unit_id']).unit_id.unique()
    cfg=settings()['model'];seed=fit_seed(j,'shr')
    for fold,role in enumerate(aux.roles):
        take=h.unit_id.isin(role['fit']).to_numpy();held=h.unit_id.isin(role['held']).to_numpy()
        w=role_weights(h.loc[take],adapt_units)
        mm=anchor_fit(h0.loc[take],h.loc[take],w,cfg,seed+fold+1)
        z[held]=anchor_predict(mm,h0.loc[held])
    assert np.isfinite(z).all()
    features[['empirical_mse','empirical_positive']]=z
    features['rul_pred']=h.rul_pred.to_numpy();features['rul_true']=h.rul_true.to_numpy();features['fit_weight']=h.fit_weight.to_numpy()
    model=new_heads(features,j,{'kind':'empirical'},candidate,HEAD_INPUTS)
    anchors=anchor_fit(h0,h,h.fit_weight.to_numpy(),cfg,seed)
    return model,anchors

def direct_controls(h,j,candidate):
    p=dict(candidate['parameters']);seed=fit_seed(j,'shr')
    x=h[BASE].to_numpy(np.float32);e=(h.rul_pred-h.rul_true).to_numpy(float)
    w=h.fit_weight.to_numpy(float);w=w*len(w)/w.sum();out={}
    for capacity in [False,True]:
        for name,y in [('joint',e*e+20*np.maximum(e,0)),('mse',e*e),('positive',np.maximum(e,0))]:
            trees=(1200 if name=='joint' else 600) if capacity else 300
            options=dict(p,n_estimators=trees)
            m=XGBRegressor(**options,objective='reg:squarederror',random_state=seed,tree_method='hist',n_jobs=4,verbosity=0).fit(x,y,sample_weight=w)
            out[(name,capacity)]=m
    return out

def direct_scores(models,h,b):
    x=h[BASE].to_numpy(np.float32);p=h.rul_pred.to_numpy(float);out={}
    for cap in [False,True]:
        lbound=np.maximum(p*p+20*p,(125-p)**2);mbound=np.maximum(p*p,(125-p)**2)
        joint=np.clip(models['joint',cap].predict(x),0,lbound)
        mse=np.clip(models['mse',cap].predict(x),0,mbound)
        positive=np.clip(models['positive',cap].predict(x),0,np.minimum(p,np.sqrt(mse)))
        suffix='_capacity' if cap else ''
        out['direct_joint'+suffix]=np.maximum(.5*b+.5*joint,0)
        out['direct_split'+suffix]=np.maximum(.5*b+.5*(mse+20*positive),0)
    return out

def run_job(j,controls=True):
    dest=PRIMARY/key(j);marker=dest/'complete.json'
    if marker.exists():return read(marker)
    start=time.perf_counter();mp, rp = model_dir(j), prediction_dir(j);h,val,meta=validation_inputs(j)
    d=read(EXPERIMENTS/'design.json');chosen={}
    for name,learner in d['primary_learners'].items():
        path=SELECTION/key(j)/f'{name}_{learner}.json'
        chosen[name]=read(path)['best']
    dest.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    aux=joblib.load(mp/'auxiliary.joblib');loss_rejection=joblib.load(mp/'loss_rejection.joblib')
    models={name:fit(j,h,meta,name,d['primary_learners'][name],choice) for name,choice in chosen.items()}
    # Assert the selected configuration is evaluated using exactly the complete score.
    np.testing.assert_allclose(validation_value(val,score(models['shr'],val,'shr')),chosen['shr']['validation_area'],rtol=0,atol=1e-8)
    selection=read(partition_path(j))
    cal=pd.read_parquet(rp/'calibration.parquet');policy=cal.unit_id.isin(selection['policy_units']).to_numpy()
    assert policy.any() and not set(selection['validation_units'])&set(selection['policy_units'])
    assert set(cal.unit_id)==set(selection['validation_units'])|set(selection['policy_units'])
    cp=EXPERIMENTS/'features'/key(j)/'calibration.parquet';cp.parent.mkdir(parents=True,exist_ok=True)
    if cp.exists():hc=pd.read_parquet(cp)
    else:hc=aux.features(cal);hc.to_parquet(cp,index=False)
    cals={name:score(model,hc,name) for name,model in models.items()};cals.update(loss_rejection_scores(loss_rejection,hc))
    np.savez_compressed(dest/'calibration_scores.npz',policy=policy,**cals)
    # Reference diagnostics use validation outcomes only; the fixed rho rule is not reselected.
    vm,vp=components(models['shr'],val);valrefs={}
    for kind in ['empirical','historical']:
        ss=.5*reference(val,kind,20)+.5*(vm+20*vp)
        valrefs[kind]=validation_value(val,ss)
    rho=float(np.mean(h.rul_true.to_numpy()>=125-1e-6))
    src,tar,_=pair_data(h)
    if len(src):
        kappa=(h.cycle.to_numpy()[tar]-h.cycle.to_numpy()[src])-(h.rul_true.to_numpy()[src]-h.rul_true.to_numpy()[tar])
        pairdiag=dict(pairs=len(src),fraction_cap_adjustment=float(np.mean(kappa>1e-6)),mean_cap_adjustment=float(kappa.mean()))
    else:pairdiag=dict(pairs=0)
    write(dest/'reference_diagnostic.json',dict(job=j,capped_fraction=rho,kind=meta['kind'],validation=valrefs,**pairdiag))
    controlled={};anchors=None;direct=None
    if controls and j['architecture']=='tcn':
        mc=dest/'controlled_models.joblib'
        if mc.exists():controlled,anchors,direct=joblib.load(mc)
        else:
            controlled['no_pair']=new_heads(h,j,meta,chosen['shr'],[c for c in BASE if c not in PAIR])
            inside=aux.features(h,False);inside['fit_weight']=h.fit_weight.to_numpy()
            controlled['in_sample']=new_heads(inside,j,meta,chosen['shr'])
            controlled['current_only'],anchors=current_only(h,j,aux,chosen['shr'])
            direct=direct_controls(h,j,chosen['shr'])
            joblib.dump((controlled,anchors,direct),mc,compress=3)
    offline=[];online=[];ablations=[];weights=[];penalties=[];official=[];curve_rows=[]
    relbase = rp / 'evaluation'
    files = sorted(relbase.glob('*/*/test.parquet'))
    if not files:
        raise FileNotFoundError(f'No held-out predictions in {relbase}')
    for path in files:
        ds,scene=path.parent.relative_to(relbase).parts;f=pd.read_parquet(path)
        core=scene!='clean' and not scene.endswith('_high')
        cached=PRIOR_DIR/'features'/key(j)/'outer'/ds/scene/'test.parquet'
        if cached.exists():hf=pd.read_parquet(cached)
        else:hf=aux.features(f)
        np.testing.assert_array_equal(f[['unit_id','cycle','corruption_id']].to_numpy(),hf[['unit_id','cycle','corruption_id']].to_numpy())
        scores={name:score(model,hf,name) for name,model in models.items()};scores.update(loss_rejection_scores(loss_rejection,hf))
        md=dict(**j,dataset_id=ds,scenario=scene,core=core)
        for name in NAMES:offline.append(dict(**md,variant=name,**evaluate(f,scores[name],20)))
        where=dest/'scores'/ds/scene;where.mkdir(parents=True,exist_ok=True)
        m,p=components(models['shr'],hf);b=reference(hf,meta['kind'],20)
        query_frame(f,scores,shr_m=m,shr_p=p,reference_m=reference(hf,meta['kind'],0),reference_p=(reference(hf,meta['kind'],20)-reference(hf,meta['kind'],0))/20).to_parquet(where/'queries.parquet',index=False)
        if not core:continue
        # All online comparisons use exactly the same held-out policy engines.
        groups=stream_groups(f)
        for name in NAMES:
            cs=np.sort(cals[name][policy]);u=np.searchsorted(cs,scores[name],side='right')/len(cs)
            rejected,forced=allocate(groups,u,.2)
            values,_=sufficient(f,~rejected,20,{})
            fixed=scores[name]>np.quantile(cals[name][policy],.8)
            online.append(dict(**md,variant=name,forced_fraction=float(forced.mean()),fixed_rejection=float(fixed.mean()),**values))
        if j['architecture']!='tcn':continue
        err=(f.rul_pred-f.rul_true).to_numpy(float)
        for name in NAMES:
            for metric,ll in [('mse',err*err),('positive',np.maximum(err,0))]:
                rc=RiskCurve(ll,scores[name])
                for c in np.linspace(.5,1,51):curve_rows.append(dict(**md,variant=name,metric=metric,coverage=c,risk=float(rc.risk(c))))
        # Reweight fitted components, without retraining or selecting on test results.
        for alpha in np.linspace(0,1,11):
            ss=(1-alpha)*b+alpha*(m+20*p)
            weights.append(dict(**md,role='test',alpha=float(alpha),joint_area=evaluate(f,ss,20)['joint_area']))
        cm,cp_=components(models['shr'],hc)
        for cost in GRID:
            ss=.5*reference(hf,meta['kind'],cost)+.5*(m+cost*p)
            cs=.5*reference(hc,meta['kind'],cost)+.5*(cm+cost*cp_)
            vals=evaluate(f,ss,cost)
            u=np.searchsorted(np.sort(cs[policy]),ss,side='right')/int(policy.sum())
            rej,_=allocate(groups,u,.2);on,_=sufficient(f,~rej,cost,{})
            penalties.append(dict(**md,cost=cost,A2=vals['mse_area'],Aplus=vals['positive_area'],Alambda=vals['joint_area'],R2=on['mse'],Rplus=on['positive'],Rlambda=on['joint'],rejection_pct=100*on['rejection']))
        if controlled:
            pm,pp=components(controlled['no_pair'],hf)
            im,ip=components(controlled['in_sample'],hf)
            hh=current_features(f);hh[['empirical_mse','empirical_positive']]=anchor_predict(anchors,hh)
            ss=score(controlled['current_only'],hh,'shr')
            empirical=reference(hf,'empirical',20);historical=reference(hf,'historical',20)
            aa=dict(shr=scores['shr'],without_distribution_features=.5*b+.5*(pm+20*pp),
                empirical_reference=.5*empirical+.5*(m+20*p),historical_reference=.5*historical+.5*(m+20*p),
                without_distribution_paths=.5*empirical+.5*(pm+20*pp),without_all_history=ss,
                in_sample_aux=.5*b+.5*(im+20*ip),reference_only=b,directional_only=m+20*p)
            aa.update(direct_scores(direct,hf,b))
            for name,ss in aa.items():ablations.append(dict(**md,variant=name,**evaluate(f,ss,20)))
    if j['architecture']=='tcn':
        for alpha in np.linspace(0,1,11):
            ss=(1-alpha)*reference(val,meta['kind'],20)+alpha*(vm+20*vp)
            weights.append(dict(**j,role='validation',alpha=float(alpha),dataset_count=int(val.dataset_id.nunique()),joint_area=validation_value(val,ss)))
    if j['architecture']=='tcn' and j['benchmark']=='cmapss' and j['protocol']=='matched':
        op=official_path(j)
        f=pd.read_parquet(op);hf=aux.features(f)
        scores={name:score(model,hf,name) for name,model in models.items()};scores.update(loss_rejection_scores(loss_rejection,hf))
        for scene,take in [('clean',f.corruption_type.eq('clean').to_numpy()),('matched',np.ones(len(f),bool))]:
            take=take&f.is_official_cutoff.to_numpy(bool)
            for name in NAMES:official.append(dict(**j,scenario=scene,variant=name,**evaluate(f.loc[take],scores[name][take],20)))
    for name,rows in [('offline',offline),('online',online),('ablation',ablations),('weights',weights),('lambda',penalties),('official',official),('curves',curve_rows)]:
        if rows:pd.DataFrame(rows).to_parquet(dest/(name+'.parquet'),index=False)
    report=dict(job=j,seconds=time.perf_counter()-start,finished_at=now(),offline_rows=len(offline),ablation_rows=len(ablations),online_rows=len(online))
    write(marker,report)
    del models,aux,controlled,anchors,direct;gc.collect()
    return report

def one(j):
    with threadpool_limits(limits=4,user_api='openmp'),threadpool_limits(limits=1,user_api='blas'):return run_job(j)

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=2);p.add_argument('--architecture',default='all');p.add_argument('--limit',type=int);a=p.parse_args()
    js=[j for j in read(EXPERIMENTS/'design.json')['jobs'] if a.architecture=='all' or j['architecture']==a.architecture]
    js=[j for j in js if not (PRIMARY/key(j)/'complete.json').exists()]
    if a.limit:js=js[:a.limit]
    start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        fs={pool.submit(one,j):j for j in js}
        for n,f in enumerate(as_completed(fs),1):
            try:r=f.result()
            except Exception:write(EXPERIMENTS/'evaluation_error.json',dict(job=fs[f],traceback=traceback.format_exc()));raise
            write(EXPERIMENTS/'evaluation_status.json',dict(complete=n,total=len(js),elapsed_seconds=time.perf_counter()-start,last=r))
            print('EVALUATE',n,'/',len(js),r,flush=True)

if __name__=='__main__':main()

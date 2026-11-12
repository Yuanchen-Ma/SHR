"""Reconstruct the full C-MAPSS query grid from fitted ensembles."""

from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import torch

from shr.data.cmapss import SETTING_COLUMNS, load_training_set
from shr.data.corruption import apply_corruption, causal_fill
from shr.data.windows import build_windows, records_to_arrays
from shr.io import fingerprint, load_config, read, sha, write
from shr.predictors.cmapss import fit_distance_model, make_preprocessor
from shr.predictors.query import build_risk_table
from shr.predictors.registry import make_rul_model
from shr.predictors.training import predict_ensemble
from shr.randomness import derive_seed
from shr.schemas import PreparedTrajectory


@lru_cache(maxsize=4)
def raw_data(directory, dataset_id):
    return load_training_set(directory, dataset_id, 125)


def replay(base, original, dataset_id, role, device, cache, signature, high=False):
    identity=dict(base=str(base),role=role,high=high,predictor_signature=signature,
        rows=sha(base/'oof_risk_table.parquet') if role=='oof' else
        fingerprint(original[['unit_id','corruption_id','corruption_seed','inner_fold_id']].drop_duplicates().to_dict('records')),
        code=sha(Path(__file__)))
    dest=cache/fingerprint(identity);path=dest/'rows.parquet'
    if (dest/'complete.json').exists():
        assert sha(path)==read(dest/'complete.json')['sha256']
        return pd.read_parquet(path),str(dest)
    cfg=load_config(base/'resolved_config.yaml');split=read(base/'split_manifest.json')
    raw=raw_data(cfg['paths']['raw_data_dir'],dataset_id)
    tables=[];audit=[]
    for fold in sorted(original.inner_fold_id.unique()):
        subset=original[original.inner_fold_id.eq(fold)]
        cp=base/'checkpoints'/('final_base' if fold<0 else f'inner_fold_{fold}')
        files=sorted(cp.glob('member_*.pt'))
        ck=[torch.load(p,map_location='cpu',weights_only=False) for p in files]
        fit_units=ck[0]['metadata']['training_units']
        assert not set(fit_units)&set(subset.unit_id),'Backbone leakage'
        if fold<0: prep=joblib.load(base/'preprocessing/preprocessor.joblib')
        else: prep=make_preprocessor(cfg,derive_seed(int(split['seed']),'fold',int(fold))).fit(raw[raw.unit_id.isin(fit_units)].copy())
        models=[]
        for checkpoint in ck:
            m=make_rul_model(int(checkpoint['input_dim']),cfg);m.load_state_dict(checkpoint['state_dict']);models.append(m)
        if fold<0: distance=joblib.load(base/'preprocessing/latent_distance.joblib')
        else:
            transformed_train=prep.transform(raw[raw.unit_id.isin(fit_units)].copy())
            distance=fit_distance_model(transformed_train,prep,[SimpleNamespace(model=m) for m in models],cfg,
                derive_seed(derive_seed(int(split['seed']),'fold',int(fold)),'distance'),device)
        transformed=prep.transform(raw[raw.unit_id.isin(subset.unit_id)].copy());trajectories=[]
        for (_,cid),g in subset.groupby(['unit_id','corruption_id'],sort=True):
            first=g.iloc[0];unit=transformed[transformed.unit_id.eq(first.unit_id)].sort_values('cycle')
            values=unit[prep.active_sensors].to_numpy(float);cycle=unit.cycle.to_numpy(int)
            severity='high' if high and first.corruption_type!='clean' else first.corruption_severity
            seed=derive_seed(int(first.corruption_seed),'cross_benchmark_high') if high else int(first.corruption_seed)
            corrupt,mask,manifest=apply_corruption(values,np.ones_like(values,bool),first.corruption_type,
                severity,int(first.unit_id),cycle,seed)
            if not high: assert manifest.corruption_id==cid
            filled,delta=causal_fill(corrupt,mask)
            trajectories.append(PreparedTrajectory(int(first.unit_id),cycle,unit.rul.to_numpy(float),filled,
                unit[SETTING_COLUMNS].to_numpy(float),mask,delta,unit.condition_id.to_numpy(int),manifest))
        records=build_windows(trajectories,30,5,cfg['preprocessing']['include_delta_t'],prep.slope_mean,prep.slope_std,None,int(split['seed']))
        # Window builder appends a terminal query: remove it BEFORE inference.
        records=[r for r in records if (r.cycle-1)%5==0]
        x,_,_=records_to_arrays(records);prediction=predict_ensemble(models,x,int(cfg['model']['batch_size']),device)
        f=build_risk_table(records,prediction,distance,split['split_id'],int(fold),fit_units,
            float(cfg['risk']['severe_delta']),float(cfg['risk']['critical_horizon']),dataset_id)
        keys=['unit_id','corruption_id','cycle'];parity={}
        if not high:
            old=subset[(subset.cycle-1)%5==0]
            at=f.set_index(keys).loc[pd.MultiIndex.from_frame(old[keys])]
            for col in ['rul_pred','qr_width','ensemble_std','latent_mahalanobis_conditioned','latent_mahalanobis_global']:
                parity[col]=float(np.max(np.abs(at[col].to_numpy()-old[col].to_numpy())))
                if not np.allclose(at[col],old[col],atol=.02,rtol=.005):raise ValueError(f'Replay mismatch {col}: {parity[col]}')
        assert f[keys].duplicated().sum()==0
        audit.append(dict(fold=int(fold),training_units=fit_units,scored_units=sorted(f.unit_id.unique().tolist()),
            checkpoints={str(p):sha(p) for p in files},parity=parity))
        tables.append(f);del x,models,records,trajectories,ck
        torch.cuda.empty_cache()
    f=pd.concat(tables,ignore_index=True)
    end=raw.groupby('unit_id').cycle.max();f['rul_raw']=f.unit_id.map(end)-f.cycle
    dest.mkdir(parents=True,exist_ok=True);f.to_parquet(path,index=False)
    write(dest/'complete.json',dict(identity=identity,sha256=sha(path),folds=audit,n_rows=len(f)))
    print('Replay', dataset_id, cfg['model']['architecture'], split['split_id'], role, high, len(f), flush=True)
    return f,str(dest)


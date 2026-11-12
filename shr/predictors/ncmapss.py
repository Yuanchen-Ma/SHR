"""Fit N-CMAPSS predictor ensembles and evaluate the fixed flight grid."""

import time

import joblib
import numpy as np
import pandas as pd
import torch

from shr.data.ncmapss import Preprocessor, windows
from shr.data.windows import records_to_arrays
from shr.io import completed, mark_complete, read, write
from shr.predictors import training
from shr.predictors.query import ConditionedLatentDistance, build_risk_table
from shr.predictors.training import load_models
from shr.randomness import derive_seed


def split_units(data,seed):
    fit,cal=[] ,[]
    rng=np.random.default_rng(seed)
    for _,g in data[data.role.eq('dev')].groupby('dataset_id',sort=True):
        units=rng.permutation(sorted(g.unit_id.unique())).tolist()
        cal.extend(units[:2]);fit.extend(units[2:])
    test=sorted(data.loc[data.role.eq('test'),'unit_id'].unique().tolist())
    if set(fit)&set(cal) or (set(fit)|set(cal))&set(test):raise ValueError('Engine roles overlap')
    return sorted(fit),sorted(cal),test


def fit_backbone(data, arch, seed, sig, dest, cfg, device, scenarios):
    if completed(dest,sig):
        return {r:pd.read_parquet(dest/(r+'.parquet')) for r in ['oof','calibration','test']}
    fit,cal,test=split_units(data,seed)
    rng=np.random.default_rng(seed)
    groups=np.array_split(rng.permutation(fit),3)
    tables=[];epoch_records=[]
    dest.mkdir(parents=True,exist_ok=True)
    write(dest/'roles.json',dict(fit=fit,calibration=cal,test=test))
    for fold in range(4):
        final=fold==3
        held=[] if final else groups[fold].astype(int).tolist()
        available=[u for u in fit if u not in held]
        if final:
            gradient_units,validation_units=available,[]
        else:
            ordered=np.random.default_rng(seed+fold).permutation(available).tolist()
            validation_units=ordered[:2];gradient_units=ordered[2:]
        folder=dest/('final' if final else f'fold_{fold}')
        if completed(folder,sig):
            prep=joblib.load(folder/'preprocessor.joblib');distance=joblib.load(folder/'distance.joblib')
            models=load_models(folder/'checkpoints',cfg,device)
            epochs=read(folder/'fit.json')['epochs']
        else:
            print(f'Train {arch} seed {seed} '+('final' if final else f'OOF {fold+1}/3'),flush=True)
            start=time.perf_counter()
            train=data[data.unit_id.isin(gradient_units)]
            prep=Preprocessor().fit(train,seed+fold)
            tr=windows(train,prep,[('clean','none')],seed,1)
            tx,ty,_=records_to_arrays(tr)
            if final:
                epochs=np.maximum(1,np.median(epoch_records,axis=0).astype(int)).tolist()
                trained=training.train_ensemble_fixed_epochs(tx,ty,epochs,cfg,derive_seed(seed,'base',fold),device)
            else:
                val=windows(data[data.unit_id.isin(validation_units)],prep,[('clean','none')],seed,1)
                vx,vy,_=records_to_arrays(val)
                trained=training.train_ensemble(tx,ty,vx,vy,cfg,derive_seed(seed,'base',fold),device)
                epochs=[r.best_epoch for r in trained]
            models=[r.model for r in trained]
            clean=windows(train,prep,[('clean','none')],seed,5)
            cx,_,_=records_to_arrays(clean)
            pred=training.predict_ensemble(models,cx,256,device)
            distance=ConditionedLatentDistance().fit(pred['embedding'],np.array([r.condition_id for r in clean]),min_samples=5)
            folder.mkdir(parents=True,exist_ok=True)
            joblib.dump(prep,folder/'preprocessor.joblib');joblib.dump(distance,folder/'distance.joblib')
            training.save_ensemble(trained,folder/'checkpoints',dict(architecture=arch,training_units=available,
                gradient_units=gradient_units,early_stopping_units=validation_units))
            write(folder/'fit.json',dict(gradient_units=gradient_units,early_stopping_units=validation_units,
                held_out_units=held,epochs=epochs,seconds=time.perf_counter()-start,
                training_curves=[r.history for r in trained],config=cfg))
            mark_complete(folder,sig)
        if not final:epoch_records.append(epochs)
        roles=[('calibration',cal),('test',test)] if final else [('oof',held)]
        for role,units in roles:
            if set(units)&set(available):raise ValueError('Backbone target engine leakage')
            records=windows(data[data.unit_id.isin(units)],prep,scenarios,seed,5)
            x,_,_=records_to_arrays(records)
            prediction=training.predict_ensemble(models,x,256,device)
            table=build_risk_table(records,prediction,distance,str(seed),-1 if final else fold,available,10.,30.,'NCMAPSS')
            table['dataset_id']=table.unit_id.map(data.drop_duplicates('unit_id').set_index('unit_id').dataset_id)
            if role=='oof':tables.append(table)
            else:table.to_parquet(dest/(role+'.parquet'),index=False)
        del models
        if device.type=='cuda':torch.cuda.empty_cache()
    pd.concat(tables,ignore_index=True).to_parquet(dest/'oof.parquet',index=False)
    mark_complete(dest,sig)
    return {r:pd.read_parquet(dest/(r+'.parquet')) for r in ['oof','calibration','test']}

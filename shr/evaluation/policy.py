"""Sequential acceptance under a cumulative rejection budget."""
import math

import numpy as np
import pandas as pd

from .metrics import losses


def stream_groups(frame):
    t=frame.cycle.to_numpy()
    return [np.asarray(idx)[np.argsort(t[idx])] for idx in frame.groupby(['unit_id','corruption_id'],sort=False).indices.values()]


def allocate(groups,u,budget,eta=.05):
    reject=np.zeros(len(u),bool)
    forced=np.zeros(len(u),bool)
    for idx in groups:
        q=1-budget
        count=0
        for n,k in enumerate(idx,1):
            want=u[k]>q
            allowed=count+1<=math.floor(budget*n+1e-12)
            action=bool(want and allowed)
            reject[k]=action
            forced[k]=bool(want and not allowed)
            count+=action
            q=min(1.,max(0.,q+eta*(int(action)-budget)))
        assert np.all(np.cumsum(reject[idx])<=np.floor(budget*np.arange(1,len(idx)+1)+1e-12))
    return reject,forced


def sufficient(frame,accepted,cost,meta):
    ll=losses(frame)
    code,ids=pd.factorize(frame.unit_id,sort=True)
    cnt=np.bincount(code,minlength=len(ids))
    acc=np.bincount(code,weights=accepted,minlength=len(ids))
    sums={k:np.bincount(code,weights=v*accepted,minlength=len(ids)) for k,v in ll.items()}
    stats=pd.DataFrame(dict(unit_id=ids,n_queries=cnt,accepted_count=acc,**{k+'_sum':v for k,v in sums.items()})).assign(**meta)
    n=acc.sum()
    values=dict(rejection=1-n/len(frame),accepted_count=n,n_queries=len(frame),
                mse=sums['mse'].sum()/n,positive=sums['positive'].sum()/n,nasa_mean=sums['nasa'].sum()/n,
                nasa_total=sums['nasa'].sum())
    values['rmse']=np.sqrt(values['mse'])
    values['joint']=values['mse']+cost*values['positive']
    return values,stats

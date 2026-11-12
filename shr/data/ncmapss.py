"""Aggregate N-CMAPSS sensor streams to end-of-flight observations."""

import h5py
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from shr.schemas import PreparedTrajectory, CorruptionManifest
from shr.data.corruption import causal_fill
from shr.data.windows import build_windows
from shr.randomness import derive_seed

DEVELOPMENT_FILES = ('N-CMAPSS_DS01-005.h5', 'N-CMAPSS_DS02-006.h5', 'N-CMAPSS_DS03-012.h5')
CONFIRMATION_FILES = ('N-CMAPSS_DS04.h5', 'N-CMAPSS_DS05.h5', 'N-CMAPSS_DS06.h5', 'N-CMAPSS_DS07.h5')
SENSORS=['T24','T30','T48','T50','P15','P2','P21','P24','Ps30','P40','P50','Nf','Nc','Wf']
SETTINGS=['alt','Mach','TRA','T2']
KINDS=['burst_missing','channel_dropout','gradual_bias']
SCENARIOS=[('clean','none')]+[(k,s) for k in KINDS for s in ['low','medium','high']]


def validate_hdf5(path):
    """Check the schema without reading the flight data arrays."""
    with h5py.File(path, "r") as data:
        for name, expected in (("X_s_var", SENSORS), ("W_var", SETTINGS)):
            column = data.get(name)
            if (not isinstance(column, h5py.Dataset)
                    or h5py.check_string_dtype(column.dtype) is None
                    or column.shape != (len(expected),)
                    or list(column.asstr()[:]) != expected):
                raise ValueError(f"Unexpected {name} in {path}")
        for role in ("dev", "test"):
            widths = {"A": 2, "X_s": len(SENSORS), "W": len(SETTINGS), "Y": 1}
            shapes = {}
            for name, width in widths.items():
                key = f"{name}_{role}"
                column = data.get(key)
                shape = column.shape if isinstance(column, h5py.Dataset) else None
                if shape is None or len(shape) != 2 or shape[0] == 0 or shape[1] < width:
                    raise ValueError(f"Missing or invalid {key} in {path}")
                if name in ("X_s", "W") and shape[1] != width:
                    raise ValueError(f"Unexpected number of columns in {key} in {path}")
                shapes[key] = shape[0]
            if len(set(shapes.values())) != 1:
                raise ValueError(f"Mismatched {role} row counts in {path}")


def aggregate_file(path,index,role):
    """One observation at the END of each completed flight. No hs/X_v/T inputs."""
    parts=[]
    with h5py.File(path,'r') as f:
        assert [v.decode() for v in f['X_s_var'][:]]==SENSORS
        assert [v.decode() for v in f['W_var'][:]]==SETTINGS
        for start in range(0,len(f['A_'+role]),250000):
            stop=start+250000
            a=f['A_'+role][start:stop,:2]
            x=f['X_s_'+role][start:stop]
            w=f['W_'+role][start:stop]
            y=f['Y_'+role][start:stop,0]
            frame=pd.DataFrame(np.column_stack([a,x,w,y]),columns=['local_unit','cycle']+SENSORS+SETTINGS+['rul_raw'])
            grouped=frame.groupby(['local_unit','cycle'],sort=True)
            sums=grouped[SENSORS+SETTINGS].sum()
            sums['n_samples']=grouped.size()
            sums['y_min']=grouped.rul_raw.min();sums['y_max']=grouped.rul_raw.max()
            parts.append(sums.reset_index())
    grouped=pd.concat(parts,ignore_index=True).groupby(['local_unit','cycle'],sort=True)
    out=grouped[SENSORS+SETTINGS+['n_samples']].sum()
    for c in SENSORS+SETTINGS:
        out[c]/=out.n_samples
    low,high=grouped.y_min.min(),grouped.y_max.max()
    if not np.allclose(low,high):
        raise ValueError('RUL changes inside a flight; aggregation protocol invalid')
    out['rul_raw']=low
    out=out.reset_index()
    out['unit_id']=index*1000+out.local_unit.astype(int)
    out['dataset_id']=f'NCMAPSS_DS{index:02d}'
    out['role']=role
    out['cycle']=out.cycle.astype(int)
    out['rul']=np.minimum(out.rul_raw,125.)
    for _,g in out.groupby('unit_id'):
        if not np.all(np.diff(g.cycle)==1) or not np.allclose(np.diff(g.rul_raw),-1):
            raise ValueError('Flight cycles must be consecutive and uncapped RUL decrease by one')
    return out


class Preprocessor:
    def fit(self,frame,seed):
        self.training_units=sorted(frame.unit_id.unique().tolist())
        self.wscale=StandardScaler().fit(frame[SETTINGS])
        w=self.wscale.transform(frame[SETTINGS])
        self.reg=Ridge(alpha=10.).fit(w,frame[SENSORS])
        self.xscale=StandardScaler().fit(frame[SENSORS].to_numpy()-self.reg.predict(w))
        self.conditions=KMeans(n_clusters=3,random_state=seed,n_init=10).fit(w)
        f=self.transform(frame)
        slopes=np.concatenate([np.diff(g[SENSORS].to_numpy(),axis=0) for _,g in f.groupby('unit_id')])
        self.slope_mean=np.mean(slopes,axis=0)
        self.slope_std=np.maximum(np.std(slopes,axis=0),1e-3)
        return self

    def transform(self,frame):
        f=frame.copy()
        w=self.wscale.transform(f[SETTINGS])
        f[SENSORS]=self.xscale.transform(f[SENSORS].to_numpy()-self.reg.predict(w))
        f[SETTINGS]=w
        f['condition_id']=self.conditions.predict(w)
        return f


def corrupt(values,cycles,unit,kind,severity,seed):
    """Absolute-time corruption specification independent of future lifetime."""
    x=np.array(values,float,copy=True);mask=np.ones_like(x,bool)
    rng=np.random.default_rng(seed)
    level={'none':0,'low':1,'medium':2,'high':3}[severity]
    channels=[];onset=None;duration=None;amplitude=None
    if kind!='clean':
        onset=int(rng.integers(35,56))
        number=([1,3,5][level-1] if kind=='channel_dropout' else [1,3,4][level-1])
        channels=sorted(rng.choice(x.shape[1],number,replace=False).tolist())
        if kind=='burst_missing':
            duration=[8,16,24][level-1]
            ix=np.flatnonzero((cycles>=onset)&(cycles<onset+duration))
            mask[np.ix_(ix,channels)]=False
        elif kind=='channel_dropout':
            ix=np.flatnonzero(cycles>=onset)
            mask[np.ix_(ix,channels)]=False
        elif kind=='gradual_bias':
            amplitude=[.5,1.,2.][level-1]
            ramp=np.clip((cycles-onset)/30.,0,1)*amplitude
            x[:,channels]+=ramp[:,None]*rng.choice([-1.,1.],len(channels))[None,:]
        else:
            raise ValueError(kind)
    manifest=CorruptionManifest(unit_id=int(unit),corruption_type=kind,severity=severity,
        sensor_indices=tuple(channels),onset_cycle=onset,duration=duration,amplitude_std=amplitude,
        seed=int(seed),corruption_id=f'u{unit}_{kind}_{severity}_s{seed}')
    return x,mask,manifest


def windows(frame,prep,scenarios,seed,stride):
    transformed=prep.transform(frame)
    trajectories=[]
    for kind,severity in scenarios:
        for unit,g in transformed.groupby('unit_id',sort=True):
            g=g.sort_values('cycle')
            cycles=g.cycle.to_numpy(int)
            x,mask,manifest=corrupt(g[SENSORS].to_numpy(),cycles,int(unit),kind,severity,derive_seed(seed,int(unit),kind,severity))
            filled,gaps=causal_fill(x,mask)
            trajectories.append(PreparedTrajectory(int(unit),cycles,g.rul.to_numpy(),filled,g[SETTINGS].to_numpy(),
                mask,gaps,g.condition_id.to_numpy(int),manifest))
    records=build_windows(trajectories,30,stride,True,prep.slope_mean,prep.slope_std,None,seed)
    # Deliberately remove the original helper's optional terminal off-grid query.
    return [r for r in records if (r.cycle-1)%stride==0]


def adaptation_split(cal,seed):
    rng=np.random.default_rng(seed+600);adapt=[]
    for _,g in cal.groupby('dataset_id',sort=True):
        units=sorted(g.unit_id.unique())
        if len(units)!=2:raise ValueError('Need two reserve engines per subset')
        adapt.append(int(rng.choice(units)))
    threshold=sorted(set(cal.unit_id)-set(adapt))
    return sorted(adapt),threshold

def fitting_rows(f,protocol):
    mask=f.corruption_severity.isin(['none','low','medium'])
    if protocol!='matched':mask &= f.corruption_type.ne(protocol)
    return f.loc[mask].reset_index(drop=True)

"""Training configuration and equal-engine weights for risk estimation."""
import numpy as np
import pandas as pd

from shr.io import load_config
from shr.paths import CONFIGS


def settings():
    return load_config(CONFIGS / "history.yaml")


def engine_weights(table: pd.DataFrame) -> np.ndarray:
    """Equal engines, equal synthetic trajectories within each engine, then equal rows."""
    nrows = table.groupby(['unit_id', 'corruption_id']).cycle.transform('size').to_numpy()
    ntraj = table.groupby('unit_id').corruption_id.transform('nunique').to_numpy()
    w = 1.0 / (nrows * ntraj)
    return w / w.mean()


def balanced_training(oof,adapt,adapt_mass=.5):
    frames=[oof,adapt];weights=[]
    for f,mass in zip(frames,[1-adapt_mass,adapt_mass]):
        w=engine_weights(f);weights.append(w/w.sum()*mass)
    f=pd.concat(frames,ignore_index=True);w=np.concatenate(weights)*len(f)
    return f,w


def role_weights(f,adapt_units):
    w=engine_weights(f);adapt=f.unit_id.isin(adapt_units).to_numpy()
    for flag in [False,True]:
        mask=adapt==flag
        if mask.any():w[mask]*=.5*len(f)/w[mask].sum()
    return w


def tree_options(config, seed, n_estimators=None):
    names = ['n_estimators', 'max_depth', 'learning_rate', 'min_child_weight',
             'subsample', 'colsample_bytree', 'reg_lambda', 'n_jobs']
    options = {n: config[n] for n in names}
    options.update(random_state=seed, tree_method='hist', verbosity=0)
    if n_estimators is not None:
        options['n_estimators'] = n_estimators
    return options

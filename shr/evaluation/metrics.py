"""Tie-aware risk curves and loss aggregation for fixed-predictor comparisons."""
from __future__ import annotations

import numpy as np


class RiskCurve:
    """Exact fractional/tie-averaged risk curve with optional engine resampling weights.

    Tied scores are exchangeable. Neither engine row order nor true loss breaks ties.
    Fractional acceptance is an offline ranking diagnostic, not a deployment action.
    """

    def __init__(self, loss, scores):
        self.loss = np.asarray(loss, float)
        scores = np.asarray(scores, float)
        if len(scores) != len(self.loss) or not len(scores) or not np.isfinite(scores).all():
            raise ValueError('Scores must be finite, nonempty, and aligned')
        self.order = np.argsort(scores, kind='stable')
        sorted_scores = scores[self.order]
        self.starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1]

    def knots(self, weights=None):
        w = np.ones(len(self.loss)) if weights is None else np.asarray(weights, float)
        if len(w) != len(self.loss) or np.any(w < 0) or not np.isfinite(w).all() or w.sum() <= 0:
            raise ValueError('Invalid sample weights')
        w = w[self.order]
        mass = np.add.reduceat(w, self.starts)
        total_loss = np.add.reduceat(w * self.loss[self.order], self.starts)
        keep = mass > 0
        mass, total_loss = mass[keep], total_loss[keep]
        total = mass.sum()
        return np.r_[0, np.cumsum(mass) / total], np.r_[0, np.cumsum(total_loss) / total]

    def risk(self, coverage, weights=None):
        c = np.asarray(coverage, float)
        if np.any(c <= 0) or np.any(c > 1):
            raise ValueError('Coverage outside (0, 1]')
        x, y = self.knots(weights)
        return np.interp(c, x, y) / c

    def area(self, low=0.1, high=1.0, weights=None, normalized=False) -> float:
        if not 0 < low < high <= 1:
            raise ValueError('Invalid coverage interval')
        x, y = self.knots(weights)
        left, right = np.maximum(x[:-1], low), np.minimum(x[1:], high)
        keep = right > left
        slope = np.diff(y)[keep] / np.diff(x)[keep]
        intercept = y[:-1][keep] - slope * x[:-1][keep]
        result = np.sum(slope * (right[keep] - left[keep]) + intercept * np.log(right[keep] / left[keep]))
        return float(result / (high - low) if normalized else result)


COVERAGES=np.array([.7,.8,.9,1.])


def losses(frame,raw=False):
    e=(frame.rul_pred-(frame.rul_raw if raw else frame.rul_true)).to_numpy(float)
    return dict(mse=e*e,positive=np.maximum(e,0),nasa=np.expm1(np.where(e<0,-e/13,e/10)))


def area(score,loss):
    if len(score)==0 or len(score)!=len(loss):
        raise ValueError('Risk area requires nonempty aligned scores and losses')
    if not np.isfinite(score).all() or not np.isfinite(loss).all():
        raise ValueError('Risk area requires finite scores and losses')
    return RiskCurve(loss,score).area(.7,.9,normalized=True)


def evaluate(frame,score,cost,raw=False):
    if len(frame)==0 or len(frame)!=len(score):
        raise ValueError('No accepted-risk estimate is defined for empty or misaligned observations')
    if not np.isfinite(score).all():
        raise ValueError('Non-finite ranking scores')
    ll=losses(frame,raw)
    curves={k:RiskCurve(v,score) for k,v in ll.items()}
    a2=curves['mse'].area(.7,.9,normalized=True)
    ap=curves['positive'].area(.7,.9,normalized=True)
    out=dict(mse_area=a2,positive_area=ap,joint_area=a2+cost*ap,nasa_area=curves['nasa'].area(.7,.9,normalized=True),
             n_queries=len(frame),n_engines=frame.unit_id.nunique())
    for c in COVERAGES:
        label=f'c{int(c*100)}'
        mse=float(curves['mse'].risk(c))
        pos=float(curves['positive'].risk(c))
        nasa=float(curves['nasa'].risk(c))
        out.update({f'rmse_{label}':np.sqrt(mse),f'positive_{label}':pos,f'nasa_{label}':nasa,
                    f'nasa_total_{label}':nasa*c*len(frame),f'joint_{label}':mse+cost*pos})
    assert np.isclose(out['rmse_c100']**2,ll['mse'].mean(),rtol=1e-10,atol=1e-8)
    return out


def weighted_area(score,loss,weights):
    order=np.argsort(score,kind='stable')
    starts=np.r_[0,np.flatnonzero(np.diff(score[order]))+1]
    w=np.asarray(weights,float)[:,order]
    mass=np.add.reduceat(w,starts,axis=1);end=np.cumsum(mass,axis=1);total=end[:,-1]
    assert (total>0).all()
    before=end-mass;x0=before/total[:,None];x1=end/total[:,None]
    left=np.maximum(x0,.7);right=np.minimum(x1,.9);active=(right>left)&(mass>0)
    gl=np.add.reduceat(w*loss[order],starts,axis=1);cumulative=np.cumsum(gl,axis=1)
    slope=np.divide(gl,mass,out=np.zeros_like(gl),where=mass>0)
    intercept=(cumulative-gl)/total[:,None]-slope*x0
    integral=slope*(right-left)+intercept*np.log(np.maximum(right,1e-300)/left)
    return np.where(active,integral,0).sum(axis=1)/.2

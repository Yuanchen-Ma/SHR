"""Conditional moment heads and asymmetric risk reconstruction."""
import numpy as np
from xgboost import XGBRegressor

from .features import AGE, ANCHOR, BASE, STATIC
from .settings import tree_options


CURRENT = STATIC + AGE
HEAD_INPUTS = CURRENT + ANCHOR


def current_features(f):
    """No grouping, previous row, reference query, or outcome is accessed."""
    h = f.copy()
    valid = np.minimum(h.cycle.to_numpy(float), 30)
    h['real_missing_fraction'] = np.clip((h.missing_fraction_all.to_numpy()*30-(30-valid))/valid, 0, 1)
    h['log_missing_age'] = np.log1p(h.max_time_since_observed.to_numpy())
    h['query_maturity'] = np.minimum(h.cycle.to_numpy()/30, 1)
    h['query_log_age'] = np.log1p(h.cycle.to_numpy())
    h['estimated_end'] = (h.cycle.to_numpy()+h.rul_pred.to_numpy())/125
    return h[CURRENT].copy()


def anchor_fit(h, f, weights, config, seed):
    e = (f.rul_pred-f.rul_true).to_numpy()
    x = h[CURRENT].to_numpy(np.float32)
    return [XGBRegressor(objective='reg:squarederror', **tree_options(config, seed)).fit(
        x, y, sample_weight=weights) for y in (e**2, np.maximum(e, 0))]


def anchor_predict(models, h):
    x = h[CURRENT].to_numpy(np.float32)
    return np.column_stack([np.maximum(m.predict(x), 0) for m in models])


def moment_components(sign, heads, h, columns, normalized=True):
    x = h[columns].to_numpy(np.float32); p = h.rul_pred.to_numpy()
    pi = np.clip(predict(sign, x), 0, 1)
    first, second, negative = [predict(m, x) for m in heads]
    if normalized:
        first = np.clip(first, 0, 1)*p
        second = np.clip(second, 0, 1)*p*p
        negative = np.clip(negative, 0, 1)*(125-p)**2
    first = np.clip(first, 0, p)
    second = np.clip(second, first*first, p*p)
    negative = np.clip(negative, 0, (125-p)**2)
    # Equivalent to u2 <- clip(u2, u1**2, u1), including p=0.
    upper = p*first
    projected = np.minimum(second, upper)
    assert np.all(projected >= first*first-1e-9)
    assert np.all(projected <= upper+1e-9)
    old_m = pi*second+(1-pi)*negative
    new_m = pi*projected+(1-pi)*negative
    return dict(m=old_m, projected_m=new_m, positive=pi*first,
                correction=pi*(second-projected), violation=second>upper+1e-9,
                excess=np.maximum(second-upper, 0))


def reference(h, kind, cost):
    prefix = 'empirical_' if kind == 'empirical' else 'pair_'
    return (h[prefix+'mse']+cost*h[prefix+'positive']).to_numpy(np.float32)


def fit_head(x,y,w,cfg,seed,objective):
    if np.ptp(y)<1e-12:return float(y[0])
    weights=w*len(w)/w.sum()
    return XGBRegressor(objective=objective,**tree_options(cfg,seed)).fit(x,y,sample_weight=weights)


def predict(m,x):return np.full(len(x),m) if isinstance(m,float) else m.predict(x)


class MomentHeads:
    """Fit the default sign model and three conditional moment models."""

    def fit(self, features, kind, config, seed):
        self.kind = kind
        self.cols = BASE
        x = features[self.cols].to_numpy(np.float32)
        prediction = features.rul_pred.to_numpy()
        error = (features.rul_pred - features.rul_true).to_numpy()
        weights = features.fit_weight.to_numpy()
        positive = error > 0
        self.sign = fit_head(x, positive.astype(float), weights, config, seed, "binary:logistic")
        upper_positive = np.maximum(prediction, 1e-6)
        upper_negative = np.maximum(125 - prediction, 1e-6)
        parts = [(positive, 1, upper_positive), (positive, 2, upper_positive ** 2),
                 (~positive, 2, upper_negative ** 2)]
        heads = []
        for mask, power, scale in parts:
            target = np.clip(np.abs(error) ** power / scale, 0, 1)
            head = fit_head(x[mask], target[mask], weights[mask], config, seed, "reg:logistic") if mask.any() else 0.
            heads.append(head)
        self.heads = {"normalized_logistic": heads}
        return self

    def score(self, features):
        moments = moment_components(self.sign, self.heads["normalized_logistic"], features, self.cols)
        return np.maximum(.5 * reference(features, self.kind, 20) +
                          .5 * (moments["m"] + 20 * moments["positive"]), 0)

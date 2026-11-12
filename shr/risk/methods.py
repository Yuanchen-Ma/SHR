"""Replace terminal risk heads; retain each method's features and target definition."""
import numpy as np
from .features import AGE, BASE, HISTORY, STATIC
from .learners import Head

METHODS = ['epsilon', 'ccr', 'rcr', 'shr']
LEARNERS = ['xgboost', 'rf', 'logistic', 'knn']


class RiskMethod:
    def __init__(self, name, learner, parameters, seed, kind):
        self.name, self.learner, self.parameters = name, learner, parameters
        self.seed, self.kind = seed, kind
        self.cols = BASE if name == 'shr' else STATIC + HISTORY + ([] if name == 'epsilon' else AGE)

    def fit(self, h):
        x = h[self.cols].to_numpy(np.float32)
        e = (h.rul_pred - h.rul_true).to_numpy(float)
        p = h.rul_pred.to_numpy(float)
        w = h.fit_weight.to_numpy(float)
        self.heads = []
        def fit(y, ww=w, mask=None, objective='fractional', bound=1.):
            take = np.ones(len(x), bool) if mask is None else mask
            if not take.any():
                return 0.
            model = Head(self.learner, self.parameters, self.seed, objective, bound)
            model.fit(x[take], y[take], ww[take])
            return model
        if self.name == 'shr':
            positive = e > 0
            self.heads.append(fit(positive.astype(float), objective='binary'))
            for mask, power, scale in [(positive, 1, np.maximum(p, 1e-6)),
                                       (positive, 2, np.maximum(p, 1e-6)**2),
                                       (~positive, 2, np.maximum(125-p, 1e-6)**2)]:
                self.heads.append(fit(np.clip(np.abs(e)**power / scale, 0, 1), mask=mask))
        elif self.name == 'epsilon':
            self.heads.append(fit(e**2, objective='raw', bound=125**2))
        elif self.name == 'ccr':
            self.heads.append(fit(e**2 + 20*np.maximum(e, 0), objective='raw', bound=125**2+20*125))
        elif self.name == 'rcr':
            loss = e**2 + 20*np.maximum(e, 0)
            self.cost = max(float(np.average(loss, weights=w)), 1.)
            target = self.cost / (loss+self.cost)
            ww = w * (loss+self.cost)
            self.heads.append(fit(target, ww=ww, objective='binary'))
        else:
            raise ValueError(self.name)
        self.actual = [head.actual if not isinstance(head, float) else {'empty_head': head}
                       for head in self.heads]
        return self

    def score(self, h):
        x = h[self.cols].to_numpy(np.float32)
        values = [np.full(len(h), head) if isinstance(head, float) else head.predict(x)
                  for head in self.heads]
        if self.name == 'rcr':
            head = self.heads[0]
            if not isinstance(head, float) and head.constant is None:
                if self.learner == 'xgboost':
                    return -head.model.predict(x, output_margin=True)
                if self.learner == 'logistic':
                    return -head.model.decision_function(head.scaler.transform(x))
            return -values[0]
        if self.name in ['epsilon', 'ccr']:
            return np.maximum(values[0], 0) if self.name == 'epsilon' else values[0]
        p = h.rul_pred.to_numpy(float)
        prob, first, second, negative = [np.clip(v, 0, 1) for v in values]
        first = first * p
        second = np.clip(second*p*p, first*first, p*p)
        negative = negative * (125-p)**2
        positive = prob*first
        risk = prob*second + (1-prob)*negative + 20*positive
        prefix = 'empirical_' if self.kind == 'empirical' else 'pair_'
        reference = (h[prefix+'mse'] + 20*h[prefix+'positive']).to_numpy(np.float32)
        return np.maximum(.5*reference + .5*risk, 0)

"""Weighted conditional-mean learners, including exact fractional logistic fitting."""
import warnings
import hashlib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

NEIGHBOR_CACHE = {}
CACHE_NEIGHBORS = False
MAX_K = 1023


class Head:
    def __init__(self, learner, parameters, seed, objective='fractional', bound=1.):
        self.learner = learner
        self.parameters = dict(parameters)
        self.seed = int(seed)
        self.objective = objective
        self.bound = float(bound)

    def fit(self, x, y, weights):
        x = np.asarray(x, np.float32)
        y = np.asarray(y, float)
        w = np.asarray(weights, float)
        assert len(x) == len(y) == len(w) and len(x)
        assert np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(w).all()
        assert (w > 0).all()
        self.n_fit = len(x)
        self.constant = float(y[0]) if np.ptp(y) < 1e-12 else None
        if self.constant is not None:
            self.actual = {'constant': self.constant, 'n_fit': len(x)}
            return self
        # Match the current SHR head convention for boosting/logistic. Preserve
        # original weights for RF: rescaling is theoretically invariant, but a
        # ~1e-16 perturbation can change tied splits in deeply grown sklearn trees.
        if self.learner != 'rf':
            w = w * len(w) / w.sum()
        p = self.parameters
        if self.learner == 'xgboost':
            objective = {'binary': 'binary:logistic', 'fractional': 'reg:logistic',
                         'raw': 'reg:squarederror'}[self.objective]
            self.model = XGBRegressor(**p, objective=objective, random_state=self.seed,
                                      tree_method='hist', n_jobs=4, verbosity=0)
            self.model.fit(x, y, sample_weight=w)
            self.actual = self.model.get_params()
        elif self.learner == 'rf':
            self.model = RandomForestRegressor(**p, criterion='squared_error',
                                                random_state=self.seed, n_jobs=4)
            self.model.fit(x, y, sample_weight=w)
            self.actual = self.model.get_params()
        elif self.learner == 'logistic':
            z = y / self.bound
            assert z.min() >= -1e-9 and z.max() <= 1 + 1e-9
            z = np.clip(z, 0, 1)
            self.scaler = StandardScaler().fit(x)
            xx = self.scaler.transform(x).astype(np.float64)
            # Two weighted binary copies give exactly the fractional BCE objective:
            # w * [-z log(p) - (1-z) log(1-p)]. No thresholding or pseudo-label draw.
            duplicated = np.concatenate([xx, xx])
            labels = np.r_[np.zeros(len(x), int), np.ones(len(x), int)]
            ww = np.r_[w * (1-z), w * z]
            keep = ww > 0
            self.model = LogisticRegression(C=p['C'], solver='lbfgs', max_iter=4000,
                                             tol=1e-7, random_state=self.seed)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('error', ConvergenceWarning)
                    self.model.fit(duplicated[keep], labels[keep], sample_weight=ww[keep])
            except ConvergenceWarning:
                # Same penalized objective and C; use a second-order solver when
                # weak regularization makes L-BFGS slow. Never accept an unfinished fit.
                self.model = LogisticRegression(C=p['C'], solver='newton-cholesky',
                                                 max_iter=1000, tol=1e-7, random_state=self.seed)
                with warnings.catch_warnings():
                    warnings.simplefilter('error', ConvergenceWarning)
                    self.model.fit(duplicated[keep], labels[keep], sample_weight=ww[keep])
            self.actual = self.model.get_params()
            self.actual.update(n_iter=self.model.n_iter_.tolist(), target_bound=self.bound)
        elif self.learner == 'knn':
            k = p['n_neighbors']
            if k > len(x):
                raise ValueError(f'k={k} exceeds head training rows {len(x)}')
            self.scaler = StandardScaler().fit(x)
            self.x = self.scaler.transform(x)
            self.y, self.w = y, w
            self.model = NearestNeighbors(n_neighbors=k, p=2, algorithm='brute', n_jobs=2).fit(self.x)
            self.actual = dict(p, p=2, n_fit=len(x), sample_weight=True)
        else:
            raise ValueError(self.learner)
        for name, value in p.items():
            assert self.actual[name] == value, (name, self.actual[name], value)
        return self

    def predict(self, x):
        if self.constant is not None:
            return np.full(len(x), self.constant)
        x = np.asarray(x, np.float32)
        if self.learner == 'logistic':
            return self.model.predict_proba(self.scaler.transform(x))[:, 1] * self.bound
        if self.learner != 'knn':
            return self.model.predict(x)
        cache_key = None
        if CACHE_NEIGHBORS:
            query = self.scaler.transform(x)
            cache_key = hashlib.sha256(self.x.tobytes() + query.tobytes()).hexdigest()
            if cache_key not in NEIGHBOR_CACHE:
                distance_parts, index_parts = [], []
                for begin in range(0, len(query), 512):
                    dd, ii = self.model.kneighbors(query[begin:begin+512], n_neighbors=min(MAX_K, len(self.x)))
                    distance_parts.append(dd.astype(np.float32))
                    index_parts.append(ii.astype(np.int32))
                NEIGHBOR_CACHE[cache_key] = (np.concatenate(distance_parts), np.concatenate(index_parts))
        output = []
        for begin in range(0, len(x), 512):
            if cache_key is None:
                distance, index = self.model.kneighbors(self.scaler.transform(x[begin:begin+512]))
            else:
                distance, index = [v[begin:begin+512, :self.parameters['n_neighbors']]
                                   for v in NEIGHBOR_CACHE[cache_key]]
            ww = self.w[index].copy()
            if self.parameters['weights'] == 'distance':
                zero = distance == 0
                has_zero = zero.any(axis=1)
                dd = 1 / np.maximum(distance, 1e-12)
                dd[has_zero] = zero[has_zero]
                ww *= dd
            output.append((ww * self.y[index]).sum(axis=1) / ww.sum(axis=1))
        return np.concatenate(output)

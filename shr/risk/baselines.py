"""Fixed predictor baseline models used as default candidate fits."""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from xgboost import XGBRegressor

from .features import AGE, HISTORY, STATIC
from .settings import tree_options


class Epsilon:
    cols = STATIC + HISTORY

    def fit(self, features, error, weights, config, seed):
        x = features[self.cols].to_numpy(np.float32)
        self.estimator = XGBRegressor(objective="reg:squarederror", **tree_options(config, seed))
        self.estimator.fit(x, error ** 2, sample_weight=weights)
        return self

    def score(self, features):
        return np.maximum(self.estimator.predict(features[self.cols].to_numpy(np.float32)), 0)


class RcR:
    cols = STATIC + HISTORY + AGE

    def fit(self, features, error, weights, config, seed):
        x = features[self.cols].to_numpy(np.float32)
        loss = error ** 2 + 20 * np.maximum(error, 0)
        reject_cost = max(float(np.average(loss, weights=weights)), 1.)
        target = reject_cost / (loss + reject_cost)
        fit_weight = (loss + reject_cost) * weights
        fit_weight *= weights.sum() / fit_weight.sum()
        self.estimator = XGBRegressor(objective="binary:logistic", **tree_options(config, seed))
        self.estimator.fit(x, target, sample_weight=fit_weight)
        return self

    def score(self, features):
        return -self.estimator.predict(features[self.cols].to_numpy(np.float32), output_margin=True)


class CCR:
    cols = STATIC + HISTORY + AGE

    def fit(self, features, error, weights, seed):
        x = features[self.cols].to_numpy(np.float32)
        loss = error ** 2 + 20 * np.maximum(error, 0)
        self.estimator = RandomForestRegressor(n_estimators=100, n_jobs=4, random_state=seed)
        self.estimator.fit(x, loss, sample_weight=weights)
        return self

    def score(self, features):
        return self.estimator.predict(features[self.cols].to_numpy(np.float32))


class LossRejection:
    """Fit linear LossRej and the default SelRej candidate."""

    cols = STATIC + HISTORY + AGE

    def fit(self, features, error, weights, seed):
        self.scaler = StandardScaler().fit(features[self.cols])
        x = self.scaler.transform(features[self.cols]).astype(np.float32)
        squared = error ** 2
        targets = np.column_stack([squared, squared + 10 * np.maximum(error, 0)])
        self.linear = LinearRegression().fit(x, targets, sample_weight=weights)
        torch.set_num_threads(4)
        torch.manual_seed(seed)
        gate = nn.Sequential(nn.Linear(x.shape[1], 64), nn.ReLU(), nn.Linear(64, 1))
        optimizer = torch.optim.Adam(gate.parameters(), lr=.003)
        xx = torch.tensor(x)
        ww = torch.tensor(weights / weights.sum(), dtype=torch.float32)
        yy = torch.tensor(squared / max(np.average(squared, weights=weights), 1), dtype=torch.float32)
        for _ in range(150):
            optimizer.zero_grad()
            acceptance = torch.sigmoid(gate(xx).squeeze(1))
            coverage = (ww * acceptance).sum()
            loss = (ww * yy * acceptance).sum() / coverage.clamp_min(1e-6)
            loss = loss + 200 * torch.relu(.8 - coverage) ** 2
            loss.backward()
            optimizer.step()
        self.gates = {0: gate.eval()}
        return self

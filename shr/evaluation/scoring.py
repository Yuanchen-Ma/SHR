"""Reconstruct the complete SHR acceptance score from fitted risk models."""

import numpy as np

from shr.risk.moments import moment_components, reference
from shr.risk.training import NativeAdapter, fitted as fit, value as validation_value


def components(model, features):
    if isinstance(model,NativeAdapter):
        native = model.model
        moments = moment_components(native.sign, native.heads["normalized_logistic"], features, native.cols)
        return moments["projected_m"], moments["positive"]
    x = features[model.cols].to_numpy(np.float32)
    values = [np.full(len(features), head) if isinstance(head, float) else head.predict(x)
              for head in model.heads]
    prob, first, positive_second, negative_second = [np.clip(value, 0, 1) for value in values]
    prediction = features.rul_pred.to_numpy(float)
    positive_second = np.minimum(first, np.maximum(positive_second, first * first))
    expected_squared = (prob * prediction * prediction * positive_second
                        + (1 - prob) * (125 - prediction) ** 2 * negative_second)
    return expected_squared, prob * prediction * first


def score(model, features, method, cost=20, alpha=.5, kind=None):
    if method != "shr":
        return model.score(features)
    squared, positive = components(model, features)
    model_kind = model.model.kind if isinstance(model, NativeAdapter) else model.kind
    ref = reference(features, kind or model_kind, cost)
    return np.maximum((1 - alpha) * ref + alpha * (squared + cost * positive), 0)

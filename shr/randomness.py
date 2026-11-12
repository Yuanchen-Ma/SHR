from __future__ import annotations

import hashlib
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def derive_seed(seed: int, *parts: object) -> int:
    text = ":".join([str(seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def stable_hash(values: list[int] | tuple[int, ...]) -> str:
    payload = ",".join(str(value) for value in sorted(values))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

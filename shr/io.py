"""Job identifiers, configuration loading and atomic metadata writes."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile

import yaml


def key(job):
    return "/".join(str(job[name]) for name in ("benchmark", "cohort", "architecture", "seed", "protocol"))


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
        temp = Path(stream.name)
        try:
            json.dump(value, stream, indent=2, ensure_ascii=False, default=str)
            stream.write("\n")
        except BaseException:
            stream.close()
            temp.unlink(missing_ok=True)
            raise
    try:
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def load_config(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def mark_complete(destination, signature, extra=None):
    artifacts = {
        path.relative_to(destination).as_posix(): sha(path)
        for path in destination.rglob("*")
        if path.is_file() and path.name != "completion.json" and path.suffix != ".tmp"
    }
    write(destination / "completion.json",
          dict(extra or {}, signature=signature, completed_at=now(), artifacts=artifacts))


def completed(destination, signature):
    marker = destination / "completion.json"
    if not marker.exists():
        return False
    metadata = read(marker)
    if metadata["signature"] != signature:
        raise ValueError(f"Input or configuration changed for {destination}")
    for name, expected in metadata["artifacts"].items():
        path = destination / name
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f"Completed predictor artifact is missing or changed: {path}")
    return True

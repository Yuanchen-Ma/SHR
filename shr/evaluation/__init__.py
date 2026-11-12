"""Validation, held-out evaluation and statistical checks for SHR."""
from shr.paths import EXPERIMENTS, RESULTS

PRIMARY = EXPERIMENTS / "primary"
SELECTION = EXPERIMENTS / "selection"
LEARNERS = EXPERIMENTS / "learner_comparison"
SELREJ = EXPERIMENTS / "selrej"
BOOTSTRAP = EXPERIMENTS / "bootstrap"
SELREJ_BOOTSTRAP = EXPERIMENTS / "selrej_bootstrap"
METRICS = RESULTS / "metrics"
DIAGNOSTICS = RESULTS / "diagnostics"

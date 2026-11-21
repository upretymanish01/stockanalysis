"""Research dataset, purged evaluation, and an append-only prediction ledger.

Import this package directly. It is not wired into live order routing.
"""

from .errors import classify_outcome
from .ledger import PredictionLedger
from .models import evaluate_horizon

__all__ = ["PredictionLedger", "classify_outcome", "evaluate_horizon"]

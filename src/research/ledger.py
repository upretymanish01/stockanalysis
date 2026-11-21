"""Append-only prediction ledger.

A prediction line is never rewritten. Settlement is a later line. Replaying
the log rebuilds the current view. A second prediction with the same id and
different forecast fields is rejected, including after the outcome is known.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

PREDICTED_FIELDS = (
    "prediction_id",
    "as_of",
    "ticker",
    "model_version",
    "horizon",
    "predicted_return",
    "predicted_probability",
    "entry_assumption",
    "exit_assumption",
)


class LedgerError(ValueError):
    pass


class PredictionLedger:
    def __init__(self, path: Optional[str | Path] = None):
        self.path = Path(path) if path is not None else None
        self.predictions: dict[str, dict] = {}
        if self.path is not None and self.path.exists():
            self._replay(self.path.read_text().splitlines())

    def add(self, record: dict) -> dict:
        stored, changed = self._insert(record)
        if changed:
            self._append({"op": "predict", **{key: stored[key] for key in PREDICTED_FIELDS}, "status": "pending"})
        return stored

    def _insert(self, record: dict) -> tuple[dict, bool]:
        missing = [key for key in PREDICTED_FIELDS if key not in record]
        if missing:
            raise LedgerError(f"Prediction is missing {missing}")
        prediction_id = str(record["prediction_id"])
        stored = {key: record[key] for key in PREDICTED_FIELDS}
        stored["status"] = "pending"
        if prediction_id in self.predictions:
            current = self.predictions[prediction_id]
            for key in PREDICTED_FIELDS:
                if current[key] != stored[key]:
                    raise LedgerError(
                        f"Prediction {prediction_id} already archived; {key} cannot change"
                    )
            return current, False
        self.predictions[prediction_id] = stored
        return stored, True

    def settle(
        self,
        prediction_id: str,
        realized_net: float,
        benchmark_return: float,
        sessions_since_decision: int,
    ) -> dict:
        current, settlement, changed = self._settle(
            prediction_id, realized_net, benchmark_return, sessions_since_decision
        )
        if changed:
            self._append(settlement)
        return current

    def _settle(
        self,
        prediction_id: str,
        realized_net: float,
        benchmark_return: float,
        sessions_since_decision: int,
    ) -> tuple[dict, dict, bool]:
        if prediction_id not in self.predictions:
            raise LedgerError(f"Unknown prediction {prediction_id}")
        current = self.predictions[prediction_id]
        if int(sessions_since_decision) < int(current["horizon"]):
            raise LedgerError(
                "Outcome is not observable yet: the holding period has not elapsed"
            )
        status = "correct" if float(realized_net) > 0 else "incorrect"
        settlement = {
            "op": "settle",
            "prediction_id": prediction_id,
            "realized_net": float(realized_net),
            "benchmark_return": float(benchmark_return),
            "prediction_error": float(realized_net) - float(current["predicted_return"]),
            "status": status,
            "sessions_since_decision": int(sessions_since_decision),
        }
        if current["status"] != "pending":
            for key in ("realized_net", "benchmark_return", "status"):
                if current[key] != settlement[key]:
                    raise LedgerError(f"Settlement for {prediction_id} is already archived")
            return current, settlement, False
        current.update(
            {
                "realized_net": settlement["realized_net"],
                "benchmark_return": settlement["benchmark_return"],
                "prediction_error": settlement["prediction_error"],
                "status": status,
            }
        )
        return current, settlement, True

    def _append(self, payload: dict) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def _replay(self, lines: list[str]) -> None:
        for line in lines:
            if not line.strip():
                continue
            payload = json.loads(line)
            op = payload.get("op")
            if op == "predict":
                self._insert({key: payload[key] for key in PREDICTED_FIELDS})
            elif op == "settle":
                self._settle(
                    payload["prediction_id"],
                    payload["realized_net"],
                    payload["benchmark_return"],
                    payload["sessions_since_decision"],
                )
            else:
                raise LedgerError(f"Unknown ledger operation {op}")

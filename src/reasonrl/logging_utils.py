"""Structured logging: human-readable stdout + machine-readable JSONL metrics."""

from __future__ import annotations

import json
import logging
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_FMT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def get_logger(name: str = "reasonrl", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FMT, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level)
    return logger


def capture_environment() -> dict:
    """Versions + git SHA, written next to every run for provenance."""
    env: dict[str, Any] = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda"] = torch.version.cuda if torch.cuda.is_available() else None
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:  # pragma: no cover
        pass
    try:
        import transformers

        env["transformers"] = transformers.__version__
    except ImportError:  # pragma: no cover
        pass
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=False
        ).stdout.strip()
        env["git_sha"] = sha or None
    except (OSError, subprocess.SubprocessError):
        env["git_sha"] = None
    return env


def _jsonable(v: Any) -> Any:
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


class MetricsLogger:
    """Append-only JSONL metrics log (one record per step) plus console output."""

    def __init__(
        self,
        out_dir: str | Path | None,
        config: dict | None = None,
        log_every: int = 1,
        name: str = "reasonrl",
    ) -> None:
        self.logger = get_logger(name)
        self.log_every = max(1, log_every)
        self.out_dir = Path(out_dir) if out_dir else None
        self._fh = None
        self._t0 = time.time()
        if self.out_dir is not None:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.out_dir / "metrics.jsonl", "a", encoding="utf-8")
            if config is not None:
                with open(self.out_dir / "config.json", "w", encoding="utf-8") as fh:
                    json.dump(config, fh, indent=2, default=str)
            with open(self.out_dir / "env.json", "w", encoding="utf-8") as fh:
                json.dump(capture_environment(), fh, indent=2)
        self.history: list[dict] = []

    def log(self, step: int, metrics: dict, prefix: str = "") -> None:
        record = {"step": step, "time_s": round(time.time() - self._t0, 3)}
        record.update({f"{prefix}{k}": _jsonable(v) for k, v in metrics.items()})
        self.history.append(record)
        if self._fh is not None:
            self._fh.write(json.dumps(record) + "\n")
            self._fh.flush()
        if step % self.log_every == 0:
            shown = " ".join(
                f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                for k, v in record.items()
                if k not in ("time_s",) and v is not None
            )
            self.logger.info(shown)

    def info(self, msg: str) -> None:
        self.logger.info(msg)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

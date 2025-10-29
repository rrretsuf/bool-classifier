from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os
from typing import Optional

def _env_float(name: str, default: float) -> float:
    try:
        v = os.getenv(name)
        return float(v) if v is not None else default
    except ValueError:
        return default

def _env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v else default

def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")

@dataclass(frozen=True)
class Paths:
    root: Path
    artifacts: Path
    embedder_dir: Path
    model_file: Path
    labels_file: Path
    metrics_file: Path
    reliability_png: Path
    logs_dir: Path
    borderline_log: Path

    @staticmethod
    def init(base: Optional[str] = None) -> Path: 
        root = Path(base) if base else Path.cwd()
        artifacts = root / "artifacts"
        embedder_dir = artifacts / "embedder"
        model_file = artifacts / "model.joblib"
        labels_file = artifacts / "labels.json"
        metrics_file = artifacts / "metrics.json"
        reliability_png = artifacts / "reliability.png"

        logs_dir = root / "logs"
        # tukaj se zapišejo vsi primeri pri katerih model ni bil dovolj samozavesten 
        borderline_logs = logs_dir / "borderline.jsonl"

        # ustvarijo se mape, če še ne obstajajo
        artifacts.mkdir(parents=True, exist_ok=True)
        embedder_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        return Paths(
            root=root,
            artifacts=artifacts,
            embedder_dir=embedder_dir,
            model_file=model_file,
            labels_file=labels_file,
            metrics_file=metrics_file,
            reliability_png=reliability_png,
            logs_dir=logs_dir,
            borderline_log=borderline_logs
        )

@dataclass(frozen=True)
class Config:
    # model & embeddings
    embedder_name: str
    # threshold za routing
    low_thresh: float
    high_thresh: float
    # openrouter config
    openrouter_api_key: Optional[str]
    openrouter_model: str
    # paths
    paths: Paths
    # misc
    debug: bool

def load_config(base_dir: Optional[str] = None) -> Config:
    """
    naloži config iz .env in postavi make.
    """
    embedder_name = _env_str("EMBEDDER_NAME", "BAAI/bge-m3")
    low_thresh = _env_float("LOW_THRESH", 0.35)
    high_thresh = _env_float("HIGH_THRESH", 0.65)

    openrouter_api_key = os.getenv("OPENROUTER_API_KEY") # lahko je None
    openrouter_model = _env_str("OPENROUTER_MODEL", "x-ai/grok-4-fast")

    debug = _env_bool("DEBUG", False)

    paths = Paths.init(base=base_dir)

    return Config(
        embedder_name=embedder_name,
        low_thresh=low_thresh,
        high_thresh=high_thresh,
        openrouter_api_key=openrouter_api_key,
        openrouter_model=openrouter_model,
        paths=paths,
        debug=debug
    )
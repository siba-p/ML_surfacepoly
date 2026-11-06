"""Central definition of repository paths, so no module depends on the current working directory."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
SPLITS = DATA / "splits"
MODELS = ROOT / "models"
PRETRAINED = MODELS / "pretrained"
RUNS = MODELS / "runs"

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = BASE_DIR / "experiments"
SEED = 42
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
RANDOM_STATE = SEED

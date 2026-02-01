from pathlib import Path
import sys

# Ensure repo root is importable
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from src.train import main

if __name__ == "__main__":
    cfg = "configs/baseline.yaml"
    if len(sys.argv) >= 2:
        cfg = sys.argv[1]
    main(cfg)

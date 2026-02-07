from pathlib import Path

def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]  # if this file lives in src/utils/
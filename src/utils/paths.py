from pathlib import Path

def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]  # if this file lives in src/utils/

def resolve_repo(path_str: str | None) -> Path | None:
    """
    Resolve a path relative to repo root.
    If absolute, return as-is.
    """
    if not path_str:
        return None

    p = Path(path_str).expanduser()
    if p.is_absolute():
        return p

    return (repo_root() / p).resolve()
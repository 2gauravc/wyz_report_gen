from pathlib import Path
from datetime import datetime
import uuid


def generate_report_filename(
    out_dir: Path,
    prefix: str = "report_output",
    ext: str = ".csv"
) -> Path:
    """
    Generate a unique report filename with timestamp + short random suffix.
    Returns full Path.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_suffix = uuid.uuid4().hex[:6]

    filename = f"{prefix}_{timestamp}_{random_suffix}{ext}"
    return out_dir / filename

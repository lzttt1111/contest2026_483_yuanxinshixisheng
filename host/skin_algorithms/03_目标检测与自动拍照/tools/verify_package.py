import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "verify_submission.py"), run_name="__main__")

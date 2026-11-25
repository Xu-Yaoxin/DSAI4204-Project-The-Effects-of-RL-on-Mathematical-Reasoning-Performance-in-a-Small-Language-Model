"""Verify the relocated project without scoring or reading Downloads."""
from pathlib import Path
import runpy

project = Path(__file__).resolve().parents[2]
runpy.run_path(str(project / "maintenance/verify_project.py"), run_name="__main__")

"""Thin wrapper: enables running the CLI without pip-installing the package."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from despotx.cli import main

if __name__ == "__main__":
    main()

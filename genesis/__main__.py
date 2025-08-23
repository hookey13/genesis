"""
Main entry point for Project GENESIS.

This module serves as the entry point when running the application
with `python -m genesis`.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main() -> int:
    """Main entry point for the application."""
    print("Project GENESIS - Initializing...")
    print("Trading system is not yet implemented.")
    print("Run 'make help' to see available commands.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
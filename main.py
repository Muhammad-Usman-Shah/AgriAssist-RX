
"""
AgriAssist-RX
Main application entry point.

Starts the six-route agricultural AI agent.
"""

import sys
from pathlib import Path


def main():

    project_root = Path(__file__).resolve().parent
    source_folder = project_root / "src"

    if str(source_folder) not in sys.path:
        sys.path.insert(0, str(source_folder))

    from agent_csv import main as run_agent

    run_agent()


if __name__ == "__main__":
    main()

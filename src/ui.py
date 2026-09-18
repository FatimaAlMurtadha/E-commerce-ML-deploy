"""Entry point for the Streamlit UI."""

import sys
from pathlib import Path

# Ensure project root and Frontend directory are in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = PROJECT_ROOT / "Frontend"

for p in (str(PROJECT_ROOT), str(FRONTEND_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from Frontend.ui import main
except ImportError:
    try:
        from ui import main
    except ImportError:
        def main():
            import streamlit as st
            st.error("Frontend UI module not found.")

if __name__ == "__main__":
    main()
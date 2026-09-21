"""Streamlit entrypoint used when the UI module is imported by tests."""
import sys
import os
 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 
from ui import main


if __name__ == "__main__":
    main()

"""Allow running as: python -m csi_node [--demo|--dashboard|...]"""
import sys
import os

# Add parent to path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run import main
main()

import sys
import os

# Add the project root (parent of tests/) to sys.path so test files can
# import project modules and scripts/ helpers directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

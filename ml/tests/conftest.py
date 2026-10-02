import os
import sys

# Make `import ml.xxx` work when pytest is run from the project root or from ml/.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

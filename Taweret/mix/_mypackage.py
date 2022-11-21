
from pathlib import Path
import sys

cwd = Path(__file__).absolute().parent
sys.path.append(cwd.parent.parent.__str__())

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import configs.laplex_unfrozen as config
from models import LaplexLinear
from train.run import run

if __name__ == "__main__":
    run(config, head_cls=LaplexLinear, optimizer_type="laplex")

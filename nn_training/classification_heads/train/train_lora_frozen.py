import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import configs.lora_frozen as config
from models import LoRALinear
from train.run import run

if __name__ == "__main__":
    run(config, head_cls=LoRALinear, optimizer_type="lora")

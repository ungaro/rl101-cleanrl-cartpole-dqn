"""Windows wrapper for Holosoma training.

Works around two Windows-specific issues:
1. Triton is unavailable on Windows, so torch.compile (dynamo) must be disabled.
2. PyTorch's distribution validation fails with bfloat16 on some configurations.

Usage (from repo root):
    set TORCHDYNAMO_DISABLE=1
    E:\\rl101-crash-course\\external\\holosoma\\.venv\\hsmujoco\\Scripts\\python.exe ^
        scripts\\holosoma_train_win.py exp:g1-29dof-fast-sac simulator:mjwarp --training.seed 1
"""

import os
import sys

# Disable torch.compile / dynamo (Triton not available on Windows)
os.environ["TORCHDYNAMO_DISABLE"] = "1"

# Disable bfloat16 distribution validation (fails on some GPU/driver combos)
import torch
torch.distributions.Distribution.set_default_validate_args(False)

# Delegate to holosoma's actual training entry point
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "external", "holosoma", "src", "holosoma"))
from holosoma.train_agent import main

if __name__ == "__main__":
    main()

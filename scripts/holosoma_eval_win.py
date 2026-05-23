"""Windows wrapper for Holosoma evaluation/visualization.

Applies the same Windows workarounds as holosoma_train_win.py:
1. Disables torch.compile (Triton unavailable on Windows)
2. Fixes emoji crash (cp1252 can't encode unicode)
3. Disables bfloat16 distribution validation

Usage (from repo root):
    set TORCHDYNAMO_DISABLE=1
    E:\\rl101-crash-course\\external\\holosoma\\.venv\\hsmujoco\\Scripts\\python.exe ^
        scripts\\holosoma_eval_win.py ^
        --checkpoint logs\\hv-g1-manager\\<run>\\model_XXXX.pt ^
        simulator:mujoco
"""

import os
import sys

# Disable torch.compile / dynamo (Triton not available on Windows)
os.environ["TORCHDYNAMO_DISABLE"] = "1"

# Fix emoji crash on Windows console (cp1252 can't encode unicode emoji)
os.environ["PYTHONIOENCODING"] = "utf-8"
# Also fix stdout/stderr encoding for the current process
if sys.stdout.encoding != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Disable bfloat16 distribution validation (fails on some GPU/driver combos)
import torch
torch.distributions.Distribution.set_default_validate_args(False)

# Delegate to holosoma's actual eval entry point
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "external", "holosoma", "src", "holosoma"))
from holosoma.eval_agent import main

if __name__ == "__main__":
    main()

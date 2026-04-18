#!/usr/bin/env bash
set -euo pipefail

echo "============================================================"
echo "cleanrl-cartpole-dqn — Setup"
echo "============================================================"

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

ENV_NAME="rl101"

# ---------------------------------------------------------------
# 1. Create conda environment with Python 3.10
#    CleanRL requires python >=3.8,<3.11
# ---------------------------------------------------------------
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "[1/6] Conda env '${ENV_NAME}' already exists."
else
    echo "[1/6] Creating conda env '${ENV_NAME}' with Python 3.10..."
    conda create -y -n "$ENV_NAME" python=3.10
fi

# Activate inside this script
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

echo "       Python: $(python --version) ($(which python))"

# ---------------------------------------------------------------
# 2. Clone CleanRL if not present
# ---------------------------------------------------------------
if [ -d "cleanrl" ]; then
    echo "[2/6] CleanRL already cloned."
else
    echo "[2/6] Cloning CleanRL..."
    git clone https://github.com/vwxyzjn/cleanrl.git
fi

# ---------------------------------------------------------------
# 3. Install PyTorch (platform-aware)
#    - Linux: nightly cu128 for RTX 5090 (Blackwell SM 12.0)
#    - macOS: stable (includes MPS/Metal support on Apple Silicon)
#    - other: stable CPU build
# ---------------------------------------------------------------
OS="$(uname -s)"
ARCH="$(uname -m)"
case "$OS" in
    Linux)
        echo "[3/6] Installing PyTorch nightly cu128 (Linux, for RTX 5090)..."
        pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
        ;;
    Darwin)
        echo "[3/6] Installing PyTorch stable (macOS $ARCH, with MPS support)..."
        pip install torch
        ;;
    *)
        echo "[3/6] Installing PyTorch stable (unknown OS '$OS')..."
        pip install torch
        ;;
esac

# ---------------------------------------------------------------
# 4. Install CleanRL with --no-deps (avoids pinned torch==2.4.1)
#    This puts cleanrl_utils on the path without version conflicts
# ---------------------------------------------------------------
echo "[4/6] Installing CleanRL (--no-deps)..."
cd cleanrl
pip install --no-deps -e .
cd "$PROJECT_DIR"

# ---------------------------------------------------------------
# 5. Install runtime dependencies directly
#    These are what dqn.py actually needs, unpinned
# ---------------------------------------------------------------
echo "[5/6] Installing runtime dependencies..."
# Pin gymnasium==0.29.1 — CleanRL's dqn_eval.py uses the old
# `infos["final_info"]` API which was removed in gymnasium 1.0+
pip install \
    "gymnasium[classic-control]==0.29.1" \
    tensorboard \
    tyro \
    wandb \
    moviepy \
    pygame \
    "rich<12.0" \
    numpy \
    huggingface-hub \
    tenacity

# Atari (optional, for Week 3 Atari PPO demos)
# ale-py provides the Atari Learning Environment; autorom downloads the ROMs.
if pip install "gymnasium[atari]==0.29.1" "autorom[accept-rom-license]" opencv-python-headless 2>/dev/null; then
    echo "       Atari deps installed — Breakout/Pong/SpaceInvaders demos will work."
    # Import ROMs so gymnasium can find them
    python -c "import ale_py" 2>/dev/null || true
else
    echo ""
    echo "       WARNING: Atari deps install failed."
    echo "       CartPole will work fine without them. For Atari games:"
    echo "         pip install 'gymnasium[atari]==0.29.1' 'autorom[accept-rom-license]'"
    echo ""
fi

# Box2D (optional, for LunarLander bonus demo)
# box2d-py builds from source via swig. The `pip install swig` package
# is a broken shim — the real fix is the system swig binary:
#   macOS:  brew install swig
#   Linux:  apt install swig  (or dnf / pacman equivalent)
# If swig isn't present, box2d-py build fails — we just warn and continue
# since CartPole (the main demo) doesn't need it.
if pip install box2d-py 2>/dev/null; then
    echo "       box2d-py installed — LunarLander bonus demo will work."
else
    echo ""
    echo "       WARNING: box2d-py install failed (needs system 'swig')."
    echo "       CartPole will work fine without it. For LunarLander:"
    case "$OS" in
        Darwin) echo "         brew install swig && pip install box2d-py" ;;
        Linux)  echo "         sudo apt install swig && pip install box2d-py" ;;
        *)      echo "         install 'swig' from your package manager, then: pip install box2d-py" ;;
    esac
    echo ""
fi

# ---------------------------------------------------------------
# 6. Verification
# ---------------------------------------------------------------
echo "[6/6] Verifying installation..."

echo ""
echo "--- PyTorch / GPU ---"
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
if torch.cuda.is_available():
    print(f'CUDA available: True')
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    cap = torch.cuda.get_device_capability(0)
    print(f'Compute capability: {cap[0]}.{cap[1]}')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('MPS (Apple Metal) available — Apple Silicon GPU will be used')
else:
    print('No GPU detected — training will run on CPU')
    print('(this is fine for CartPole; its Q-network is tiny)')
"

echo ""
echo "--- Gymnasium ---"
python -c "
import gymnasium as gym
env = gym.make('CartPole-v1')
obs, _ = env.reset()
print(f'CartPole-v1: observation shape={obs.shape}, actions={env.action_space.n}')
env.close()
print('Gymnasium OK.')
"

echo ""
echo "--- CleanRL ---"
if [ -f "cleanrl/cleanrl/dqn.py" ]; then
    echo "CleanRL dqn.py found. OK."
else
    echo "ERROR: cleanrl/cleanrl/dqn.py not found!"
    exit 1
fi

python -c "
from cleanrl_utils.buffers import ReplayBuffer
print('cleanrl_utils importable. OK.')
"

echo ""
echo "============================================================"
echo "Setup complete!"
echo ""
echo "Activate the environment first:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "Then try:"
echo "  make random      — watch random agent fail"
echo "  make train       — train DQN on CartPole"
echo "  make tensorboard — watch training metrics"
echo "============================================================"

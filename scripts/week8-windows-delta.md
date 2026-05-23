# Week 8 Demos -- Windows Compatibility Delta

Tested on: Windows 11 (build 26200), RTX 5090 32 GB, NVIDIA Driver 591.74, CUDA 13.1, Isaac Sim 5.1.0-rc.19, conda 23.9.0, uv 0.11.7.

---

## Summary

| Demo | Status | Blockers |
|------|--------|----------|
| 1a. mjlab sanity (`uvx demo`) | Works with workarounds | Missing scipy dep, cp1252 emoji crash |
| 1b. G1 spin kick replay | Partially blocked | W&B artifacts private to `gcbc_researchers` |
| 2. Isaac Lab ANYmal-D | Works with workarounds | flatdict build-isolation bug, env var setup |
| 3. Holosoma G1 FastSAC | Partially blocked | Setup script Linux-only, Triton missing, bfloat16 bug, emoji crash, version conflicts, NaN rewards (mujoco-warp 0.0.2 tensor bug), eval viewer CPU-only |

---

## Demo 1 -- mjlab / G1 spin kick

### What the docs say

```bash
uvx --from mjlab --refresh demo
```

### What breaks on Windows

1. **`ModuleNotFoundError: No module named 'scipy'`**
   scipy is not in mjlab's declared dependencies. On Linux it may be pulled in transitively; on a clean Windows `uvx` run it is missing.

   **Fix**: `uvx --from mjlab --with scipy demo`

2. **`UnicodeEncodeError: 'charmap' codec can't encode character '\U0001f3ae'`**
   mjlab prints emoji characters. The default Windows console codepage (cp1252) cannot encode them.

   **Fix**: Set `PYTHONIOENCODING=utf-8` before running.

   **Working command**:
   ```powershell
   $env:PYTHONIOENCODING = "utf-8"
   uvx --from mjlab --with scipy demo
   ```

3. **G1 spin kick W&B artifacts are private**
   The `--wandb-run-path gcbc_researchers/...` path requires membership in the `gcbc_researchers` W&B team. The docs do not mention this restriction. The ONNX file shipped in the repo (`spinkick_safe.onnx`) is for the `RoboJuDo` integration, not the `play` command. Without a compatible motion `.npz` file, the spin kick replay cannot run.

   **Suggested doc addition**: Note that the W&B artifacts require team access, and point users to the RoboJuDo offline path or provide a public download link for the motion file.

---

## Demo 2 -- Isaac Lab ANYmal-D

### What the docs say

```bash
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
./isaaclab.sh --install rsl_rl
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Rough-Anymal-D-v0 --headless
```

### What breaks on Windows

1. **`isaaclab.sh` does not exist on Windows**
   The docs reference `./isaaclab.sh` everywhere. The Windows equivalent is `isaaclab.bat`, which ships in the same repo.

   **Fix**: Replace `./isaaclab.sh` with `isaaclab.bat` in all commands.

2. **`isaaclab.bat --install rsl_rl` fails: flatdict build-isolation bug**
   `isaaclab` depends on `flatdict==4.0.1`. flatdict's `setup.py` imports `pkg_resources`. pip's build isolation creates a temp virtualenv with a modern setuptools that no longer bundles `pkg_resources`, causing:
   ```
   ModuleNotFoundError: No module named 'pkg_resources'
   ```
   This affects pip >= 24 on Windows (and likely Linux too, but less commonly hit because distro setuptools still ships `pkg_resources`).

   **Fix**: Pre-install flatdict before running the Isaac Lab install:
   ```powershell
   conda activate rl101-isaac
   pip install flatdict==4.0.1 --no-build-isolation
   ```
   Then manually install each source extension (replicating what `isaaclab.bat --install` does):
   ```powershell
   $extensions = Get-ChildItem "external\IsaacLab\source" -Directory
   foreach ($ext in $extensions) {
       pip install --editable $ext.FullName
   }
   pip install -e "external\IsaacLab\source\isaaclab_rl[rsl_rl]"
   ```

3. **Binary Isaac Sim install requires environment variables**
   When using a pre-installed Isaac Sim (not pip-installed), the `isaacsim` Python module is not on `sys.path`. The `isaaclab.bat --conda` command sets these automatically, but if you created the conda env manually, you must set them yourself:
   ```powershell
   $env:PYTHONPATH = "C:\isaac-sim\site"
   $env:ISAAC_PATH = "C:\isaac-sim"
   $env:CARB_APP_PATH = "C:\isaac-sim\kit"
   $env:EXP_PATH = "C:\isaac-sim\apps"
   $env:RESOURCE_NAME = "IsaacSim"
   ```

4. **Symlink to Isaac Sim requires admin privileges**
   The docs imply creating a symlink `_isaac_sim -> /path/to/isaac-sim`. On Windows, `New-Item -ItemType SymbolicLink` requires admin. A directory junction works without admin:
   ```cmd
   cmd /c mklink /J _isaac_sim C:\isaac-sim
   ```

5. **Training performance**
   Training ran at ~47k steps/s, ~2.1s per iteration. Reward reached ~7.0 at iteration ~130 (~5 min). This is consistent with the documented ~5 min to first forward velocity, ~10-15 min for competent walking. ETA for full 1500 iterations is ~55 min.

### Working command (Windows, binary Isaac Sim)

A wrapper batch script is at `scripts/train_anymal_win.bat`. Or run directly:
```powershell
conda activate rl101-isaac
$env:PYTHONPATH = "C:\isaac-sim\site"
$env:ISAAC_PATH = "C:\isaac-sim"
$env:CARB_APP_PATH = "C:\isaac-sim\kit"
$env:EXP_PATH = "C:\isaac-sim\apps"
$env:RESOURCE_NAME = "IsaacSim"
cd external\IsaacLab
python scripts\reinforcement_learning\rsl_rl\train.py `
    --task Isaac-Velocity-Rough-Anymal-D-v0 --headless
```

---

## Demo 3 -- Holosoma G1 FastSAC

### What the docs say

```bash
bash scripts/setup_mujoco_via_uv.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-fast-sac simulator:mujoco_warp --training.seed 1
```

### What breaks on Windows

1. **Setup script is Linux-only**
   `setup_mujoco_via_uv.sh` uses `source bin/activate`, Ubuntu detection, and bash-specific constructs. It does not work on Windows.

   **Fix**: Manual setup with uv and Python 3.12:
   ```powershell
   cd external\holosoma
   uv venv --python 3.12 .venv\hsmujoco
   .venv\hsmujoco\Scripts\activate
   uv pip install mujoco mujoco-python-viewer
   uv pip install -e src/holosoma
   uv pip install "mujoco-warp==0.0.2"
   uv pip install "numpy>=1.23.5,<2"
   uv pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
   ```

2. **Python 3.13 incompatible**
   `open3d` (a holosoma dependency) does not have Python 3.13 wheels. Use Python 3.12.

3. **warp-lang version conflict**
   holosoma pins `warp-lang==1.10.0`. `mujoco-warp >= 3.5.0` requires `warp-lang >= 1.13.0`. These are incompatible -- warp 1.13 removed `wp.types.array()` used internally, and mujoco-warp 3.5+ uses `wp.array2d[int]` syntax not available in warp 1.10.

   **Fix**: Pin `mujoco-warp==0.0.2` which works with `warp-lang==1.10.0`.

4. **PyTorch defaults to CPU-only**
   `pip install torch` from PyPI installs CPU-only torch. Must explicitly use the CUDA index.

   **Fix**: `pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128`

5. **Doc error: wrong simulator subcommand**
   Docs say `simulator:mujoco_warp`. The actual Hydra config name is `simulator:mjwarp`.

   **Fix**: Use `simulator:mjwarp` in the training command.

6. **Triton unavailable on Windows**
   `torch.compile` requires Triton, which is Linux-only. Holosoma's training crashes with `TritonMissing`.

   **Fix**: `set TORCHDYNAMO_DISABLE=1` or use the wrapper script.

7. **bfloat16 validation bug**
   `torch.distributions.Normal` raises `ValueError: Expected parameter loc... to satisfy Real()` with bfloat16 tensors on some GPU/driver combos.

   **Fix**: `torch.distributions.Distribution.set_default_validate_args(False)` or use the wrapper script.

8. **Training speed: ~4x slower than documented**
   Docs say ~15 min for a walking gait. On Windows with `mujoco-warp==0.0.2` + `warp-lang==1.10.0`, training runs at ~3.5 it/s with ETA of ~4 hours. The `mujoco-warp==0.0.2` version is older and likely less optimized. This is the biggest practical gap.

   Episode length did improve from 19 to 75 over ~4000 iterations, and some reward terms showed NaN values. The root cause is the `penalty_action_rate` reward term producing NaN due to a Warp→PyTorch zero-copy tensor conversion issue in `mujoco-warp==0.0.2`. This NaN poisons the total reward, causing `actor_loss=nan`, `qf_loss=nan`, and `actor_grad_norm=0.0` — meaning the policy network never actually learns. The physics simulation itself works correctly (other reward terms compute fine).

   **Workarounds** (untested):
   - Disable the action_rate penalty: `--reward.penalties.action_rate.weight=0`
   - Run in WSL/Linux where the Warp tensor conversion may work correctly
   - Patch `penalty_action_rate()` in `managers/reward/terms/locomotion.py` to use `torch.nan_to_num()`

9. **Emoji crash in `print_mujoco_model_tree()`**
   Holosoma prints emoji characters (📊, 🏗️, etc.) during model loading. The default Windows console codepage (cp1252) cannot encode them, crashing both training and eval. The `PYTHONIOENCODING=utf-8` env var alone is insufficient because W&B's `console_capture.py` intercepts stdout before the encoding takes effect.

   **Fix**: Set encoding early AND wrap stdout/stderr:
   ```python
   os.environ["PYTHONIOENCODING"] = "utf-8"
   if sys.stdout.encoding != "utf-8":
       import io
       sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
       sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
   ```

10. **Eval viewer: mass randomization unsupported on CPU backend**
    Running `eval_agent.py` with `simulator:mujoco` (CPU) crashes with `RandomizerNotSupportedError: Mass randomization not supported for simulator type 'MuJoCo'`.

    **Fix**: Add `--randomization.ignore_unsupported=True` to the eval command.

11. **Eval viewer: mjwarp backend crashes with shape mismatch**
    Running `eval_agent.py` with `simulator:mjwarp` (GPU) crashes with `ValueError: could not broadcast input array from shape (0,35) into shape (0,)` in `mujoco_warp/_src/io.py`. This is a bug in mujoco-warp 0.0.2's `get_render_data()`.

    **Fix**: Use `simulator:mujoco` (CPU) for eval/visualization instead. Physics runs on CPU but policy inference still uses GPU.

### Working commands (Windows)

**Training** — wrapper script at `scripts/holosoma_train_win.py`:
```powershell
$env:TORCHDYNAMO_DISABLE = "1"
$env:PYTHONIOENCODING = "utf-8"
E:\rl101-crash-course\external\holosoma\.venv\hsmujoco\Scripts\python.exe `
    scripts\holosoma_train_win.py `
    exp:g1-29dof-fast-sac simulator:mjwarp --training.seed 1
```

**Eval/visualization** — wrapper script at `scripts/holosoma_eval_win.py`:
```powershell
$env:TORCHDYNAMO_DISABLE = "1"
$env:PYTHONIOENCODING = "utf-8"
E:\rl101-crash-course\external\holosoma\.venv\hsmujoco\Scripts\python.exe `
    scripts\holosoma_eval_win.py `
    --checkpoint logs\hv-g1-manager\<run>\model_XXXX.pt `
    simulator:mujoco --randomization.ignore_unsupported=True
```
Note: eval must use `simulator:mujoco` (CPU) because `simulator:mjwarp` crashes in `get_render_data()`.

---

## Proposed doc changes to `docs/week8-demos.md`

### 1. Add a "Running on Windows" subsection after the existing WSL2 note (line 45-49)

Cover:
- All `./isaaclab.sh` references should note `isaaclab.bat` for native Windows
- `uvx` commands need `--with scipy` and `PYTHONIOENCODING=utf-8`
- Holosoma setup script is Linux-only; provide manual Windows steps
- `simulator:mujoco_warp` should be corrected to `simulator:mjwarp` (this is a bug, not platform-specific)

### 2. Update Demo 1 Setup section

Add after the `uvx --from mjlab --refresh demo` command:
```
# Windows: add --with scipy and set encoding
$env:PYTHONIOENCODING = "utf-8"
uvx --from mjlab --with scipy demo
```

Add note about W&B artifact access requirements.

### 3. Update Demo 2 Setup section

Add Windows-specific instructions:
- Use `isaaclab.bat` instead of `isaaclab.sh`
- Pre-install flatdict to work around build-isolation bug
- Set Isaac Sim environment variables if using binary install
- Use `cmd /c mklink /J` instead of symlink

### 4. Update Demo 3 Setup section

- Correct `simulator:mujoco_warp` to `simulator:mjwarp`
- Add Windows manual setup instructions (Python 3.12, version pins)
- Note that `TORCHDYNAMO_DISABLE=1` is required on Windows
- Add bfloat16 workaround
- Note that training will be significantly slower with the mujoco-warp 0.0.2 fallback

### 5. Update Troubleshooting section

Add Windows-specific entries:
- flatdict/pkg_resources build failure and the `--no-build-isolation` workaround
- cp1252 encoding crash with emoji-printing Python packages
- Triton/dynamo unavailability
- Directory junction vs symlink for Isaac Sim path

---

## Helper scripts created

| Script | Purpose |
|--------|---------|
| `scripts/holosoma_train_win.py` | Wraps holosoma training with Triton disable + bfloat16 validation fix + emoji encoding fix |
| `scripts/holosoma_eval_win.py` | Wraps holosoma eval/visualization with same Windows workarounds |
| `scripts/train_anymal_win.bat` | Sets Isaac Sim env vars and runs ANYmal-D training |
| `scripts/install_isaaclab.bat` | Attempted batch install (superseded by manual approach) |

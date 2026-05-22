.PHONY: setup random train train-lunar train-ppo train-breakout train-pong train-spaceinvaders eval eval-lunar tensorboard demo help \
        demos-help setup-mjlab setup-spinkick demo-mjlab-sanity demo-spinkick \
        setup-isaaclab demo-anymal demo-anymal-play \
        setup-holosoma demo-humanoid-15min demo-humanoid-play

# Run commands inside the `rl101` conda env automatically so users don't
# need to `conda activate rl101` first. --no-capture-output preserves
# live stdout/stderr for training progress and pygame windows.
CONDA_ENV ?= rl101
CONDA_RUN := conda run -n $(CONDA_ENV) --no-capture-output

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: ## Run setup.sh (clone CleanRL, install deps)
	bash setup.sh

random: ## Run random agent on CartPole (baseline)
	$(CONDA_RUN) python scripts/random_agent.py

train: ## Train DQN on CartPole-v1 (500K steps)
	$(CONDA_RUN) python scripts/train_cartpole.py

train-lunar: ## Train DQN on LunarLander-v3 (1M steps, bonus)
	$(CONDA_RUN) python scripts/train_lunarlander.py

train-ppo: ## Train PPO on CartPole-v1 (Week 3 — actor-critic)
	$(CONDA_RUN) python scripts/train_cartpole_ppo.py

train-breakout: ## Train PPO on Breakout (Atari, 10M steps)
	$(CONDA_RUN) python scripts/train_atari_ppo.py --game breakout

train-pong: ## Train PPO on Pong (Atari, 5M steps)
	$(CONDA_RUN) python scripts/train_atari_ppo.py --game pong

train-spaceinvaders: ## Train PPO on Space Invaders (Atari, 10M steps)
	$(CONDA_RUN) python scripts/train_atari_ppo.py --game spaceinvaders

eval: ## Evaluate latest CartPole model
	@MODEL=$$(find runs -name "dqn.cleanrl_model" -path "*CartPole*" 2>/dev/null | sort | tail -1); \
	if [ -z "$$MODEL" ]; then \
		echo "No trained CartPole model found. Run 'make train' first."; \
		exit 1; \
	fi; \
	echo "Loading model: $$MODEL"; \
	$(CONDA_RUN) python scripts/evaluate.py --model-path "$$MODEL" --env-id CartPole-v1

eval-lunar: ## Evaluate latest LunarLander model
	@MODEL=$$(find runs -name "dqn.cleanrl_model" -path "*LunarLander*" 2>/dev/null | sort | tail -1); \
	if [ -z "$$MODEL" ]; then \
		echo "No trained LunarLander model found. Run 'make train-lunar' first."; \
		exit 1; \
	fi; \
	echo "Loading model: $$MODEL"; \
	$(CONDA_RUN) python scripts/evaluate.py --model-path "$$MODEL" --env-id LunarLander-v3

tensorboard: ## Open TensorBoard on runs/
	$(CONDA_RUN) tensorboard --logdir runs/

demo: ## Full demo flow: random → train → eval
	@echo "=== Step 1: Random Agent (baseline) ==="
	$(CONDA_RUN) python scripts/random_agent.py
	@echo ""
	@echo "=== Step 2: Training DQN ==="
	$(CONDA_RUN) python scripts/train_cartpole.py
	@echo ""
	@echo "=== Step 3: Evaluating Trained Agent ==="
	$(MAKE) eval

# ---------------------------------------------------------------------------
# Week 8 visual demos: mjlab + spinkick, Isaac Lab ANYmal-D, Holosoma G1
# Full run guide:  docs/week8-demos.md
# External repos are cloned into ./external/ (gitignored).
# ---------------------------------------------------------------------------

EXTERNAL_DIR ?= external

demos-help: ## Print the Week 8 visual demo overview
	@echo "Week 8 visual demos — see docs/week8-demos.md for the full guide."
	@echo ""
	@echo "  Demo 1 (MuJoCo, pretrained replay)"
	@echo "    make setup-mjlab          # install uv + run mjlab demo sanity check"
	@echo "    make setup-spinkick       # clone g1_spinkick_example into external/"
	@echo "    make demo-spinkick        # play the pretrained Unitree G1 spin kick"
	@echo ""
	@echo "  Demo 2 (Isaac Lab, live PPO ~10-15 min)"
	@echo "    make setup-isaaclab       # guided install (needs Python 3.11 env)"
	@echo "    make demo-anymal          # train ANYmal-D rough terrain"
	@echo "    make demo-anymal-play     # visualize trained policy"
	@echo ""
	@echo "  Demo 3 (Holosoma, live SAC ~15 min)"
	@echo "    make setup-holosoma       # clone + setup_mujoco_via_uv.sh"
	@echo "    make demo-humanoid-15min  # train Unitree G1 walking from scratch"
	@echo "    make demo-humanoid-play   # visualize the trained G1"

# ---- Demo 1: mjlab + g1_spinkick_example ---------------------------------

setup-mjlab: ## Install uv + run mjlab demo as a sanity check
	@command -v uv >/dev/null 2>&1 || { \
		echo ">>> Installing uv..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	}
	@echo ">>> Running mjlab demo (downloads to uv cache, opens MuJoCo viewer)..."
	uvx --from mjlab --refresh demo

setup-spinkick: ## Clone g1_spinkick_example into external/ and uv sync
	@mkdir -p $(EXTERNAL_DIR)
	@if [ ! -d "$(EXTERNAL_DIR)/g1_spinkick_example" ]; then \
		echo ">>> Cloning g1_spinkick_example..."; \
		git clone https://github.com/mujocolab/g1_spinkick_example.git \
			$(EXTERNAL_DIR)/g1_spinkick_example; \
	else \
		echo ">>> g1_spinkick_example already cloned at $(EXTERNAL_DIR)/g1_spinkick_example"; \
	fi
	cd $(EXTERNAL_DIR)/g1_spinkick_example && uv sync

demo-mjlab-sanity: ## Re-run the mjlab one-line demo (uvx --from mjlab demo)
	uvx --from mjlab demo

demo-spinkick: ## Play the pretrained Unitree G1 double-spin-kick policy
	@echo "Requires a W&B run path or local ONNX. See docs/week8-demos.md."
	@echo "Falling back to mjlab demo command — substitute --wandb-run-path as needed."
	cd $(EXTERNAL_DIR)/g1_spinkick_example && uv run play \
		Mjlab-Spinkick-Unitree-G1 --num-envs 1

# ---- Demo 2: Isaac Lab ANYmal-D ------------------------------------------

ISAACLAB_DIR ?= $(EXTERNAL_DIR)/IsaacLab
ISAACLAB_ENV ?= rl101-isaac

setup-isaaclab: ## Guided Isaac Lab install — prints the conda-env command sequence
	@echo "Isaac Lab needs a separate Python 3.11 conda env. Run these manually:"
	@echo ""
	@echo "  conda create -y -n $(ISAACLAB_ENV) python=3.11"
	@echo "  conda activate $(ISAACLAB_ENV)"
	@echo "  pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com"
	@echo "  pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128"
	@echo "  mkdir -p $(EXTERNAL_DIR) && cd $(EXTERNAL_DIR)"
	@echo "  git clone https://github.com/isaac-sim/IsaacLab.git"
	@echo "  cd IsaacLab && ./isaaclab.sh --install rsl_rl"
	@echo ""
	@echo "Reserve ~25 GB disk. Full guide: docs/week8-demos.md"

demo-anymal: ## Train ANYmal-D rough terrain (Isaac Lab, ~10-15 min on 5090)
	@if [ ! -x "$(ISAACLAB_DIR)/isaaclab.sh" ]; then \
		echo "Isaac Lab not found at $(ISAACLAB_DIR). Run 'make setup-isaaclab' first."; \
		exit 1; \
	fi
	cd $(ISAACLAB_DIR) && ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
		--task Isaac-Velocity-Rough-Anymal-D-v0 --headless

demo-anymal-play: ## Visualize the trained ANYmal-D policy across new terrain
	@if [ ! -x "$(ISAACLAB_DIR)/isaaclab.sh" ]; then \
		echo "Isaac Lab not found at $(ISAACLAB_DIR). Run 'make setup-isaaclab' first."; \
		exit 1; \
	fi
	cd $(ISAACLAB_DIR) && ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
		--task Isaac-Velocity-Rough-Anymal-D-Play-v0 --num_envs 32

# ---- Demo 3: Holosoma humanoid in 15 minutes -----------------------------

HOLOSOMA_DIR ?= $(EXTERNAL_DIR)/holosoma

setup-holosoma: ## Clone Holosoma and run setup_mujoco_via_uv.sh (MJWarp backend)
	@mkdir -p $(EXTERNAL_DIR)
	@if [ ! -d "$(HOLOSOMA_DIR)" ]; then \
		echo ">>> Cloning holosoma..."; \
		git clone https://github.com/amazon-far/holosoma.git $(HOLOSOMA_DIR); \
	else \
		echo ">>> holosoma already cloned at $(HOLOSOMA_DIR)"; \
	fi
	cd $(HOLOSOMA_DIR) && bash scripts/setup_mujoco_via_uv.sh

demo-humanoid-15min: ## Train Unitree G1 with FastSAC on MJWarp (~15 min on 5090)
	@if [ ! -d "$(HOLOSOMA_DIR)" ]; then \
		echo "Holosoma not found at $(HOLOSOMA_DIR). Run 'make setup-holosoma' first."; \
		exit 1; \
	fi
	cd $(HOLOSOMA_DIR) && python src/holosoma/holosoma/train_agent.py \
		exp:g1-29dof-fast-sac simulator:mujoco_warp --training.seed 1

demo-humanoid-play: ## Visualize the latest Holosoma G1 checkpoint
	@if [ ! -d "$(HOLOSOMA_DIR)" ]; then \
		echo "Holosoma not found at $(HOLOSOMA_DIR). Run 'make setup-holosoma' first."; \
		exit 1; \
	fi
	cd $(HOLOSOMA_DIR) && python src/holosoma/holosoma/play_agent.py \
		exp:g1-29dof-fast-sac simulator:mujoco_warp \
		--checkpoint runs/latest/checkpoint.pt

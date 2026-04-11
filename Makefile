.PHONY: setup random train train-lunar train-ppo eval eval-lunar tensorboard demo help

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

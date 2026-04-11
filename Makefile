.PHONY: setup random train train-lunar eval tensorboard demo help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: ## Run setup.sh (clone CleanRL, install deps)
	bash setup.sh

random: ## Run random agent on CartPole (baseline)
	python scripts/random_agent.py

train: ## Train DQN on CartPole-v1 (500K steps)
	python scripts/train_cartpole.py

train-lunar: ## Train DQN on LunarLander-v3 (1M steps, bonus)
	python scripts/train_lunarlander.py

eval: ## Evaluate latest CartPole model
	@MODEL=$$(find runs -name "dqn.cleanrl_model" -path "*CartPole*" 2>/dev/null | sort | tail -1); \
	if [ -z "$$MODEL" ]; then \
		echo "No trained CartPole model found. Run 'make train' first."; \
		exit 1; \
	fi; \
	echo "Loading model: $$MODEL"; \
	python scripts/evaluate.py --model-path "$$MODEL" --env-id CartPole-v1

eval-lunar: ## Evaluate latest LunarLander model
	@MODEL=$$(find runs -name "dqn.cleanrl_model" -path "*LunarLander*" 2>/dev/null | sort | tail -1); \
	if [ -z "$$MODEL" ]; then \
		echo "No trained LunarLander model found. Run 'make train-lunar' first."; \
		exit 1; \
	fi; \
	echo "Loading model: $$MODEL"; \
	python scripts/evaluate.py --model-path "$$MODEL" --env-id LunarLander-v3

tensorboard: ## Open TensorBoard on runs/
	tensorboard --logdir runs/

demo: ## Full demo flow: random → train → eval
	@echo "=== Step 1: Random Agent (baseline) ==="
	python scripts/random_agent.py
	@echo ""
	@echo "=== Step 2: Training DQN ==="
	python scripts/train_cartpole.py
	@echo ""
	@echo "=== Step 3: Evaluating Trained Agent ==="
	$(MAKE) eval

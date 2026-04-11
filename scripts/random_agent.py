"""Random agent on CartPole-v1 — the "before" baseline.

Shows how a random policy fails in ~10-20 steps.
Run: python scripts/random_agent.py
"""

import gymnasium as gym


def main():
    env = gym.make("CartPole-v1", render_mode="human")
    obs, info = env.reset()

    total_reward = 0
    step = 0

    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        if terminated or truncated:
            break

    print(f"Random agent: {step} steps, total reward: {total_reward}")
    print("A trained DQN agent can reach 500 steps (the maximum).")
    env.close()


if __name__ == "__main__":
    main()

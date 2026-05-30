#!/usr/bin/env python3
"""Controlled tabular RL experiments for the Q-learning survey (§VI).

Isolates algorithmic mechanism from deep-learning confounds. Reports
model-free learners (Q-learning, SARSA, Expected SARSA, n-step Q) over
many seeds with bootstrap CIs, and dynamic-programming PLANNING ORACLES
(value/policy iteration) separately as upper bounds — the oracles have
full model access (P, R) and are NOT direct competitors to the
model-free agents.

Pure NumPy / CPU: these are table updates, not neural nets — a GPU adds
nothing. Deterministic given --seeds for reproducibility.

Usage:
    python scripts/tabular_experiments.py            # full run (100 seeds)
    python scripts/tabular_experiments.py --seeds 10 # quick smoke test
"""
from __future__ import annotations
import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import gymnasium as gym


# ----------------------------- environments -----------------------------
def make_env(name):
    """Return (env, label). Handles CliffWalking v0/v1 naming drift."""
    if name == "CliffWalking":
        for v in ("CliffWalking-v1", "CliffWalking-v0"):
            try:
                return gym.make(v), v
            except Exception:
                continue
        raise RuntimeError("CliffWalking not registered")
    return gym.make(name), name


ENVS = {
    "FrozenLake-v1": dict(kwargs=dict(map_name="4x4", is_slippery=True),
                          episodes=15000, alpha=0.10, gamma=0.99),
    "Taxi-v3":       dict(kwargs=dict(),
                          episodes=12000, alpha=0.10, gamma=0.99),
    "CliffWalking":  dict(kwargs=dict(),
                          episodes=2500, alpha=0.50, gamma=0.99),
}
N_STEP = 3
EVAL_EPISODES = 300
EPS_START, EPS_END = 1.0, 0.05


from gymnasium.wrappers import TimeLimit
DEFAULT_MAX_STEPS = 500  # cap for envs with no built-in limit (CliffWalking),
                         # so a looping greedy policy truncates instead of hanging


def _wrap(env):
    if getattr(env.spec, "max_episode_steps", None) is None:
        env = TimeLimit(env, max_episode_steps=DEFAULT_MAX_STEPS)
    return env


def _make(name, kwargs):
    if name == "CliffWalking":
        for v in ("CliffWalking-v1", "CliffWalking-v0"):
            try:
                return _wrap(gym.make(v, **kwargs)), v
            except Exception:
                continue
        raise RuntimeError("CliffWalking not registered")
    return _wrap(gym.make(name, **kwargs)), name


# ----------------------------- model-free -------------------------------
def epsilon_greedy(Q, s, eps, rng, nA):
    if rng.random() < eps:
        return rng.integers(nA)
    # random tie-break among argmax
    m = Q[s].max()
    return rng.choice(np.flatnonzero(Q[s] == m))


def eps_schedule(ep, total):
    frac = min(1.0, ep / (0.8 * total))
    return EPS_START + frac * (EPS_END - EPS_START)


def train_model_free(algo, env_name, cfg, seed):
    env, _ = _make(env_name, cfg["kwargs"])
    nS, nA = env.observation_space.n, env.action_space.n
    rng = np.random.default_rng(seed)
    Q = np.zeros((nS, nA))
    alpha, gamma, episodes = cfg["alpha"], cfg["gamma"], cfg["episodes"]

    for ep in range(episodes):
        eps = eps_schedule(ep, episodes)
        s, _ = env.reset(seed=int(rng.integers(1 << 31)))
        a = epsilon_greedy(Q, s, eps, rng, nA)
        if algo == "nstep_q":
            buf = []  # (s, a, r)
        done = False
        while not done:
            ns, r, term, trunc, _ = env.step(a)
            done = term or trunc
            if algo == "q_learning":
                target = r + (0.0 if term else gamma * Q[ns].max())
                Q[s, a] += alpha * (target - Q[s, a])
                na = epsilon_greedy(Q, ns, eps, rng, nA)
            elif algo == "sarsa":
                na = epsilon_greedy(Q, ns, eps, rng, nA)
                target = r + (0.0 if term else gamma * Q[ns, na])
                Q[s, a] += alpha * (target - Q[s, a])
            elif algo == "expected_sarsa":
                na = epsilon_greedy(Q, ns, eps, rng, nA)
                # E_pi[Q(ns,.)] under eps-greedy
                best = Q[ns].max()
                greedy = (Q[ns] == best)
                pi = np.full(nA, eps / nA)
                pi[greedy] += (1.0 - eps) / greedy.sum()
                exp_q = 0.0 if term else gamma * float(pi @ Q[ns])
                Q[s, a] += alpha * (r + exp_q - Q[s, a])
            elif algo == "nstep_q":
                buf.append((s, a, r))
                na = epsilon_greedy(Q, ns, eps, rng, nA)
                if len(buf) >= N_STEP:
                    G = sum((gamma ** i) * buf[i][2] for i in range(N_STEP))
                    G += 0.0 if (done and not trunc) else (gamma ** N_STEP) * Q[ns].max()
                    s0, a0, _ = buf.pop(0)
                    Q[s0, a0] += alpha * (G - Q[s0, a0])
            s, a = ns, na
        if algo == "nstep_q":  # flush remaining
            while buf:
                k = len(buf)
                G = sum((gamma ** i) * buf[i][2] for i in range(k))
                s0, a0, _ = buf.pop(0)
                Q[s0, a0] += alpha * (G - Q[s0, a0])
    env.close()
    return Q


def evaluate(Q, env_name, cfg, n_episodes, seed):
    env, _ = _make(env_name, cfg["kwargs"])
    rng = np.random.default_rng(seed + 99991)
    returns = []
    for _ in range(n_episodes):
        s, _ = env.reset(seed=int(rng.integers(1 << 31)))
        done, total = False, 0.0
        while not done:
            a = int(np.argmax(Q[s]))
            s, r, term, trunc, _ = env.step(a)
            total += r
            done = term or trunc
        returns.append(total)
    env.close()
    return float(np.mean(returns))


# --------------------------- planning oracles ---------------------------
def extract_PR(env_name, cfg):
    env, _ = _make(env_name, cfg["kwargs"])
    nS, nA = env.observation_space.n, env.action_space.n
    P = np.zeros((nS, nA, nS))
    R = np.zeros((nS, nA))
    term = np.zeros(nS, dtype=bool)
    model = env.unwrapped.P
    for s in range(nS):
        for a in range(nA):
            for prob, ns, r, done in model[s][a]:
                P[s, a, ns] += prob
                R[s, a] += prob * r
                if done:
                    term[ns] = True
    env.close()
    return P, R, term, nS, nA


def value_iteration(env_name, cfg, theta=1e-10):
    P, R, term, nS, nA = extract_PR(env_name, cfg)
    g = cfg["gamma"]
    V = np.zeros(nS)
    while True:
        Vb = V.copy()
        Vb[term] = 0.0          # do not bootstrap through terminal states
        Q = R + g * (P @ Vb)
        nV = Q.max(axis=1)
        nV[term] = 0.0          # terminal states have zero value
        if np.max(np.abs(nV - V)) < theta:
            break
        V = nV
    Vb = V.copy(); Vb[term] = 0.0
    pi = (R + g * (P @ Vb)).argmax(axis=1)
    return pi


def oracle_eval(env_name, cfg, n_episodes, seed):
    pi = value_iteration(env_name, cfg)
    Q = np.zeros((len(pi), 1))  # dummy not used; eval via policy
    env, _ = _make(env_name, cfg["kwargs"])
    rng = np.random.default_rng(seed)
    returns = []
    for _ in range(n_episodes):
        s, _ = env.reset(seed=int(rng.integers(1 << 31)))
        done, total = False, 0.0
        while not done:
            s, r, term, trunc, _ = env.step(int(pi[s]))
            total += r
            done = term or trunc
        returns.append(total)
    env.close()
    return float(np.mean(returns)), float(np.std(returns))


# ------------------------------ aggregation -----------------------------
def _run_seed(task):
    """Top-level worker (picklable): train one seed and return its eval score."""
    algo, env_name, cfg, seed = task
    Q = train_model_free(algo, env_name, cfg, seed)
    return evaluate(Q, env_name, cfg, EVAL_EPISODES, seed)


def bootstrap_ci(xs, reps=10000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    xs = np.asarray(xs)
    idx = rng.integers(0, len(xs), size=(reps, len(xs)))
    means = xs[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(__file__), "..",
        "notes/q-survey-revision/draft-tai/data/tabular_results.json"))
    args = ap.parse_args()

    algos = ["q_learning", "sarsa", "expected_sarsa", "nstep_q"]
    results = {"config": {"seeds": args.seeds, "n_step": N_STEP,
                          "eval_episodes": EVAL_EPISODES,
                          "workers": args.workers, "envs": {}}}
    t0 = time.time()
    print(f"[{time.time()-t0:6.1f}s] parallel over {args.workers} workers "
          f"× {args.seeds} seeds", flush=True)
    with Pool(args.workers) as pool:
        for env_name, cfg in ENVS.items():
            results[env_name] = {"model_free": {}, "oracle": {}}
            results["config"]["envs"][env_name] = {
                k: v for k, v in cfg.items() if k != "kwargs"}
            print(f"[{time.time()-t0:6.1f}s] === {env_name} ===", flush=True)
            for algo in algos:
                tasks = [(algo, env_name, cfg, seed) for seed in range(args.seeds)]
                per_seed = pool.map(_run_seed, tasks)
                mean = float(np.mean(per_seed))
                lo, hi = bootstrap_ci(per_seed)
                results[env_name]["model_free"][algo] = dict(
                    mean=mean, ci95=[lo, hi], std=float(np.std(per_seed)))
                print(f"[{time.time()-t0:6.1f}s]   {algo:16s} "
                      f"{mean:8.3f}  CI[{lo:.3f},{hi:.3f}]", flush=True)
            om, osd = oracle_eval(env_name, cfg, 5000, seed=0)
            results[env_name]["oracle"]["value_iteration"] = dict(mean=om, std=osd)
            print(f"[{time.time()-t0:6.1f}s]   ORACLE(VI/PI/MPI)  {om:8.3f}  "
                  f"(planning upper bound, full model access)", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[{time.time()-t0:6.1f}s] DONE -> {args.out}", flush=True)


if __name__ == "__main__":
    main()

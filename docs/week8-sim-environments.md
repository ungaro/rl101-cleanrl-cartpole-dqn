---
marp: true
theme: default
paginate: true
header: "RL 101 — Week 8 — Robotics Simulation RL: Isaac Sim & MuJoCo"
footer: "rl101-crash-course"
math: katex
---

<!--
To view as slides, install marp-cli and run:
  npx @marp-team/marp-cli docs/week8-sim-environments.md --preview
  npx @marp-team/marp-cli docs/week8-sim-environments.md --pdf
On GitHub this file also renders as a normal document with math and code.
-->

# Robotics Simulation RL — Isaac Sim & MuJoCo Under the Hood

## Companion to the Extra Special Week — Robotics Sim & Sim-to-Real

**Week 8 — RL 101 Study Group**

A deep dive into the two simulators that dominate modern robotics RL —
**MuJoCo** (the physics-first lineage, now hosted by Google DeepMind) and
**NVIDIA Isaac Sim / Isaac Lab** (the GPU-parallel, Omniverse-based stack
that replaced Isaac Gym). We walk through what each one actually does,
why the field moved off Isaac Gym onto Isaac Lab, how sim-to-real transfer
works in practice (domain randomization, RMA, teacher–student), and which
2024–2026 humanoid and quadruped demos used which tool. The earlier weeks
gave us the algorithms (DQN, PPO, GRPO); this week gives us the
**environments those algorithms train in** when the target is a physical
robot.

> **Runnable companion:** [`docs/week8-demos.md`](week8-demos.md) ships
> the same ideas as a session-friendly demo guide — Unitree G1 spin kick
> (mjlab + pretrained ONNX), ANYmal-D rough terrain (Isaac Lab, live PPO
> ~10–15 min), and the "humanoid in 15 minutes" recipe (Holosoma, FastSAC
> on Unitree G1). Use this file for the math and lineage; use that one
> when you want to actually press play.

---

## How this doc relates to other weeks

- **Week 2 (DQN)** taught value-based RL on a 4-d toy state. Robotics sim
  is the same Bellman equation, but the state is a 50–200-d continuous
  vector from joint sensors and an IMU, and the action is a vector of
  desired joint positions or torques.
- **Week 3 (PPO)** introduced the actor-critic split. PPO is still the
  workhorse of robotics RL — almost every legged-locomotion sim-to-real
  paper from 2021 to 2026 uses PPO over thousands of parallel envs.
- **Week 4 (Forge / agent RL)** was about *language* agents — long
  horizons, sparse rewards, expensive rollouts. Robotics RL has the same
  long-horizon-sparse-reward shape, but the cheap-rollout assumption
  flips: in sim, rollouts are extremely cheap (millions of steps/sec on
  a GPU), so we can train PPO from scratch instead of fine-tuning a
  pretrained policy.
- **Week 5–6 (RLHF / RFT)** is about post-training a model whose prior
  is already good. Robotics RL is more like classical RL: the policy
  starts from random and gets to good by interacting with the simulator.
- **Week 7 (World Models / RWML)** asked *can the agent learn the
  environment?*. Week 8 is the complementary question: *if a
  high-fidelity hand-built simulator already exists, how do we use it
  efficiently and cross the gap back to reality?*

If you want one mental anchor: Week 3's PPO + Week 8's Isaac Lab + a
robot URDF is most of what powers the humanoid locomotion demos you see
on Twitter.

---

## Table of Contents

### Part I — Why simulate at all?

1. [The robot-data scarcity problem](#1-the-robot-data-scarcity-problem)
2. [What "the reality gap" actually means](#2-what-the-reality-gap-actually-means)
3. [The two complementary directions: better sim vs. robust policy](#3-the-two-complementary-directions-better-sim-vs-robust-policy)

### Part II — MuJoCo: the physics-first lineage

4. [What MuJoCo is and who maintains it](#4-what-mujoco-is-and-who-maintains-it)
5. [The physics: generalized coordinates and convex contacts](#5-the-physics-generalized-coordinates-and-convex-contacts)
6. [The MJCF model format and the Python API](#6-the-mjcf-model-format-and-the-python-api)
7. [MJX — MuJoCo on JAX / GPU](#7-mjx--mujoco-on-jax--gpu)
8. [The DeepMind ecosystem: Menagerie, Playground, dm_control](#8-the-deepmind-ecosystem-menagerie-playground-dm_control)

### Part III — NVIDIA Isaac: the GPU-parallel stack

9. [Decoding the Isaac brand: Sim vs. Lab vs. Gym vs. GR00T vs. Cosmos](#9-decoding-the-isaac-brand-sim-vs-lab-vs-gym-vs-gr00t-vs-cosmos)
10. [Why Isaac Gym was deprecated and Isaac Lab took over](#10-why-isaac-gym-was-deprecated-and-isaac-lab-took-over)
11. [PhysX 5 and the tensor API](#11-physx-5-and-the-tensor-api)
12. [Isaac Lab task structure for RL practitioners](#12-isaac-lab-task-structure-for-rl-practitioners)
13. [GR00T, Cosmos, and the humanoid foundation-model stack](#13-groot-cosmos-and-the-humanoid-foundation-model-stack)

### Part IV — The sim-to-real workflow

14. [Domain randomization: the OpenAI recipe](#14-domain-randomization-the-openai-recipe)
15. [Rapid Motor Adaptation (RMA) and teacher–student distillation](#15-rapid-motor-adaptation-rma-and-teacherstudent-distillation)
16. [System identification and real-to-sim](#16-system-identification-and-real-to-sim)
17. [Differentiable simulation: when gradients beat sampling](#17-differentiable-simulation-when-gradients-beat-sampling)
18. [The PPO-dominance story and when SAC or Dreamer wins instead](#18-the-ppo-dominance-story-and-when-sac-or-dreamer-wins-instead)

### Part V — Landmark wins, 2024–2026

19. [Quadruped locomotion: the canonical sim-to-real success](#19-quadruped-locomotion-the-canonical-sim-to-real-success)
20. [Humanoids in 2025–2026: Berkeley, Unitree, Booster, Figure, 1X](#20-humanoids-in-20252026-berkeley-unitree-booster-figure-1x)
21. [Manipulation: Aloha, ManiSkill, π0, and the imitation-vs-RL boundary](#21-manipulation-aloha-maniskill-π0-and-the-imitation-vs-rl-boundary)

### Part VI — Hands-on starting points

22. [Minimal MuJoCo: load, step, render](#22-minimal-mujoco-load-step-render)
23. [Minimal MJX: vectorized rollouts](#23-minimal-mjx-vectorized-rollouts)
24. [Minimal Isaac Lab: training a quadruped in one command](#24-minimal-isaac-lab-training-a-quadruped-in-one-command)
25. [Common gotchas: control frequency, termination, reward hacking](#25-common-gotchas-control-frequency-termination-reward-hacking)

### Part VII — Bridge back to the rest of the course

26. [From CartPole to a quadruped: what changes and what doesn't](#26-from-cartpole-to-a-quadruped-what-changes-and-what-doesnt)
27. [Where world models (Week 7) fit into sim-to-real](#27-where-world-models-week-7-fit-into-sim-to-real)
28. [The reward-source arc revisited: RLHF → RLVR → RWML → physical sim](#28-the-reward-source-arc-revisited-rlhf--rlvr--rwml--physical-sim)

### Part VIII — Q&A

29. [Q&A — Which simulator should I pick?](#29-qa--which-simulator-should-i-pick)
30. [Q&A — Sim-to-real failure modes](#30-qa--sim-to-real-failure-modes)
31. [Q&A — Hardware, scale, and reproducibility](#31-qa--hardware-scale-and-reproducibility)

### Part IX — Resources

32. [Papers, blogs, videos, code](#32-papers-blogs-videos-code)
33. [Key Takeaways](#33-key-takeaways)

---

## Part I — Why simulate at all?

### 1. The robot-data scarcity problem

For the entirety of Weeks 1–7 the data was either free or already
collected: CartPole runs in microseconds, Atari frames are deterministic,
ALFWorld text is just strings, and even ALFRED uses pre-rendered scenes.
Robotics breaks that assumption. One real robot trajectory is:

- **Slow** — wall-clock-bound, you cannot speed up reality.
- **Expensive** — humans supervise, hardware wears out, falls break
  things.
- **Serial** — one body, one episode, one trial at a time.
- **Hard to reset** — every "reset" is a human walking the robot back to
  start, or scripting an inverse policy.

A GPU-batched simulator collapses every one of those bottlenecks. Isaac
Lab on a single RTX 4090 reaches **~94k steps/sec on a 29-DoF Unitree G1
humanoid** with rough terrain and **~1.1M steps/sec on a Cartpole-style
toy** ([Isaac Lab performance benchmarks][isaac-bench]). MJX on a single
RTX 4090 trains a 30-DoF REEM-C humanoid PPO policy for **200M
environment steps in 56 minutes** ([Singh et al. 2024][brax-mjx-2024]).
Comparable real-world budgets would be on the order of **years**, with
non-trivial probability the robot is broken before training converges.

That ~100–1000× throughput advantage is the entire reason robotics RL
training happens in sim first and reality second. It is also why the rest
of this document spends 80% of its time on simulators and only 20% on the
"cross back to reality" step.

[isaac-bench]: https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/performance_benchmarks.html
[brax-mjx-2024]: https://arxiv.org/abs/2407.05148

### 2. What "the reality gap" actually means

If sim were exact, training in sim would be training on the real robot
and we would stop here. The simulators are not exact, and the gap shows
up along the following axes:

| Axis | What sim usually does | What reality does |
|---|---|---|
| **Inertia / mass** | Designer's URDF estimate | Real values, drift with wear / payload |
| **Joint friction** | Single constant per joint | Stick-slip, temperature-dependent, gear backlash |
| **Motor model** | Ideal torque source | PID loop with current/thermal limits, voltage sag |
| **Contact** | Soft convex (MuJoCo) or PhysX impulses | Rigid, occasional slip, ground compliance |
| **Sensor noise** | Zero, or hand-tuned Gaussian | IMU drift, encoder quantization, depth holes |
| **Latency** | 0 ms | 5–30 ms control loop + sensor processing |
| **Vision** | RTX-rendered, clean | Lens flare, motion blur, distractors, novel textures |

A policy that fits sim's *exact* dynamics will exploit any of those
mismatches and fail on the real robot. The classic failure pattern is a
quadruped policy that "hooks" its feet through the floor in sim to gain
free forward velocity — runs perfectly in MuJoCo, falls over on grass.

The term "reality gap" itself goes back to evolutionary-robotics work in
the 1990s; the modern formulation in deep RL is essentially the
**covariate shift** between the sim state distribution
```math
p_{\text{sim}}(s_t, a_t)
```
and the real-world state distribution
```math
p_{\text{real}}(s_t, a_t).
```
If your policy minimizes loss on the first distribution and is deployed
on the second, you are in classic out-of-distribution territory — and
the standard ML answer applies: **either make the training distribution
include the test distribution, or make the policy robust to the shift.**

### 3. The two complementary directions: better sim vs. robust policy

Every published sim-to-real result is some mixture of these two:

**Direction A — close the gap (make sim look like reality).**
- High-fidelity contact models, joint friction calibration, sensor noise
  models, latency injection.
- System identification: fit physics parameters to short real-world
  rollouts.
- Real-to-sim: train a residual model that corrects sim trajectories to
  match observed real ones.
- World models trained on real data (e.g. DayDreamer) bypass the
  hand-built simulator entirely.

**Direction B — make the policy robust to the gap (transfer across).**
- Domain randomization (DR): train across many randomized physics
  parameters so the policy treats real-world as just another sample.
- Rapid Motor Adaptation (RMA): explicit online identification of
  environment latents via proprioceptive history.
- Teacher–student: a privileged teacher in sim trains a deployable
  student via supervised distillation.

In practice every modern legged-locomotion paper does **both**: calibrate
the simulator with system ID **and** randomize the rest with DR **and**
distill a proprioception-only student from a privileged teacher. The
remainder of Part IV covers these tools individually.

---

## Part II — MuJoCo: the physics-first lineage

### 4. What MuJoCo is and who maintains it

**MuJoCo** — *Multi-Joint dynamics with Contact* — was written by
Emanuel Todorov at the University of Washington and his company Roboti
LLC. It built a reputation in robotics research over a decade as the
physics engine that ran fast enough for RL and modeled contacts well
enough for legged robots.

The history matters because the access model changed twice:

- **2012** — first MuJoCo paper at IROS ([Todorov, Erez, Tassa][mujoco-paper]).
  Commercial license sold by Roboti LLC; widely used in research via
  paid academic licenses.
- **October 2021** — **DeepMind acquires MuJoCo** and immediately makes
  it free.
- **May 23, 2022** — **MuJoCo open-sourced under Apache 2.0**, repo
  moves to [github.com/google-deepmind/mujoco][mujoco-repo]. This is the
  current home; bug fixes and new features happen here.

The takeaway: any tutorial older than May 2022 that talks about
`mujoco-py` (the old third-party Python wrapper) or paid licenses is
out of date. The current official Python package is `mujoco` on PyPI,
maintained by DeepMind.

[mujoco-paper]: https://homes.cs.washington.edu/~todorov/papers/TodorovIROS12.pdf
[mujoco-repo]: https://github.com/google-deepmind/mujoco

### 5. The physics: generalized coordinates and convex contacts

MuJoCo's modeling choices are what made it dominant for RL:

**Generalized (joint) coordinates.** State is `(qpos, qvel)` in joint
space, not Cartesian per body. A 7-DoF arm has a 7-d position vector
plus a 7-d velocity vector — not 7 bodies × 6 DoF each. This keeps the
equations of motion minimal-coordinate and removes the need for
constraint enforcement on the articulated structure itself. For RL the
practical consequence is that observation vectors stay small even for
complex robots.

The equations of motion are the classical Lagrangian form
```math
M(q)\,\dot v + c(q, v) = \tau + J^\top f,
```
where `M(q)` is the inertia matrix, `c(q,v)` lumps Coriolis / centrifugal
/ gravity terms, `τ` is actuator torque, `f` is contact force, and
`J^⊤` is the contact Jacobian. The integrator computes `\dot v` and
steps forward.

**Soft convex contacts.** Contacts (and other constraints) are formulated
as a **convex optimization problem** that MuJoCo solves with Newton or
conjugate gradient. "Soft" means a small amount of constraint violation
is tolerated and penalized rather than enforced exactly. This is the
single biggest difference from Bullet / ODE / PhysX-classic, which use
strict linear-complementarity-problem (LCP) contact solvers.

The practical consequences:

- Contacts are smooth in MuJoCo, which makes losses smooth in
  trajectory-optimization and gives MJX clean gradients.
- The same softness produces *physically wrong* behavior at high speed
  / hard contact — penetration, free energy if you exploit it. Real-robot
  transfer requires care.
- Bullet/PhysX are physically harder to fool but produce non-smooth
  dynamics that gradient-based methods hate.

This trade-off is why MuJoCo dominates *gradient-friendly* / RL training
and PhysX dominates *visually-accurate* simulation in Isaac Sim.

### 6. The MJCF model format and the Python API

MuJoCo robots are described in **MJCF**, an XML dialect:

```xml
<mujoco>
  <worldbody>
    <body name="pole" pos="0 0 1">
      <joint name="hinge" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.05" fromto="0 0 0 0 0 1"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="hinge" gear="100"/>
  </actuator>
</mujoco>
```

URDF (the ROS standard) can also be loaded and converted. MJCF adds a
`<default>` mechanism for shared geometry/joint properties and an
`<include>` directive that keeps large humanoid models manageable.

The Python API is intentionally low-level — closer to the C bindings
than to Gymnasium:

```python
import mujoco

model = mujoco.MjModel.from_xml_path("humanoid.xml")
data  = mujoco.MjData(model)

# Set some control input
data.ctrl[:] = 0.1

# Step physics one timestep
mujoco.mj_step(model, data)

# Inspect state
print(data.qpos)         # joint positions
print(data.qvel)         # joint velocities
print(data.sensordata)   # configured sensors
```

`MjModel` is compiled (immutable) and `MjData` is the per-step mutable
state. There is no built-in observation or reward — those are part of
*your* env definition. That intentional minimalism is why the same
engine powers `dm_control`, Gymnasium-Robotics, Robosuite, ManiSkill,
and Isaac Lab's MuJoCo-backed envs.

For visualization, the `mujoco.viewer` module has two modes:

- `viewer.launch(model, data)` — blocking GUI.
- `viewer.launch_passive(model, data)` — returns a handle so your script
  keeps stepping while the GUI renders; call `viewer.sync()` after each
  `mj_step`.

On macOS, passive mode requires the `mjpython` launcher binary
(`mjpython my_script.py`) because Cocoa's main thread must own the GUI
window — a recurring "why doesn't this work on my Mac?" gotcha.

### 7. MJX — MuJoCo on JAX / GPU

C MuJoCo is fast per-step on CPU but doesn't batch on accelerators.
**MJX** is a reimplementation of the MuJoCo step function as pure
JAX/XLA primitives, shipped inside the same repo under `mjx/`
([docs][mjx-docs]). It runs on NVIDIA GPUs, AMD GPUs, Apple Silicon, and
TPU.

[mjx-docs]: https://mujoco.readthedocs.io/en/stable/mjx.html

**Why it exists.** Put the whole step inside XLA, then `jax.vmap`
thousands of envs and `jax.jit` the entire rollout. Bonus: the result
is **differentiable**, so analytic policy gradients and trajectory
optimization become tractable.

**Differences from C MuJoCo:**

- Explicit device placement: `mjx_model = mjx.put_model(model)`,
  `mjx_data = mjx.make_data(model)`.
- JAX arrays everywhere → fully functional. No in-place mutation, so
  `mjx_data = mjx_data.replace(qpos=...)` instead of `data.qpos[:] = ...`.
- Reduced feature surface in pure-JAX mode: only FREE / BALL / SLIDE /
  HINGE joints, CG and Newton solvers, limited geom collision pairs
  (some BOX/MESH/HFIELD pairs unsupported), no IMPLICIT integrator.

**MJX-Warp.** A sibling backend that uses **NVIDIA Warp** kernels. It
has nearly full MuJoCo feature parity on NVIDIA GPUs but no autodiff. As
of GTC 2025 NVIDIA and DeepMind announced **MJWarp** as one of the
solvers powering **Newton**, an open-source GPU/differentiable physics
engine contributed by NVIDIA, DeepMind, and Disney Research to the Linux
Foundation in September 2025 — and Newton is the physics backend behind
the new Newton path in Isaac Lab. **The two ecosystems are converging.**

**Throughput (verified from docs and papers):**

- Single scene: MJX is ~10× **slower** than C MuJoCo. Use C for
  single-arm latency-sensitive code.
- Many scenes: ~950k steps/sec on TPU v4 humanoid; ~2.7M steps/sec on
  8-chip TPU v5; ~3M steps/sec with MJX-Warp on H100-class GPUs.
- End-to-end PPO: 30-DoF REEM-C humanoid, 8192 parallel envs, 200M steps
  in **56 minutes on one RTX 4090** ([Singh et al. 2024][brax-mjx-2024]).

### 8. The DeepMind ecosystem: Menagerie, Playground, dm_control

Three sibling repos sit on top of MuJoCo/MJX:

**MuJoCo Menagerie** ([repo][menagerie]) — DeepMind-curated MJCF models
for ~80 robots, calibrated by the manufacturers where possible. Notable
inclusions:

- Humanoids: Unitree H1 (19 DoF), Unitree G1 (29 DoF), Booster T1
  (23 DoF), Apptronik Apollo, Berkeley Humanoid, Robotis OP3, PAL TALOS.
- Quadrupeds: Unitree A1/Go1/Go2, ANYmal B/C, Boston Dynamics Spot,
  Google Barkour.
- Arms: Franka Panda, UR5e/UR10e, KUKA iiwa, Kinova Gen3, UFactory xArm.
- End-effectors: Shadow Hand, Allegro, Leap Hand, Robotiq grippers.
- Plus Skydio X2 / Crazyflie drones and biomechanical models.

If you want to train on a specific commercial robot in MuJoCo, look here
first.

**dm_control** — the classic DeepMind env suite (Tassa et al., 2018,
[arXiv:1801.00690][dm-control]). The original Humanoid / Cheetah / Walker
/ Ant tasks that defined the modern MuJoCo benchmark.

**MuJoCo Playground** ([site][playground-site],
[arXiv:2502.08844][playground-paper]) — the **2025** offering: an
open-source robot-learning framework built on MJX. Three env families:

1. **DM Control Suite** — `dm_control` tasks reimplemented on MJX.
2. **Locomotion** — Unitree Go1/Go2, ANYmal, Spot, Barkour, plus
   bipeds/humanoids (Unitree G1 with a joystick policy, Berkeley
   Humanoid, Booster T1, Apptronik Apollo).
3. **Manipulation** — Franka Panda pick/place (`PandaPickCube`),
   dexterous hands, non-prehensile pushing.

Playground includes a **batched renderer** (via MJWarp) so vision-based
policies train on GPU without CPU image roundtrips. It won the
**Outstanding Demo Paper Award at RSS 2025**, and sim-to-real demos on
Unitree G1 joystick locomotion were shown live at the conference.

[menagerie]: https://github.com/google-deepmind/mujoco_menagerie
[dm-control]: https://arxiv.org/abs/1801.00690
[playground-site]: https://playground.mujoco.org/
[playground-paper]: https://arxiv.org/abs/2502.08844

---

## Part III — NVIDIA Isaac: the GPU-parallel stack

### 9. Decoding the Isaac brand: Sim vs. Lab vs. Gym vs. GR00T vs. Cosmos

NVIDIA's naming has churned. Here is the current (May 2026) layout —
worth keeping straight because every other section in Part III assumes
you know which name maps to which artifact.

| Name | What it is | Status |
|---|---|---|
| **Isaac Sim** | Full robotics simulator on Omniverse + OpenUSD. Scene authoring, sensor sim, photoreal RTX rendering, synthetic data | Current. **Isaac Sim 5.x / 6.0**. Fully open-source on GitHub since 5.0. |
| **Isaac Lab** | RL / imitation-learning framework on top of Isaac Sim. Replaces Isaac Gym, OmniIsaacGymEnvs, and Orbit. | Current. **v2.3.2 → v3.0 develop branch**. [Paper][isaaclab-paper] Nov 6, 2025. |
| **Isaac Gym** (legacy) | 2021 GPU-physics RL preview release. | **Deprecated Feb 7, 2025.** Still downloadable, unsupported. |
| **Isaac GR00T** | Humanoid foundation model line (VLA architecture). | **GR00T N1** Mar 2025 ([paper][groot-paper]), **N1.5** June 2025. |
| **Cosmos** | World-foundation-model platform for Physical AI; generates synthetic video / world rollouts. | [Paper][cosmos-paper] Jan 2025. Open weights at `github.com/NVIDIA/Cosmos`. |
| **Isaac ROS** | ROS 2 packages for GPU-accelerated perception (VSLAM, depth, detection). | Deployed on robots alongside the policy. |

[isaaclab-paper]: https://arxiv.org/abs/2511.04831
[groot-paper]: https://arxiv.org/abs/2503.14734
[cosmos-paper]: https://arxiv.org/abs/2501.03575

The architecture top to bottom is:

```
Cosmos                         ── generates synthetic data
    ↓
Isaac Sim (Omniverse + PhysX 5 + RTX renderer)
    ↓
Isaac Lab (gym-like envs, thousands of parallel instances on one GPU)
    ↓
Policy (PPO, VLAs like GR00T)
    ↓
Isaac ROS (perception on the deployed robot)
```

If a 2025 humanoid demo says "trained in NVIDIA Isaac Lab," that means
all five pieces in some combination. If a 2022 quadruped demo says
"trained in Isaac Gym," that is the deprecated stack — translate
mentally to "trained in Isaac Lab today."

### 10. Why Isaac Gym was deprecated and Isaac Lab took over

The pre-2024 NVIDIA RL stack was a tangle:

- **Isaac Gym (Preview 4)** — the original GPU-physics RL framework
  from 2021 ([Makoviychuk et al.][isaac-gym-paper]). Standalone, no
  renderer, research-grade. Brilliant throughput (the first paper to
  show **~1M steps/sec on a single GPU** for legged-robot training),
  but a research preview.
- **OmniIsaacGymEnvs (OIGE)** — a re-port to Omniverse / Isaac Sim
  that added rendering.
- **Orbit** — a community framework on top of Isaac Sim with a
  manager-based env design.

This was confusing. **Isaac Lab unified all three.** It absorbed Orbit's
env API, replaced OIGE, and obsoleted Isaac Gym Preview. NVIDIA formally
deprecated Isaac Gym on **February 7, 2025** ([forum
notice][isaac-gym-deprecation]) — every new project is expected to start
on Isaac Lab.

The Isaac Lab paper landed November 6, 2025 ([Mittal et al.][isaaclab-paper]).
It is now the canonical reference.

[isaac-gym-paper]: https://arxiv.org/abs/2108.10470
[isaac-gym-deprecation]: https://forums.developer.nvidia.com/t/isaac-gym-deprecation-transition-to-isaac-lab/322978

### 11. PhysX 5 and the tensor API

Two things make Isaac Lab fast.

**PhysX 5.** The underlying GPU rigid-body physics engine. Unlike
MuJoCo's convex soft contacts, PhysX uses an impulse-based contact
solver tuned for game physics first and robotics second. It is faster
than MuJoCo on hard contacts and visually more correct (no
penetration), but it is also less smooth — gradients through PhysX are
not first-class. (NVIDIA's answer to that is **Newton**, the
DeepMind/NVIDIA/Disney effort mentioned in §7 that is rolling into
Isaac Lab as an experimental backend.)

**Tensor API.** Observations and rewards live on the GPU as PyTorch
tensors. There is no CPU↔GPU shuffling per env per step — the entire
inner loop runs on the device.

**Concrete throughput** (single RTX 4090, [official benchmarks][isaac-bench]):

| Env | Steps/sec | Notes |
|---|---|---|
| `Isaac-Cartpole-Direct-v0` | **~1.1M** | toy upper bound |
| `Isaac-Velocity-Rough-G1-v0` (Unitree G1, rough terrain) | **~94k** | humanoid locomotion |
| `Isaac-Repose-Cube-Shadow-Direct-v0` (dexterous hand) | **~200k** | in-hand manipulation |
| `Isaac-Cartpole-RGB-Camera-Direct-v0` (with camera) | **~50k** | vision policy |
| Multi-node 4×4 L40, cartpole | **~10.2M** | scaling sanity check |
| Multi-node 4×4 L40, G1 locomotion | **~1.2M** | scaling sanity check |

NVIDIA's marketing material for **Unitree H1 locomotion** quotes
**~135k FPS** on Isaac Lab, with training time in **minutes** instead
of days for CPU-bound stacks. Reproduce-it-yourself caveats apply, but
the order of magnitude is real and is the entire reason Isaac Lab won.

### 12. Isaac Lab task structure for RL practitioners

Isaac Lab ships two env workflows. You will see both in the wild:

**Manager-based** (`envs.ManagerBasedRLEnv`). The env is decomposed into
managers — `ObservationManager`, `ActionManager`, `RewardManager`,
`TerminationManager`, `CommandManager`, `CurriculumManager`. You write
config dataclasses; the env composes them. Most locomotion configs use
this path because it makes reward shaping and curricula easy to mix
and match.

```python
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
import isaaclab.envs.mdp as mdp

@configclass
class CartpoleRewardsCfg:
    alive   = RewTerm(func=mdp.is_alive, weight=1.0)
    upright = RewTerm(
        func=mdp.pole_pos_l2,
        params={"asset_cfg": SceneEntityCfg("robot")},
        weight=-1.0,
    )

@configclass
class CartpoleTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    cart_oob = DoneTerm(
        func=mdp.joint_pos_out_of_limit,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"])},
    )
```

The full pattern: define `SceneCfg`, `ObservationsCfg`, `ActionsCfg`,
`RewardsCfg`, `TerminationsCfg`, glue them into an `EnvCfg`, register a
Gym ID, then `isaaclab.sh -p .../train.py --task <id>` runs across
thousands of parallel envs on one GPU.

**Direct** (`envs.DirectRLEnv`). A single Python class with explicit
`_get_observations`, `_get_rewards`, `_get_dones`. Faster for tight
inner loops and simpler tasks; less reusable.

**RL framework integrations.** Each lives under
`scripts/reinforcement_learning/<lib>/train.py`:

- `rl_games` — NVIDIA's high-performance PPO, most locomotion configs
  default here.
- `rsl_rl` — ETH's RL library, the canonical choice for legged-robot
  papers from the ANYmal lineage.
- `skrl` — modular, multi-algorithm.
- `stable-baselines3` — familiar from Week 2/3 CleanRL crowd.

**Built-in tasks.** As of v2.3.2:

- Locomotion velocity-tracking across **11 morphologies**: Unitree
  A1/Go1/Go2/H1/G1, ANYmal-B/C/D, Boston Dynamics Spot, Agility Digit,
  Cassie. Flat + rough terrain variants.
- Manipulation: Franka Panda (reach, lift, stack), UR10, Shadow Hand
  cube reposing, OpenArm envs (new in v2.3.2).
- Navigation: Jetbot, wheeled bases.
- Drone/multirotor support, visual tactile sensors, Haply teleop —
  all in v2.3.2.

### 13. GR00T, Cosmos, and the humanoid foundation-model stack

The 2025 NVIDIA story is no longer "train a PPO policy in Isaac Lab" —
it is "**train a humanoid foundation model on synthetic data from
Cosmos, simulated in Isaac Sim, fine-tuned in Isaac Lab, deployed on
Apptronik Apollo or 1X Neo**." That stack has three new pieces.

**Isaac GR00T** ([N1 paper][groot-paper], March 2025) is NVIDIA's open
humanoid VLA — *vision-language-action* foundation model. Architecture:

- **System 2** — a vision-language model (VLM) for reasoning. In N1.5
  this is **Eagle 2.5**, frozen.
- **System 1** — a diffusion transformer that turns the VLM's latent
  goal into continuous robot actions.

GR00T N1 was trained on humanoid teleoperation data + synthetic data
from Cosmos + Isaac Sim. N1.5 (June 2025) keeps the System 2 frozen and
adds the FLARE objective for learning from human ego-video. Open weights
at `nvidia/GR00T-N1-2B` on HuggingFace.

**Cosmos** ([paper][cosmos-paper], January 2025) is a world-foundation-model
platform: train large generative models on millions of hours of physical
video, then sample synthetic robot data from them. The marketing slogan
is "**Physical AI**." Cosmos-Drive-Dreams ([arXiv:2506.09042][cosmos-drive])
is the driving variant.

**Isaac-Sim**-based synthetic data pipelines named **GR00T-Dreams** and
**GR00T-Mimic** turn Cosmos rollouts into supervised training data for
GR00T fine-tuning. The actual humanoid loop looks like:

```
Cosmos rollouts → Isaac Sim render → GR00T-Mimic / GR00T-Dreams
                                              ↓
                          GR00T training (synthetic + real teleop)
                                              ↓
                              Isaac Lab fine-tuning (PPO / DAgger)
                                              ↓
                                       Deploy on H1 / G1 / Apollo
```

This is the new NVIDIA recipe in one diagram. PPO is still in there, but
it is the *fine-tuner* on top of a pretrained VLA — exactly the shape
the LLM world has been training in since Week 5.

[cosmos-drive]: https://arxiv.org/abs/2506.09042

---

## Part IV — The sim-to-real workflow

### 14. Domain randomization: the OpenAI recipe

If your simulator is wrong by some unknown amount, the simplest defense
is to train on a *distribution* of simulators wide enough that reality
is somewhere inside it. That is **domain randomization** (DR).

Three canonical references, in order:

1. **Tobin et al., 2017 — visual DR** ([arXiv:1703.06907][tobin]).
   Randomized textures, lighting, and camera positions in sim so that a
   vision policy trained only on synthetic images transferred zero-shot
   to a real grasping task.
2. **Peng et al., 2017 — dynamics DR** ([arXiv:1710.06537][peng]).
   Randomized mass, friction, damping for a robot arm. Established the
   "randomize the physics" half of the recipe.
3. **OpenAI et al., 2019 — Automatic DR (ADR) on Dactyl /
   Rubik's Cube** ([arXiv:1910.07113][adr]). The randomization range
   **grows automatically** as the policy gets better; this is what kept
   training stable instead of mode-collapsing on too-wide a distribution.

[tobin]: https://arxiv.org/abs/1703.06907
[peng]: https://arxiv.org/abs/1710.06537
[adr]: https://arxiv.org/abs/1910.07113

**Standard axes** (compose all of them; this is the modern recipe):

| Family | Axis | Typical range |
|---|---|---|
| Physics | Link masses | ±20–40% |
| Physics | Friction coefficients | ±50% |
| Physics | Joint damping / armature | ±50% |
| Physics | Motor strength / PD gains | ±20% |
| Latency | Control delay | 0–30 ms |
| Sensor | IMU noise (Gaussian) | σ tuned per axis |
| Sensor | Encoder quantization | hardware-matched |
| Disturbance | Pushes / external forces | episodic |
| Terrain | Height-field roughness | flat → step-up |
| Vision | Lighting / textures / distractors | broad |

**When DR works.** When the policy can use proprioceptive history to
*implicitly identify dynamics* — a memory-equipped network like an LSTM
or a transformer policy effectively does system identification in
latent space. This is the principle that makes RMA (§15) work and what
the Figure / Booster / Unitree humanoid demos rely on.

**When DR fails.**

- **Too wide.** Mode collapse: policy becomes overly conservative
  ("walk slowly so nothing breaks"), or refuses to learn at all because
  the value function never converges.
- **Too narrow.** Reality is outside the training distribution and
  transfer breaks.

ADR fixes both by curriculum: start narrow, grow when reward exceeds a
threshold. This is the de-facto modern recipe.

### 15. Rapid Motor Adaptation (RMA) and teacher–student distillation

**Kumar, Fu, Pathak, Malik, 2021 — "RMA: Rapid Motor Adaptation for
Legged Robots"** ([arXiv:2107.04034][rma], RSS 2021). Two-phase
architecture.

[rma]: https://arxiv.org/abs/2107.04034

**Phase 1.** Train a base policy that has access to *privileged
environment information* — true mass, true friction, true terrain
height — via an environment encoder that produces a latent extrinsics
vector `z`:

```math
z_t = \mu(e_t), \qquad a_t = \pi(o_t, z_t)
```

where `e_t` is the privileged info, `μ` is the encoder, and `π` is the
policy. PPO trains everything end-to-end with full access to `e_t`.

**Phase 2.** Train an **adaptation module** `φ` that predicts `z_t`
*from a short window of proprioceptive history* — joint angles, joint
torques, IMU readings — via supervised regression against the
privileged encoder:

```math
\hat z_t = \phi(o_{t-k:t}, a_{t-k:t-1}), \qquad
\mathcal{L} = \| \hat z_t - z_t \|^2
```

At deployment the robot only has proprioception, so it uses `φ` to
estimate `\hat z_t`, then `π(o_t, \hat z_t)` produces the action. The
adaptation happens in *fractions of a second* — fast enough that the
robot can step from grass to oil to stairs and re-identify on the fly.

The original RMA paper deployed zero-shot on Unitree A1 across grass,
mud, oil, payload changes, and stairs without retraining. Extensions
include:

- **RMA 2.0** / "Coupling Vision and Proprioception" (Fu, Kumar, Malik,
  Pathak 2022) — vision + proprioception with the same teacher-student
  pattern.
- **Extreme Parkour** (Cheng, Pathak, 2023) — same skeleton, harder
  terrain.

The conceptual predecessor is **Hwangbo et al., 2019 — "Learning agile
and dynamic motor skills for legged robots"** ([arXiv:1901.08652][hwangbo],
*Science Robotics*) on ANYmal, which established the modern
sim-to-real pipeline with **actuator network modeling** (learn the
motor's transfer function from real data, then bake it into the sim).

Lee et al. then made the **teacher–student split explicit** in 2020 —
"Learning Quadrupedal Locomotion over Challenging Terrain"
([arXiv:2010.11251][lee], *Science Robotics*): a teacher gets the
privileged terrain map; a student gets only proprioception and is
trained via supervised distillation from the teacher.

[hwangbo]: https://arxiv.org/abs/1901.08652
[lee]: https://arxiv.org/abs/2010.11251

**The pattern dominates legged sim-to-real because:**

1. RL is much easier when the teacher can "cheat" with privileged info.
2. The student's observation space is fixed by hardware (proprioception
   + maybe vision) and cannot be expanded.
3. Distillation is a stable supervised problem — no exploration, no
   variance.

If you read any 2024–2026 quadruped or humanoid locomotion paper, the
core algorithm will look like the above with new bells and whistles.

### 16. System identification and real-to-sim

DR/RMA make the **policy** robust to gap. The complementary direction
makes the **simulator** match reality.

**Classical system identification.** Roll out the real robot under
specific torques, fit the rigid-body equations of motion to the
observed joint trajectories, and update the URDF / actuator parameters.
This is the engineering-first answer — boring and effective.

**Real-to-sim.** Use a small batch of real rollouts to train a residual
dynamics model that corrects sim trajectories to match observed real
ones. The policy is then trained or fine-tuned in the corrected sim.
The 2025 paper **PolySim** ([arXiv:2510.01708][polysim]) is the
state-of-the-art version — train across multiple simulators in
parallel, randomize between them, and the policy generalizes to
real-world without per-robot recalibration.

[polysim]: https://arxiv.org/abs/2510.01708

**Actuator networks** (Hwangbo 2019). Specifically learn the motor's
transfer function from real torque-command / joint-velocity pairs and
splice that learned model into the simulator. This is what made ANYmal
work and is still the standard sub-recipe for legged hardware.

### 17. Differentiable simulation: when gradients beat sampling

If the simulator is differentiable, you can compute
$\nabla_{\theta} \mathcal{L}$ directly instead of estimating it via
score-function gradients on millions of rollouts. The promise is
**1000× sample efficiency**; the catch is that contact non-smoothness
breaks gradients exactly where the interesting physics happens.

The lineage:

- **Brax** ([Freeman et al., 2021][brax-paper], arXiv:2106.13281) —
  fully differentiable rigid-body engine in JAX. Originally with its
  own custom physics; in 2024 absorbed MJX as the recommended physics
  backend.
- **MJX** — JAX reimplementation of MuJoCo (§7). Bundled with Brax
  through `mjx.brax`.
- **Hard Contacts with Soft Gradients** ([arXiv:2506.14186][hard-soft],
  June 2025) — refined MJX gradients for learning and control. State of
  the art for getting useful gradients out of stiff contact dynamics.

[brax-paper]: https://arxiv.org/abs/2106.13281
[hard-soft]: https://arxiv.org/abs/2506.14186

**When differentiable sim wins:** smooth-dynamics tasks like quadrotor
control, soft-body manipulation, dexterous trajectory optimization with
contact schedules pre-specified.

**When it loses:** the moment hard contact behavior is the point —
running, jumping, hitting a ball — where the gradient becomes a delta
function and PPO-on-MJX-rollouts is empirically more reliable than
analytic policy gradients.

### 18. The PPO-dominance story and when SAC or Dreamer wins instead

The robotics RL community has converged on **PPO**. ANYmal, Unitree,
Booster, Berkeley Humanoid, Figure, the canonical Isaac Lab
configurations, the canonical MuJoCo Playground configurations — all
PPO-style. Why:

- **Parallelism.** PPO is trivially parallelizable across thousands of
  envs and fits the GPU-batched-sim shape perfectly. The on-policy
  rollouts ARE the data.
- **Stability.** Tolerates reward shaping and DR noise; the clipped
  surrogate makes large parameter steps safe.
- **Mature recipes.** Every framework ships a PPO baseline tuned for
  legged robots. No hyperparameter mystery.

**When SAC wins.**

- Small-scale continuous control where you cannot run massive parallel
  envs (e.g. real-robot fine-tuning).
- Sample efficiency matters more than wall-clock throughput.
- You want a deterministic policy at deployment.

Empirically, on the same hardware budget PPO almost always wins for
locomotion; SAC almost always wins for sample-budget-constrained
manipulation.

**When model-based wins.**

- When *real-world* sample efficiency is paramount and there is no
  simulator at all. The canonical example is **DayDreamer** (Wu,
  Escontrela, Hafner et al., [arXiv:2206.14176][daydreamer], 2022) —
  Dreamer trained a real quadruped from scratch in **~1 hour** of real
  time, using only a learned world model. No simulator. This is the
  bridge to Week 7 — see §27.

[daydreamer]: https://arxiv.org/abs/2206.14176

**Where RLHF / GRPO / RLVR are.** Not used for low-level robot control.
The closest analog is preference-based reward learning for ill-specified
tasks ("walk *gracefully*"), which remains a niche subfield.

---

## Part V — Landmark wins, 2024–2026

### 19. Quadruped locomotion: the canonical sim-to-real success

Quadruped locomotion is the most-solved sim-to-real problem in robotics.
The arc:

- **2019 — Hwangbo et al. on ANYmal** ([arXiv:1901.08652][hwangbo]).
  Established the actuator-network + DR pipeline. First *Science
  Robotics* paper on RL legged locomotion.
- **2020 — Lee et al., ANYmal over rough terrain**
  ([arXiv:2010.11251][lee]). Explicit teacher–student.
- **2021 — RMA on Unitree A1** ([arXiv:2107.04034][rma]). Open-source
  recipe, ran on consumer hardware.
- **2022 — DayDreamer on quadruped** ([arXiv:2206.14176][daydreamer]).
  Real-world from scratch, no sim, in 1 hour.
- **2023 — Extreme Parkour** (Cheng, Pathak). The RMA recipe scaled to
  obstacle courses.
- **2024–2026** — every commercial quadruped (Unitree Go2, ANYmal D,
  Boston Dynamics Spot via their hybrid stack) ships with RL-trained
  locomotion policies trained in some combination of MuJoCo / Isaac Lab
  / proprietary sim. Most academic papers in 2025 use Isaac Lab.

The current open-source community baseline is `legged_gym` on Isaac Lab
+ rsl_rl + PPO + DR + teacher–student distillation, tuned on Unitree Go1
or Go2.

### 20. Humanoids in 2025–2026: Berkeley, Unitree, Booster, Figure, 1X

Humanoid sim-to-real became *the* RL story of 2025. Three buckets of
work:

**A — Academic, paper-backed.**

- **Berkeley Humanoid** ([arXiv:2407.21781][berkeley], July 2024). A
  mid-scale humanoid research platform; light DR, walks outdoor
  terrain. The training stack is open-sourced.
- **ASAP — Aligning Simulation and Real-world Physics** for whole-body
  humanoid skills on Unitree G1 ([arXiv:2502.01143][asap], Feb 2025).
  Uses Isaac Lab; one of the strongest sim-to-real humanoid results of
  the year.
- **Booster Gym** ([arXiv:2506.15132][booster-gym], June 2025). End-to-end
  RL framework for Booster T1 humanoid locomotion. Open recipe.
- **VIRAL — Visual Sim-to-Real at Scale for Humanoid Loco-Manipulation**
  ([arXiv:2511.15200][viral], Nov 2025).
- **Opening the Sim-to-Real Door** ([arXiv:2512.01061][s2r-door],
  Dec 2025) — first humanoid sim-to-real for articulated
  loco-manipulation from pure RGB.
- **Sim-to-Real Humanoid Locomotion in 15 Minutes**
  ([arXiv:2512.01996][s2r-15min], Dec 2025). FastTD3/FastSAC recipe;
  trains robust Unitree G1 / Booster T1 walking on a **single RTX
  4090 in 15 minutes**. The "you can do this at home" entry point.

[berkeley]: https://arxiv.org/abs/2407.21781
[asap]: https://arxiv.org/abs/2502.01143
[booster-gym]: https://arxiv.org/abs/2506.15132
[viral]: https://arxiv.org/abs/2511.15200
[s2r-door]: https://arxiv.org/abs/2512.01061
[s2r-15min]: https://arxiv.org/abs/2512.01996

**B — Industry, blog-confirmed only (no peer-reviewed paper).**

- **Figure 02 / 03.** Figure's "Natural Humanoid Walk Using
  Reinforcement Learning" [blog][figure-walk] confirms: thousands of
  robots in parallel in high-fidelity sim, per-instance randomized
  physics, single neural-net policy across all instances, zero-shot
  transfer enabled by DR + high-frequency torque feedback. Figure 03's
  Helix 02 / "System 0" [blog][figure-helix02] claims 200,000+
  parallel envs and >1,000 hours of human motion data.
- **Apptronik Apollo.** Publicly announced integration with NVIDIA
  Project GR00T / Isaac Lab. No technical paper.
- **1X Neo.** Public neural-net policies and sim training; no paper.
- **Tesla Optimus Gen 3.** Tesla material claims a sim-to-real pipeline
  combined with imitation from human video; in 2025 reportedly pivoted
  toward vision-only training using recorded worker video. No paper.

[figure-walk]: https://www.figure.ai/news/reinforcement-learning-walking
[figure-helix02]: https://www.figure.ai/news/helix-02

**C — Foundation models.** GR00T N1 / N1.5 (§13) — VLA-style humanoid
foundation models trained on Cosmos synthetic data + real teleop, then
fine-tuned per-task in Isaac Lab.

The story across all three: **DR + teacher-student + Isaac Lab (or MJX
Playground) + PPO** is the boring backbone; what differs is the data
and the system around the policy.

### 21. Manipulation: Aloha, ManiSkill, π0, and the imitation-vs-RL boundary

Locomotion is the canonical *RL* sim-to-real success. Manipulation
flipped the other way — most of the splashy 2024–2026 results are
**imitation learning** on real teleop data, not RL on sim data. The
reason: high-dimensional contact-rich manipulation has very hard reward
specification, and physical robots in human-supervised teleop are
faster to get than tuning a hand-shaped sim reward.

The state of the art:

- **ALOHA** ([arXiv:2304.13705][aloha], RSS 2023). Low-cost bimanual
  teleop. **ALOHA 2** ([arXiv:2405.02292][aloha2], 2024). **Mobile
  ALOHA** ([arXiv:2401.02117][mobile-aloha], 2024). All imitation, no
  sim RL.
- **RT-2** ([arXiv:2307.15818][rt2], 2023) and **Open X-Embodiment / RT-X**
  ([arXiv:2310.08864][rt-x], 2023). VLM-action models trained on 22
  robots, 21 institutions, 527 skills.
- **π0** ([arXiv:2410.24164][pi0], Oct 2024) — Physical Intelligence's
  flow-matching VLA. **π0.5** ([arXiv:2504.16054][pi05], April 2025)
  generalizes to unseen homes. **π0.6** (model card Nov 17, 2025).
- **ManiSkill / ManiSkill3** ([arXiv:2410.00425][maniskill3], Oct 2024)
  — GPU-parallel manipulation sim on SAPIEN; the manipulation analog of
  Isaac Lab. RSS 2025 paper.

[aloha]: https://arxiv.org/abs/2304.13705
[aloha2]: https://arxiv.org/abs/2405.02292
[mobile-aloha]: https://arxiv.org/abs/2401.02117
[rt2]: https://arxiv.org/abs/2307.15818
[rt-x]: https://arxiv.org/abs/2310.08864
[pi0]: https://arxiv.org/abs/2410.24164
[pi05]: https://arxiv.org/abs/2504.16054
[maniskill3]: https://arxiv.org/abs/2410.00425

The takeaway from this section: **if your problem is locomotion, train
in sim. If your problem is fine manipulation, collect teleop. The
boundary is moving — π0-style VLAs blur it — but it is the right
default in 2026.**

---

## Part VI — Hands-on starting points

The snippets in this section are intentionally minimal — load a model,
step physics, train one task. For a curated, session-ready demo
sequence with copy-pasteable commands, expected wall-clock per step,
and a 90-minute lesson plan that wires three demos together, see the
**companion run guide:**
[`docs/week8-demos.md`](week8-demos.md).

### 22. Minimal MuJoCo: load, step, render

A complete script that loads a humanoid, steps it for 30 seconds, and
shows a passive viewer:

```python
import time, mujoco, mujoco.viewer

m = mujoco.MjModel.from_xml_path("humanoid.xml")
d = mujoco.MjData(m)

with mujoco.viewer.launch_passive(m, d) as viewer:
    start = time.time()
    while viewer.is_running() and time.time() - start < 30:
        step_start = time.time()
        mujoco.mj_step(m, d)
        viewer.sync()
        dt = m.opt.timestep - (time.time() - step_start)
        if dt > 0:
            time.sleep(dt)
```

Run with `python script.py` on Linux/Windows or `mjpython script.py`
on macOS. Pull `humanoid.xml` from
[Menagerie](https://github.com/google-deepmind/mujoco_menagerie).

### 23. Minimal MJX: vectorized rollouts

```python
import jax
import mujoco
from mujoco import mjx

model     = mujoco.MjModel.from_xml_path("humanoid.xml")
mjx_model = mjx.put_model(model)

@jax.vmap
def batched_step(vel):
    d = mjx.make_data(mjx_model)
    d = d.replace(qvel=d.qvel.at[0].set(vel))
    return mjx.step(mjx_model, d).qpos[0]

vel = jax.numpy.arange(0.0, 1.0, 0.01)       # 100 parallel envs
pos = jax.jit(batched_step)(vel)
```

That single function call evaluates 100 parallel humanoid steps on the
GPU. Combine with a PPO loop (Brax ships one) and you have the entire
training inner loop.

### 24. Minimal Isaac Lab: training a quadruped in one command

After installing Isaac Lab per the [official guide][isaaclab-install]:

```bash
# Headless PPO on Cartpole (sanity check, finishes in minutes)
./isaaclab.sh -p scripts/reinforcement_learning/sb3/train.py \
    --task Isaac-Cartpole-v0 --num_envs 64 --headless

# rsl_rl on Unitree G1 rough-terrain locomotion
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Rough-G1-v0 --headless

# Visualize the trained checkpoint
./isaaclab.sh -p scripts/reinforcement_learning/sb3/play.py \
    --task Isaac-Cartpole-v0 --num_envs 32 --use_last_checkpoint
```

That is the entire pipeline — define an env config, run the train
script, run the play script. The 4096 parallel envs and the GPU physics
are invisible to your code.

[isaaclab-install]: https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html

### 25. Common gotchas: control frequency, termination, reward hacking

A non-exhaustive list of things that will burn you the first time:

- **Control frequency vs. physics dt.** Sim physics dt is typically
  1–4 ms; policy frequency is typically 50–200 Hz. **Decouple them.**
  The standard pattern is `for _ in range(decimation): mj_step(...)`
  between policy actions. A common bug is matching them and getting a
  policy that only works at sim's tiny dt — fails completely at the
  robot's actual control rate.
- **Action space.** PD-target joint positions transfer far better than
  raw torques. Clamp magnitudes, clip rates of change, and inject
  randomized PD gains during training.
- **Observation space.** Never feed raw absolute world position to a
  locomotion policy — use body-frame velocities, gravity vector, joint
  angles, joint velocities, and short observation history.
- **Termination handling.** Terminating on falls vs. letting episodes
  run produces very different policies; reward-on-termination must be
  signed carefully to avoid biasing toward "suicide" (jump off the
  cliff because it ends the negative-reward episode faster).
- **Reward hacking.** Watch for policies that hook legs through the
  floor, exploit contact penetration, or find numerical edges in the
  soft-contact solver. Diagnostic: if reward collapses when you
  slightly perturb physics, you were exploiting a sim bug. See
  Lilian Weng's [reward hacking survey][weng-reward-hacking] for the
  full taxonomy.
- **Determinism.** GPU-batched physics is **not** bitwise deterministic
  across runs. Pin seeds and use deterministic flags but expect some
  drift. CleanRL-style single-file scripts (Week 2's DQN, Week 3's PPO)
  are gold for reproducibility once you escape Isaac/Brax.

[weng-reward-hacking]: https://lilianweng.github.io/posts/2024-11-28-reward-hacking/

---

## Part VII — Bridge back to the rest of the course

### 26. From CartPole to a quadruped: what changes and what doesn't

The Week 2 DQN script and a Unitree G1 humanoid training run are the
**same algorithm template**. What changes:

| | Week 2 CartPole DQN | Isaac Lab G1 PPO |
|---|---|---|
| State dim | 4 | ~100 (joint pos/vel + IMU + history) |
| Action dim | 2 (discrete) | 29 (continuous, PD targets) |
| Episode length | ~500 | ~1000–2000 |
| Algorithm | DQN | PPO |
| Parallel envs | 1 | 4096 |
| Sim | Gymnasium CartPole | Isaac Sim + PhysX 5 |
| Reward | +1 per alive step | Shaped: velocity, orientation, smoothness |
| Wall clock to solve | ~5 min | ~1 hour |

What doesn't change:

- The Bellman equation, the value-function intuition, and the
  exploration/exploitation tradeoff (Week 2).
- The actor-critic split, GAE, and the clipped surrogate (Week 3).
- The "rollout → update → repeat" loop structure.

If you understood DQN on CartPole, the conceptual jump to PPO on a
humanoid is mostly **environment engineering**, not new RL.

### 27. Where world models (Week 7) fit into sim-to-real

Week 7 was about teaching an *LLM* agent to predict its environment in
the action language (RWML). The world-models story in robotics is older
and more concrete:

- **Hand-built simulator** (MuJoCo, Isaac Sim) = the world model is
  *given*, written by engineers in C++.
- **Learned world model in pixels** (Dreamer V3, V-JEPA 2) = learn a
  latent dynamics model from video and rollouts.
- **Hybrid** (DayDreamer) = learn the world model on the real robot
  while also acting — bypasses the hand-built sim entirely.

The trade-off is sample efficiency vs. bias:

- A perfect hand-built sim has zero learned-model error and infinite
  sample efficiency but a non-trivial reality gap.
- A learned world model has model error but matches real distribution
  by construction.

Modern research is converging on the obvious hybrid: **train mostly in
sim with DR, then refine the world model with a few real rollouts** —
which is exactly the PolySim / real-to-sim story in §16. The Week 7
world model is the language-agent analog of the Week 8 simulator.

### 28. The reward-source arc revisited: RLHF → RLVR → RWML → physical sim

A through-line of this course has been *where does the reward signal
come from?*

- **Week 2 (DQN, CartPole)** — engineered scalar (+1 alive).
- **Week 3 (PPO, Atari)** — game-engine score.
- **Week 5–6 (RLHF)** — human preferences via a learned reward model.
- **Week 6 (RLVR)** — verifier (math grader, code interpreter).
- **Week 7 (RWML)** — environment ground truth, scored by sentence
  similarity to the predicted next state.
- **Week 8 (physical sim)** — engineered scalar **plus** physics
  constraints. The reward is hand-shaped (velocity tracking, foot
  clearance, energy), but the *physics enforces* what "valid" means.

Each step the reward moves further from a single number a human typed
into a Python file and closer to a multi-modal verifier the model
cannot game. Physical sim is the most concrete verifier of all — break
the laws of motion and you simply fall over. That is also why
sim-to-real exists: the verifier is too clean. Reality has noise that
the verifier never knew about.

---

## Part VIII — Q&A

### 29. Q&A — Which simulator should I pick?

> **Q: I'm starting a project. MuJoCo or Isaac Sim / Lab?**
>
> **A:** It depends on what you actually care about.
>
> Pick **MuJoCo / MJX / Playground** if:
> - You're a JAX person, or you want differentiability.
> - You want a small dependency footprint (one `pip install mujoco` and
>   you're running on macOS / Linux / Windows).
> - You want smooth contacts for trajectory optimization or analytic
>   gradients.
> - You're publishing a paper and reproducibility matters.
>
> Pick **Isaac Sim + Isaac Lab** if:
> - You want photorealistic vision-based policies (RTX rendering).
> - You want to scale to thousands of parallel envs out of the box with
>   PyTorch.
> - You're on the NVIDIA stack (PhysX 5, GR00T, Cosmos) and want
>   one-vendor support.
> - Your robot model exists in Isaac's catalogue and you don't want to
>   port it.
>
> **Use both** if you're working on humanoids. Several 2025 papers
> train in one, validate in the other — multi-sim is itself a form of
> domain randomization (see PolySim §16).

> **Q: What about Gazebo, PyBullet, SAPIEN, Brax, Drake?**
>
> **A:** Each has a niche.
>
> - **Gazebo** — the ROS default. Great for full-system integration
>   testing, mediocre for RL throughput. Used downstream of Isaac.
> - **PyBullet** — the 2017–2020 community default. Largely superseded
>   by MuJoCo (now free) for RL.
> - **SAPIEN** — the physics engine under **ManiSkill3** (§21). Great
>   for manipulation; smaller community than MuJoCo/Isaac.
> - **Brax** — the Google JAX-native RL engine; in 2024 it switched
>   from its own custom physics to MJX as the recommended backend.
>   Effectively a sibling of MJX now.
> - **Drake** — Toyota / Russ Tedrake's planning-and-control simulator.
>   Excellent for model-based control research; not optimized for
>   high-throughput RL.

### 30. Q&A — Sim-to-real failure modes

> **Q: My policy works perfectly in sim but the real robot won't even
> stand up. What now?**
>
> **A:** Classical pattern. Diagnose in order:
>
> 1. **Latency.** Add 10–30 ms of simulated control delay during
>    training. If the policy collapses, you were over-fitting on a
>    zero-delay sim.
> 2. **PD gains.** Are the sim PD gains matched to the real motor
>    controller? If sim ran at 200 Hz with stiff PD and the robot runs
>    at 100 Hz with soft PD, you have an actuator gap.
> 3. **Friction.** Real friction is rarely the URDF nominal value. Run
>    DR with ±50% friction; if the sim policy now wobbles, that was the
>    issue.
> 4. **Observation noise.** Are you feeding the robot's raw noisy IMU
>    to a policy trained on perfect sim observations?
> 5. **The classic reward hack.** Watch the sim policy frame by frame.
>    If it does anything subtly weird — hooks a foot, slides instead
>    of stepping — your sim was being exploited.

> **Q: What's the cheapest thing I can do to improve sim-to-real?**
>
> **A:** Random pushes during training. Episodically apply a random
> external force to the robot's base. It's a 5-line change that buys
> ~half the robustness DR gives you, for free.

### 31. Q&A — Hardware, scale, and reproducibility

> **Q: What hardware do I need?**
>
> **A:** Surprisingly little for the headline results.
>
> - **MJX Playground recipes** (Unitree G1 joystick): a single RTX 4090
>   is more than enough.
> - **Isaac Lab built-in tasks**: a single RTX 4090; an RTX 3090 works
>   with smaller `num_envs`. **Cartpole-Direct-v0** runs >1M
>   steps/sec on a 4090.
> - **The "humanoid in 15 min" recipe** (arXiv:2512.01996): a single
>   RTX 4090.
> - **Multi-GPU** matters for vision-based policies and Cosmos-scale
>   synthetic data, not for proprioception locomotion.

> **Q: Can I reproduce a paper's sim-to-real result on my own robot?**
>
> **A:** Maybe.
>
> - **Quadruped (Unitree Go1/Go2/A1)** — yes. The open recipe works
>   and there is a robust community.
> - **Humanoid (Unitree G1, Booster T1)** — yes if you can get the
>   hardware. Codebases (`unitree_rl_lab`, `Booster_Gym`,
>   `humanoid-gym`) are open.
> - **Custom hardware** — expect 2–6 weeks of system-ID and actuator
>   modeling before training pays off.

---

## Part IX — Resources

### 32. Papers, blogs, videos, code

**Foundations and surveys.**

- Todorov, Erez, Tassa, "MuJoCo: A physics engine for model-based
  control," IROS 2012 — the original MuJoCo paper.
- Makoviychuk et al., "Isaac Gym: High-Performance GPU-Based Physics
  Simulation for Robot Learning,"
  [arXiv:2108.10470](https://arxiv.org/abs/2108.10470) (2021).
- Mittal et al., "Isaac Lab: Robot Learning at Scale,"
  [arXiv:2511.04831](https://arxiv.org/abs/2511.04831) (Nov 2025) — the
  canonical Isaac Lab paper.
- Zakka et al., "MuJoCo Playground,"
  [arXiv:2502.08844](https://arxiv.org/abs/2502.08844) (Feb 2025) — RSS
  2025 Outstanding Demo Paper.
- Singh et al., "Learning Velocity-based Humanoid Locomotion with Brax
  and MJX," [arXiv:2407.05148](https://arxiv.org/abs/2407.05148) — useful
  concrete training-budget numbers.
- Freeman et al., "Brax — A Differentiable Physics Engine for Large-Scale
  Rigid Body Simulation,"
  [arXiv:2106.13281](https://arxiv.org/abs/2106.13281).

**Sim-to-real foundations.**

- Tobin et al., "Domain Randomization,"
  [arXiv:1703.06907](https://arxiv.org/abs/1703.06907) (2017).
- Peng et al., "Sim-to-Real Transfer with Dynamics Randomization,"
  [arXiv:1710.06537](https://arxiv.org/abs/1710.06537) (2017).
- OpenAI et al., "Solving Rubik's Cube with a Robot Hand" — ADR,
  [arXiv:1910.07113](https://arxiv.org/abs/1910.07113) (2019).
- Kumar et al., "RMA: Rapid Motor Adaptation for Legged Robots,"
  [arXiv:2107.04034](https://arxiv.org/abs/2107.04034) (RSS 2021).
- Hwangbo et al., "Learning Agile and Dynamic Motor Skills for Legged
  Robots," [arXiv:1901.08652](https://arxiv.org/abs/1901.08652) (*Science
  Robotics* 2019).
- Lee et al., "Learning Quadrupedal Locomotion over Challenging Terrain,"
  [arXiv:2010.11251](https://arxiv.org/abs/2010.11251) (*Science Robotics*
  2020).
- Wu, Escontrela, Hafner, et al., "DayDreamer,"
  [arXiv:2206.14176](https://arxiv.org/abs/2206.14176) (2022).
- "PolySim: Bridging the Sim-to-Real Gap via Multi-Simulator Dynamics
  Randomization,"
  [arXiv:2510.01708](https://arxiv.org/abs/2510.01708) (Oct 2025).
- "Hard Contacts with Soft Gradients,"
  [arXiv:2506.14186](https://arxiv.org/abs/2506.14186) (June 2025).

**Humanoid sim-to-real 2024–2026.**

- Berkeley Humanoid,
  [arXiv:2407.21781](https://arxiv.org/abs/2407.21781).
- ASAP (Unitree G1, Isaac Lab),
  [arXiv:2502.01143](https://arxiv.org/abs/2502.01143).
- Booster Gym, [arXiv:2506.15132](https://arxiv.org/abs/2506.15132).
- VIRAL Humanoid Loco-Manipulation,
  [arXiv:2511.15200](https://arxiv.org/abs/2511.15200).
- Opening the Sim-to-Real Door (pixel-to-action),
  [arXiv:2512.01061](https://arxiv.org/abs/2512.01061).
- Sim-to-Real Humanoid Locomotion in 15 Minutes,
  [arXiv:2512.01996](https://arxiv.org/abs/2512.01996).
- GR00T N1 (Isaac humanoid foundation model),
  [arXiv:2503.14734](https://arxiv.org/abs/2503.14734).
- Cosmos World Foundation Model Platform,
  [arXiv:2501.03575](https://arxiv.org/abs/2501.03575).

**Manipulation foundations.**

- ALOHA / ALOHA 2 / Mobile ALOHA —
  [2304.13705](https://arxiv.org/abs/2304.13705),
  [2405.02292](https://arxiv.org/abs/2405.02292),
  [2401.02117](https://arxiv.org/abs/2401.02117).
- π0 / π0.5 — [2410.24164](https://arxiv.org/abs/2410.24164),
  [2504.16054](https://arxiv.org/abs/2504.16054).
- RT-2 / RT-X (Open X-Embodiment) —
  [2307.15818](https://arxiv.org/abs/2307.15818),
  [2310.08864](https://arxiv.org/abs/2310.08864).
- ManiSkill3, [arXiv:2410.00425](https://arxiv.org/abs/2410.00425).
- Robosuite, [arXiv:2009.12293](https://arxiv.org/abs/2009.12293).
- Meta-World, [arXiv:1910.10897](https://arxiv.org/abs/1910.10897);
  Meta-World+, [arXiv:2505.11289](https://arxiv.org/abs/2505.11289).
- DM Control Suite, [arXiv:1801.00690](https://arxiv.org/abs/1801.00690).

**Docs and tutorials.**

- Official MuJoCo docs — [mujoco.readthedocs.io](https://mujoco.readthedocs.io/).
- MJX page — [mjx.html](https://mujoco.readthedocs.io/en/stable/mjx.html).
- MJX tutorial notebook —
  [github.com/google-deepmind/mujoco/blob/main/mjx/tutorial.ipynb](https://github.com/google-deepmind/mujoco/blob/main/mjx/tutorial.ipynb).
- MuJoCo Playground site — [playground.mujoco.org](https://playground.mujoco.org/).
- MuJoCo Menagerie repo —
  [github.com/google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie).
- Isaac Lab docs — [isaac-sim.github.io/IsaacLab](https://isaac-sim.github.io/IsaacLab/).
- Isaac Sim 6.0 docs —
  [docs.isaacsim.omniverse.nvidia.com/6.0.0](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/).
- Isaac Lab performance benchmarks —
  [performance_benchmarks.html](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/performance_benchmarks.html).
- NVIDIA's "Getting Started with Isaac Lab" course —
  [docs.nvidia.com/learning/physical-ai/getting-started-with-isaac-lab](https://docs.nvidia.com/learning/physical-ai/getting-started-with-isaac-lab/).
- Isaac Gym deprecation notice —
  [forums.developer.nvidia.com/t/322978](https://forums.developer.nvidia.com/t/isaac-gym-deprecation-transition-to-isaac-lab/322978).

**Blogs worth reading.**

- DeepMind, "Opening up a physics simulator for robotics" (May 2022) —
  [deepmind.google/blog/open-sourcing-mujoco](https://deepmind.google/blog/open-sourcing-mujoco/).
- NVIDIA Developer, "Announcing Newton, an Open-Source Physics Engine
  for Robotics Simulation" (2025) —
  [Newton announcement](https://developer.nvidia.com/blog/announcing-newton-an-open-source-physics-engine-for-robotics-simulation/).
- NVIDIA Developer, "R²D² — Scaling Multimodal Robot Learning with
  NVIDIA Isaac Lab" —
  [r2d2-isaac-lab](https://developer.nvidia.com/blog/r2d2-scaling-multimodal-robot-learning-with-nvidia-isaac-lab/).
- Figure, "Natural Humanoid Walk Using Reinforcement Learning" —
  [figure.ai/news/reinforcement-learning-walking](https://www.figure.ai/news/reinforcement-learning-walking).
- Figure, "Helix 02 / System 0" —
  [figure.ai/news/helix-02](https://www.figure.ai/news/helix-02).
- Physical Intelligence, "π0 blog" —
  [pi.website/blog/pi0](https://www.pi.website/blog/pi0).
- Lilian Weng, "Reward Hacking in Reinforcement Learning" —
  [lilianweng.github.io/posts/2024-11-28-reward-hacking](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/).

**Videos / talks.**

- Pieter Abbeel — *Foundations of Deep RL* lecture series (UC Berkeley).
- Sergey Levine — *CS 285: Deep Reinforcement Learning* (UC Berkeley),
  the canonical robotics-RL graduate course.
- Russ Tedrake — *Robotic Manipulation* (MIT 6.4210), the canonical
  manipulation course (Drake-based but conceptually broad).
- Marco Hutter — *ANYmal* talks on YouTube; the practical sim-to-real
  story straight from the source.
- Jemin Hwangbo — *Sim-to-Real RL for Legged Robots* (Science Robotics
  follow-ups).
- Robohub / NVIDIA GTC keynotes — the GR00T / Isaac Lab announcements
  from 2025 (free, indexed on YouTube).
- TalkRL podcast — Hafner on Dreamer, Hwangbo on legged sim-to-real.

**Code repositories worth cloning.**

- [`google-deepmind/mujoco`](https://github.com/google-deepmind/mujoco) — the
  engine itself; includes MJX under `mjx/`.
- [`google-deepmind/mujoco_playground`](https://github.com/google-deepmind/mujoco_playground)
  — the 2025 env suite.
- [`google-deepmind/mujoco_menagerie`](https://github.com/google-deepmind/mujoco_menagerie)
  — calibrated robot models.
- [`isaac-sim/IsaacLab`](https://github.com/isaac-sim/IsaacLab) — the RL
  framework.
- [`leggedrobotics/legged_gym`](https://github.com/leggedrobotics/legged_gym)
  — the canonical open community recipe for quadruped/humanoid PPO on
  Isaac Gym (now being ported to Isaac Lab).
- [`unitreerobotics/unitree_rl_lab`](https://github.com/unitreerobotics/unitree_rl_lab)
  — Unitree's official open recipe.
- [`BoosterRobotics/booster_gym`](https://github.com/BoosterRobotics/booster_gym)
  — Booster T1 humanoid recipe (paper above).
- [`google/brax`](https://github.com/google/brax) — JAX RL engine, now
  MJX-backed.
- [`NVIDIA/Cosmos`](https://github.com/NVIDIA/Cosmos) — world foundation
  model platform.

### 33. Key Takeaways

1. **Robotics RL lives in sim because the real world is 100–1000× too
   slow.** A single GPU runs 4096 parallel humanoids; a real lab runs
   one robot, gingerly.
2. **Two simulators dominate.** MuJoCo (DeepMind, smooth-contact,
   JAX-friendly via MJX) and Isaac Sim + Isaac Lab (NVIDIA, PhysX 5,
   photoreal, PyTorch). They are converging via Newton.
3. **Isaac Gym is dead.** Anything you read pre-2024 saying "use Isaac
   Gym" should be translated to "use Isaac Lab" — NVIDIA deprecated
   Isaac Gym on **Feb 7, 2025**.
4. **The reality gap is real and the answer is "both."** Close the gap
   with system ID, real-to-sim, and actuator networks; transfer across
   it with DR, RMA, and teacher–student distillation. Every modern
   paper uses both.
5. **PPO won robotics RL.** Almost every legged-locomotion sim-to-real
   paper from 2021 to 2026 uses PPO over thousands of parallel envs.
   SAC wins in sample-budget-limited corners; model-based wins when
   there's no sim at all (DayDreamer).
6. **Domain randomization is the cheap lever.** A wide enough sim
   distribution + a memory-equipped policy = implicit system ID at
   deploy time. ADR (the curriculum version) is the production
   default.
7. **RMA / teacher–student is the dominant legged-locomotion pattern.**
   Privileged teacher trains in sim with full info; deployable student
   gets only proprioception via supervised distillation. Read RMA
   (arXiv:2107.04034), Hwangbo 2019, and Lee 2020 — those three papers
   are 80% of the recipe.
8. **Quadrupeds are solved-ish. Humanoids are the 2025–2026 story.**
   Berkeley Humanoid, ASAP on G1, Booster Gym, the "15 minute"
   recipe — DR + Isaac Lab (or MJX Playground) + PPO is now reaching
   convincing humanoid demos on consumer hardware.
9. **Manipulation flipped to imitation learning.** ALOHA, RT-X, π0 —
   teleop + diffusion VLAs dominate where sim-RL once would have.
   ManiSkill3 keeps the sim path alive, but most public 2025
   manipulation demos collected real teleop.
10. **Foundation models are entering the loop.** GR00T (Isaac) + Cosmos
    synthetic data + Isaac Lab fine-tuning is NVIDIA's new shape.
    Think of PPO not as the whole training run but as the *fine-tuner*
    on top of a pretrained VLA — exactly the LLM post-training pattern
    from Week 5.
11. **Reward source has moved from "human types a number" toward
    "physics enforces it."** The arc Week 5 (RLHF) → Week 6 (RLVR) →
    Week 7 (RWML) → Week 8 (physical sim) is one continuous
    progression toward verifiers the policy cannot game.
12. **You can do this on a single 4090.** "Humanoid in 15 minutes,"
    MJX Playground recipes, the Isaac Lab quadruped configs — none
    require a cluster. The most accessible the field has ever been.

---

*Week 8 wraps the RL 101 sequence by closing the loop back to physical
robots — the place RL started before it pivoted to games and then LLMs.
Whatever you build next — locomotion, manipulation, agent RL, RLHF — the
core ideas (Bellman, actor–critic, advantage estimation, KL constraints,
verifier-based rewards) carry over. The simulators just give us cheap,
fast, reproducible playgrounds in which to use them.*

*Ready to press play? Continue to [`docs/week8-demos.md`](week8-demos.md)
for the runnable companion — three demos, three Unitree robots, one
RTX 5090, ninety minutes.*

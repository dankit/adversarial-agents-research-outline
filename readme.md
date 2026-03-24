## Note that this is all still a work in progress. Things will change as I go along making this. I am currently facing a major blocker with gpu availability on cloud services, so this project is going slower than anticipated.

## Project Overview & Motivation

This project maps the threat landscape of AI-enabled manipulation across major social media platforms. It studies coordinated inauthentic behavior, including bot-driven narratives and mass upvote/downvote campaigns that exploit platform mechanics and human cognitive biases at scale.

The work is motivated by persistent disinformation campaigns across domains such as politics, advertising, reputation repair for prominent figures, state-sponsored propaganda, and meme stock/coin hype cycles. The goal is to better understand current offensive capabilities, evaluate the gap between offensive and defensive capacity, inform better decision-making, and contribute practical insight to AI safety while pushing technical depth through hands-on research.

## Starting premise

Web signup and onboarding flows are among the most adversarial interactive environments available today: they combine non-stationary page layouts, multi-step verification gates, invisible behavioral scoring, and sophisticated fingerprinting — all designed to distinguish humans from automated agents. This makes them a compelling testbed for training robust, generalizable computer-use agents via reinforcement learning.

This project trains an LLM — Qwen 3.5-35B-A3B (MoE, 3B active parameters) — to complete real web tasks by learning a policy that maps page observations to browser actions. The model is served via vLLM on a remote cloud GPU (accessed locally through an SSH tunnel to `localhost:8000`), and fine-tuned on the same hardware via LoRA. A stronger same-family teacher model (Qwen3.5-397B-A17B) generates seed demonstrations, and an LLM-as-judge (most likely going to be gpt or claude) evaluation system scores trajectory quality before SFT to ensure data quality. The agent improves through a **teacher demonstrations → supervised fine-tuning → GRPO reinforcement learning** pipeline, with reward signals derived from task completion, intermediate progress, and behavioral realism.

### What makes this technically interesting

- **Sparse, delayed rewards in a long-horizon task.** A signup flow is 10–30 steps. The terminal reward (account created) is binary and only observed at the end. Intermediate shaping signals must be designed carefully to avoid reward hacking.
- **Adversarial, non-stationary environment.** Detection systems retrain continuously; page layouts change; challenge frequency is conditioned on the agent's own behavioral history within a session. The environment is not an offline dataset — it fights back.
- **Real-world grounding with no simulator.** There is no reset-to-identical-state. Each episode runs against a live environment with server-side state, rate limits, and IP reputation. Exploration is expensive.
- **Cross-signal consistency constraints.** The agent operates within a high-dimensional identity envelope (browser fingerprint, network origin, behavioral cadence, session history) where any single inconsistency across signals can invalidate the entire episode.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     RL Training Loop                      │
│                                                           │
│   ┌───────────┐    ┌────────────┐    ┌────────────────┐  │
│   │  Policy    │───▶│  Agent     │───▶│  Browser       │  │
│   │  (Qwen    │    │  Loop      │    │  Environment   │  │
│   │   vLLM)   │    │            │    │  (Playwright)  │  │
│   └───────────┘    └────────────┘    └────────────────┘  │
│        ▲                 │                  │             │
│        │                 ▼                  ▼             │
│   ┌───────────┐    ┌────────────┐    ┌────────────────┐  │
│   │  RL       │◀───│  Reward    │◀───│  Task          │  │
│   │  Trainer  │    │  Signal    │    │  Verifier      │  │
│   │  (GRPO)   │    │            │    │                │  │
│   └───────────┘    └────────────┘    └────────────────┘  │
│                                             │             │
│                                      ┌──────┴─────────┐  │
│                                      │ Verification   │  │
│                                      │ Oracle         │  │
│                                      │ (email / SMS)  │  │
│                                      └────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

Three pillars:

1. **Browser environment** (`browser_env/environment.py`) — Playwright with real Chromium, exposing a Gym-style `reset()`/`step()` interface. Creates a fresh browser context per episode (isolating cookies, storage, and state). The agent receives a DOM-text observation (interactive elements prioritized: form inputs > buttons > links, capped at 40 elements) and emits a discrete action.
2. **Agent loop** (`agent/agent.py`) — Observe → prompt LLM → parse structured JSON action → execute in browser → collect (observation, action, reward) tuple. Includes universal recovery logic (stall detection, overlay escape, captcha back-off) but deliberately contains zero site-specific heuristics — all site knowledge lives in the task's `objective` and `form_data`.
3. **RL trainer** — Collects N full trajectory rollouts, scores each with the shaped reward function, and applies GRPO-style group-relative advantage weighting to update the policy toward higher-reward trajectories. No critic network required (though per-step credit assignment is coarser as a result).

### Design philosophy: prompt-driven, not heuristic-driven

All site-specific knowledge lives in two places:
1. **`objective`** — natural language description the LLM reads (what to click, what to avoid, what flow to follow)
2. **`form_data`** — `dict[str, str]` injected into the system prompt so the LLM knows exactly what to type

The agent code contains only **universal browser recovery** logic:
- Stall detection (same URL + no reward for N steps) → `back()` / `goto(start)`
- Pointer interception → `Escape` / restart
- Captcha detection → `back()` / restart
- No-elements fallback → `Tab`

This means new sites need zero Python heuristic code — just a declarative task definition.

---

## Project Structure

```
root/
├── docker-compose.yml              # vLLM model server (Qwen3.5-35B-A3B)
├── .env.example                    # Environment variable template
├── requirements.txt                # Core deps: playwright, openai, python-dotenv, pydantic
├── README.md                       # Project-level docs, quick start, adding tasks
├── AGENT_QUICKREF.md               # Operational reference: architecture, action contract, how to run
├── PLAN.md                         # Full project plan: phases, RL design, bot detection, proxy/email setup
│
├── agent/                          # Agent loop and LLM integration
│   ├── agent.py                    # Episode loop: observe → think → act → log; universal recovery
│   ├── llm_client.py               # OpenAI-compatible vLLM client; think-tag stripping; tool call support
│   └── prompts.py                  # build_system_prompt(objective, form_data) with observation format docs
│
├── browser_env/                    # Playwright browser environment
│   ├── environment.py              # Gym-like reset/step API; observation extraction; action execution
│   └── reward.py                   # Milestone-based reward shaping with per-field detection via form_data keys
│
├── tasks/                          # Declarative task definitions
│   ├── base.py                     # WebTask base class (objective, form_data, success signals, captcha keywords)
│   └── example_signup.py           # example signup task for website A (inherits WebTask)
│
├── training/                       # Trajectory collection and training pipeline
│   ├── collect.py                  # Multi-episode trajectory collector with task registry
│   ├── prepare_sft.py              # Filter heuristic actions, build SFT train/val splits from trajectories
│   ├── sft.py                      # LoRA fine-tuning with TRL SFTTrainer
│   └── replay_artifacts.py         # Locate/open latest screenshots, videos, traces
│
├── evals/                          # Two-layer trajectory evaluation system
│   ├── patterns.py                 # 12 deterministic failure pattern detectors (instant, zero LLM cost)
│   ├── criteria.py                 # 10 LLM-as-judge criteria with 1-5 rubrics
│   ├── judge.py                    # LLM-as-judge trajectory scoring (summarizes + prompts judge model)
│   └── run.py                      # CLI: python -m evals.run (patterns-only or full judge)
│
├── trajectories/                   # Collected episode data (generated at runtime)
│   ├── summary.csv                 # Episode-level metrics (reward, success, steps)
│   └── <episode_id>.jsonl          # Per-episode transitions with task metadata header
│
└── artifacts/                      # Runtime artifacts (generated at runtime)
    ├── screenshots/                # Per-step screenshots
    ├── videos/                     # Playwright session videos
    └── traces/                     # Per-episode Playwright traces (.zip)
```

## Observation and Action Design

### Observation strategy

| Approach | Observation format | Model requirement | Tradeoff |
|----------|-------------------|-------------------|----------|
| **DOM / accessibility tree** (Phase 1–3) | Structured text: element roles, labels, IDs, input states | Text-only LLM (Qwen 3.5-35B-A3B) | Smaller observation space, faster RL iterations. More brittle across sites with different DOM structures. |
| **Vision / screenshot** (Phase 4+) | Raw pixel image of the viewport | Vision-language model | Generalizes better (buttons look like buttons). Handles canvas/WebGL UIs and obfuscated DOM. Higher compute cost per step. |


### Observation format (implemented)

```
URL: https://www.example.com/register
TITLE: Create Account
ELEMENTS:
- id=1 role=input type=email label=Email address filled=false enabled=true
- id=2 role=input type=password label=Password filled=false enabled=true
- id=3 role=button label=Sign Up filled=false enabled=true
- id=4 role=a label=Already have an account? filled=false enabled=true
```

Elements are extracted via JavaScript, prioritized (form inputs first, then buttons, then links), and capped at 40 per observation. Each element is tagged with a stable `data-wa-oid` attribute on the page so the agent's `id=N` targets resolve deterministically to the correct DOM element.

### Action space (implemented)

```python
ACTIONS = [
    "click(target)",        # target is id=N from observation
    "type(target, text)",   # target is id=N, types with human-like keystroke delays (60-180ms)
    "press(key)",           # Enter, Tab, Escape, ArrowDown, ArrowUp
    "wait(seconds)",        # max 8 seconds
    "goto(url)",            # navigate to URL
    "back()",               # browser back navigation
]
```

### Planned additions (Phase 2+)

```python
FUTURE_ACTIONS = [
    "click(x, y)",          # vision-mode: click by pixel coordinate (Phase 4+)
    "scroll(direction)",    # up | down
    "request_email_code",   # signals the Verification Oracle
    "request_sms_code",     # signals the Verification Oracle
]
```
Actions are emitted as structured JSON by the LLM and parsed deterministically, so structural integrity is important.
---

## Reward Design

Reward shaping is critical because the terminal signal (account created or not) is too sparse for sample-efficient learning. The reward function combines progress milestones, completion, and behavioral quality:

| Signal | Reward | Detection method |
|--------|--------|------------------|
| Reached signup form | +0.3 | URL pattern or DOM contains signup-specific elements |
| Filled a field correctly | +0.1 per field | DOM state inspection after typing (matched against `form_data` keys) |
| Advanced to next step in flow | +0.1 | Step transition detected via URL or DOM change |
| Completed verification | +0.15 | Verification input accepted |
| **Account created successfully** | **+1.0** | Success page URL, confirmation text, or API check |
| Timed out / max steps exceeded | 0.0 | Episode truncation |
| Action executed too fast (< 500ms gap) | −0.1 | Timing check in environment wrapper |
| Efficiency bonus | +0.2 × (1 − steps/max_steps) | Fewer steps = higher bonus |

**Implementation note:** The reward tracker (`browser_env/reward.py`) currently implements the milestone-based subset: target page detection (+0.3), per-field fill detection via `form_data` keys (+0.1 each), CAPTCHA penalty (−0.05), and success bonus (+1.0). These are currently brittle hard coded rules, and break easily on different webpages. I am looking towards migrating to RLAIF for better critera generalization across domains, also incorporating RLVR, because of the account creation process being a verifiable reward.

**Design rationale:** Negative rewards for bot-like micro-behaviors incentivize the policy to learn human-plausible interaction patterns — not as hardcoded heuristics, but as emergent behavior shaped by the reward landscape. For instance, The agent should implicitly discover that browsing before signing up reduces CAPTCHA frequency through trial-and-error RL.

---

## Training Pipeline

### Training strategy

The pipeline has three phases. The current approach uses **same-family distillation** for demonstration collection:

**Teacher model selection:** Same-family distillation works best — the teacher and student share the same tokenizer and reasoning style, so demonstrations transfer more cleanly.

| Teacher | Active params | VRAM (Int4) | Notes |
|---|---|---|---|
| **Qwen3.5-122B-A10B** | 10B | fits on 1x H100 80 GB or 1x GH200 (96 GB HBM3) | Best practical choice: strong enough for quality demos, serveable on a single GPU, concerns around context length |
| **Qwen3.5-397B-A17B** | 17B | 100GB+ — needs multiple GPUs | Stronger but much harder to serve; diminishing returns for behavioral cloning |

~~Prefer **Qwen3.5-122B-A10B** — it's the sweet spot between demonstration quality and serving cost. The student only sees actions, not logits, so the marginal quality gain from 397B rarely justifies 3x the hardware.~~

Due to limited gpu capability, I have switched to a managed service for hosting qwen 3.5 397B A17B on mulerouter/alibaba.


### Phase 1 — Demonstration collection (implemented)

Collect trajectories using a teacher model. The model receives the DOM observation, task objective, form data, and recent action history via a system prompt describing the action format. Trajectories (successful and failed) are logged as JSONL files with per-step (observation, action, reward) transitions and a metadata header containing the objective and form data.

The trajectory collector (`training/collect.py`) supports:
- Multi-episode batch collection with configurable step limits
- Watch mode (headed browser with slow-motion for debugging)
- Per-step screenshots, video recording, and Playwright trace capture
- Summary CSV with episode-level metrics (reward, success, steps)
- Task registry for easy extension to new sites

**Purpose:** Establish baseline performance, collect seed trajectories for SFT, and iterate on prompt engineering + environment instrumentation.

### Phase 2 — Supervised fine-tuning (SFT) (implemented)

Filter Phase 1 trajectories to successful completions. The SFT preparation script (`training/prepare_sft.py`) reads task metadata (objective, form_data) from each trajectory JSONL so SFT prompts include the same form data the agent saw during collection. It filters out heuristic/recovery actions by default (training only on genuine model decisions), then formats as multi-turn conversations and splits into train/val sets.

Fine-tune **Qwen 3.5-35B-A3B** with LoRA on the remote GPU using `trl.SFTTrainer`:

SFT provides a warm-start policy that succeeds more often than prompting alone, producing richer training signal for RL.

**SFT data quality controls:**
- Heuristic/recovery actions filtered out by default (only clean model decisions)
- Parse-error fallback actions excluded
- Train/val split with configurable ratio
- Per-step metadata preserved for analysis

### Phase 3 — RL fine-tuning (trajectory-level GRPO)

Browser interaction is multi-step: the agent generates one action per turn, observes the result, and generates the next action conditioned on the updated DOM. This is fundamentally different from single-turn text generation, and standard `trl.GRPOTrainer` does not natively support interleaved environment interaction.

**Approach: trajectory-level rollout collection with GRPO-style updates.**

1. Roll out N complete episodes (each a multi-step observe → act → observe loop) using the current policy.
2. Score each trajectory with the full reward function (progress + completion + behavioral penalties).
3. Rank trajectories by total reward. Compute per-trajectory advantages using group-relative normalization (the GRPO objective).
4. For each (observation, action) pair in the training batch, weight the policy gradient by the trajectory-level advantage.

```python
def collect_trajectory(policy, env, task):
    """Single multi-step rollout: observe → act → observe → ... → done."""
    obs = env.reset(task.start_url)
    trajectory = []
    total_reward = 0.0
    for _ in range(task.max_steps):
        action = policy.generate_action(obs)
        next_obs, step_reward, done, info = env.step(action)
        trajectory.append((obs, action, step_reward))
        total_reward += step_reward
        obs = next_obs
        if done:
            break
    return trajectory, total_reward

def grpo_training_step(policy, env, task, N=8):
    """Collect N trajectories, rank, compute GRPO advantages."""
    trajectories = []
    rewards = []
    for _ in range(N):
        traj, reward = collect_trajectory(policy, env, task)
        trajectories.append(traj)
        rewards.append(reward)

    # Group-relative advantage: normalize within the batch
    mean_r, std_r = np.mean(rewards), np.std(rewards) + 1e-8
    advantages = [(r - mean_r) / std_r for r in rewards]

    # Build training batch: every (obs, action) pair weighted by trajectory advantage
    train_batch = []
    for traj, adv in zip(trajectories, advantages):
        for obs, action, _ in traj:
            train_batch.append((obs, action, adv))

    policy.update(train_batch)  # weighted policy gradient step
```

**Why trajectory-level GRPO instead of PPO:** PPO's per-step value function provides finer-grained credit assignment, which is generally preferable for multi-step tasks. However, it requires fitting a critic network — for a 35B MoE model, even a smaller critic adds significant VRAM pressure alongside the policy weights, LoRA optimizer states, and KV cache on a single GPU. Trajectory-level GRPO avoids the critic entirely. The shaped reward function (intermediate progress signals at each step) partially compensates for coarser credit assignment: even though the trajectory-level advantage is shared across all steps, each (observation, action) training pair individually carries step-level signal about what worked. PPO with a learned critic remains a potential upgrade if multi-GPU training becomes a necessity and if I am not prohibited by pricing.

**Why not vanilla `trl.GRPOTrainer`:** The standard API assumes a static prompt with N sampled completions. In an interactive environment, the "prompt" (observation) changes after every action, so rollouts must be collected externally. `veRL` (ByteDance) and `OpenRLHF` are designed for LLM RL with environment interaction and are candidates for replacing the custom training loop as the project matures.

**Exploration and sample efficiency:** Each episode consumes real resources (proxy bandwidth, IP reputation). To manage exploration cost: (a) the SFT warm-start ensures the initial RL policy succeeds at a non-trivial rate, avoiding the cold-start problem of all-failures-no-signal; (b) N is set high enough (8–16 trajectories per batch) that at least some trajectories succeed, giving a ranking signal; (c) a trajectory replay buffer retains high-reward trajectories from prior batches for off-policy augmentation; (d) curriculum ordering starts with simplified task variants (pre-navigated to the signup form) before full end-to-end episodes.

### Phase 4 — Cross-task generalization

Train on multiple platforms with random task sampling per episode. Evaluate on held-out sites the model has never seen. The hypothesis: a policy trained on diverse adversarial signup flows should generalize to novel flows better than one memorizing a single site's DOM structure.

---

## Quick Start

```powershell
# 1. Install (local machine)
python -m venv .venv && .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt && playwright install chromium

# 2. Configure
copy .env.example .env   # edit HF_TOKEN, TENSOR_PARALLEL_SIZE, proxy settings

# 3. SSH tunnel to remote model server (separate terminal)
ssh -L 8000:localhost:8000 ubuntu@<REMOTE_IP>
# If port 8000 is busy locally, use 8001 and set OPENAI_BASE_URL=http://localhost:8001/v1

# 4. Start model server (on remote GPU machine)
docker compose up -d
docker compose logs -f qwen   # wait for "Uvicorn running on ..."

# 5. Collect trajectories (local machine, browser runs locally)
python -m training.collect --task website_signup --episodes 5 --max-steps 40
python -m training.collect --task website_signup --episodes 1 --watch --verbose-actions  # visible browser

# 6. Evaluate collected trajectories (instant, no LLM needed)
python -m evals.run --patterns-only

# 7. Full eval with LLM-as-judge
python -m evals.run --output-json evals_report.json

# 8. Prepare SFT data (reads objective + form_data from trajectory metadata)
python -m training.prepare_sft --success-only

# 9. Fine-tune (on remote GPU)
python -m training.sft --dataset datasets/sft_turns_train.jsonl --output-dir checkpoints/sft
```

Outputs: `trajectories/*.jsonl`, `trajectories/summary.csv`, `artifacts/{screenshots,videos,traces}/`, `datasets/sft_turns_{train,val}.jsonl`.

---

## Infrastructure

- **Model serving (remote via SSH tunnel):** Qwen 3.5-35B-A3B served via vLLM in Docker on a remote cloud GPU. The local machine connects via SSH tunnel (`ssh -L 8000:localhost:8000 ubuntu@<REMOTE_IP>`), so the OpenAI-compatible API appears at `localhost:8000` locally. The model runs in bfloat16 — no quantization at inference time. Configurable via `.env` (model, dtype, GPU utilization, tensor parallelism, max model length).
- **Training compute:** Same remote GPU (NVIDIA GH200 96 GB HBM3 or H100 80 GB). LoRA fine-tuning of the 35B MoE model — training fits on a single GPU. Rollout collection and training alternate on the same GPU.
Potential concern: MoE sensitivity to LoRA/finetuning
- **Browser automation:** Playwright + Chromium with WebRTC leak prevention (`--force-webrtc-ip-handling-policy=disable_non_proxied_udp`), residential proxy with sticky sessions (configurable via env vars). Creates fresh browser context per episode for state isolation. Runs locally — does not require GPU.
- **Evaluation:** Two-layer system — deterministic failure pattern detectors (instant, zero cost) catch known failure modes like action loops, wait cascades, and observation-unchanged stalls; LLM-as-judge scores trajectories against 10 weighted criteria (goal achievement, efficiency, action quality, resilience, comprehension).
- **Training stack:** `trl` (SFTTrainer), `unsloth` (for peft/LoRA), `transformers`, `accelerate`; custom trajectory-level GRPO training loop (candidates for replacement: `veRL`, `OpenRLHF`).
- **Core dependencies:** `playwright`, `openai` (Python client), `python-dotenv`, `pydantic`. Phase 2 deps (trl, transformers, etc.) installed separately.
- **Cost estimate:** GH200 on Lambda ≈ $2–3/hr. SFT (a few hours) + RL (tens of hours over multiple sessions) ≈ $50–150 total training compute.

---

## Current Status

Phase 1 (demonstration collection) is **implemented and operational**:
- The vLLM serving infrastructure runs via Docker Compose on a remote cloud GPU, accessed locally via SSH tunnel.
- The browser environment wrapper exposes a Gym-style `reset()`/`step()` API with DOM-text observations.
- The agent loop collects trajectories with universal recovery logic and no site-specific heuristics.
- The declarative task system supports a simple web signup; new platforms are added by subclassing `WebTask`.
- Trajectory collection produces JSONL files with task metadata headers and a summary CSV.
- Platform-specific research (detection stacks, DOM structures, verification flows) is complete.

**Trajectory evaluation is implemented:**
- 12 deterministic failure pattern detectors (action loops, wait cascades, parse error spikes, observation-unchanged stalls, etc.) for instant, zero-cost trajectory triage.
- 10-criterion LLM-as-judge scoring system with weighted 1-5 rubrics covering task completion, efficiency, action quality, resilience, and comprehension.
- CLI: `python -m evals.run` supports `--patterns-only`, `--failure-only`, `--judge-model`, `--list-patterns`, with JSON/CSV export.
- Cursor rules ensure future agent sessions automatically know the eval conventions and the workflow for adding new patterns when debugging failures.

Phase 2 (SFT) tooling is **implemented**:
- SFT data preparation filters heuristic actions, reads task metadata from trajectories, and builds train/val splits.
- LoRA fine-tuning script uses TRL SFTTrainer with configurable LoRA rank/alpha.

**Immediate next steps:** Collect a sufficient corpus of successful trajectories for SFT warm-start using the 397B model, run SFT on the remote GPU, and begin iterating toward the GRPO training phase.


Misc notes:
Human-like behavior has been implemented through things like bezier curve mouse movement, jitters, variance in typing speed, and more.
Added real-time observability for each computer use agent, able to see tokens streamed in real time, task, history, etc.
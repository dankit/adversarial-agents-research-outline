## Project Overview & Motivation

The work is motivated by persistent disinformation campaigns via bot farms across domains such as politics, advertising, reputational repair, state-sponsored propaganda, and meme stock/coin hype cycles. The goal is to better understand current offensive capabilities, evaluate the gap between offensive and defensive capacity, inform better decision-making, and contribute practical insight to AI safety while pushing technical depth through hands-on research.

The main purpose of the project is NOT to use it for adversarial purposes. Once main functionalities are in place, it can be extended for other web-based tasks.

## Starting premise

Web signup and onboarding flows are among the most dynamic interactive environments available today: they combine non-stationary page layouts, multi-step verification gates, invisible behavioral scoring, and sophisticated fingerprinting — all designed to distinguish humans from bots. This makes them a compelling testbed for training robust, generalizable browser-use agents via reinforcement learning to mimic real human interactions.

This project trains a VLM — Qwen 3.5-35B-A3B (MoE, 3B active parameters) — to complete real web tasks by learning a policy that maps page observations to browser actions. The model is served via vLLM on a remote cloud GPU (accessed locally through an SSH tunnel to `localhost:8000`), and fine-tuned on the same hardware via LoRA. A stronger same-family teacher model (Qwen3.5-397B-A17B) generates synthetic reasoning traces alongside manual trajectory gathering within the browser harness. The agent improves through a **human/teacher demonstrations → supervised fine-tuning → GRPO reinforcement learning** pipeline, with reward signals derived from task completion, intermediate progress, and behavioral realism.

### What makes this technically interesting

- **Sparse, delayed rewards in a long-horizon task.** A signup flow is 10-60 steps. The terminal reward (account created) is binary and only observed at the end. Intermediate shaping signals must be designed carefully to avoid reward hacking.
- **Adversarial, non-stationary environment.** Detection systems retrain continuously; page layouts change; challenge frequency is conditioned on the agent's own behavioral history within a session. The environment is not an offline dataset — it fights back.
- **Real-world grounding with no simulator.** There is no reset-to-identical-state. Each episode runs against a live environment with server-side state, rate limits, and IP reputation. Exploration is expensive.
- **Cross-signal consistency constraints.** The agent operates within a high-dimensional identity envelope (browser fingerprint, network origin, behavioral cadence, session history) where any single inconsistency across signals can invalidate the entire episode.
- **Captcha solving.** VLMs struggle with captchas, and depending on the captcha provider have a near 0% success rate on "hard" captchas. This project aims to see if we can train captcha solving as well.
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

1. **Browser environment** (`browser_env/environment.py`) — Playwright with real Chromium, exposing a Gym-style `reset()`/`step()` interface. Creates a fresh browser context per episode (isolating cookies, storage, and state). The agent receives a visual observation and emits a discrete harness-specific action.
2. **Agent loop** (`agent/agent.py`) — Observe → prompt LLM → parse structured JSON action → execute in browser → collect (observation, action, reward) tuple. Includes universal recovery logic (stall detection, overlay escape, captcha back-off) but deliberately contains zero site-specific heuristics — all site knowledge lives in the task's `objective` and `form_data`.
3. **RL trainer** — Collects N full trajectory rollouts, scores each with the shaped reward function, and applies GRPO-style group-relative advantage weighting to update the policy toward higher-reward trajectories. No critic network required (though per-step credit assignment is coarser as a result).

### Design philosophy: prompt-driven, not heuristic-driven

All site-specific knowledge lives in two places:
1. **`objective`** — natural language description the LLM reads (what to click, what to avoid, what flow to follow)
2. **`form_data`** — `dict[str, str]` injected into the system prompt so the LLM knows exactly what to type

The agent code contains only **universal browser recovery** logic:
- Stall detection (same URL + no reward for N steps) → `back()` / `goto(start)`
- Pointer interception → `Escape` / restart
- No-elements fallback → `Tab`

This means new sites need zero Python heuristic code — just a declarative task definition.

---

## Observation and Action Design

### Action space (implemented)

```python
ACTIONS = [
    "type(text)",           # types with human-like keystroke delays (60-180ms)
    "press(key)",           # Enter, Tab, Escape, ArrowDown, ArrowUp
    "wait(seconds)",        # max 8 seconds
    "goto(url)",            # navigate to URL
    "back()",               # browser back navigation
    "click(x, y)",          # click by pixel coordinate
    "scroll(direction)",    # up | down
    "request_email_code",   # signals the Verification Oracle
    "request_sms_code",     # signals the Verification Oracle
    "exit(boolean, reason)" # signal to end the episode, whether signup was successful, or failed (e.g. ip is blocked)
    ]
```

Actions are emitted as structured JSON by the LLM and parsed deterministically, so structural integrity is important.
---

## Reward Design

Reward shaping is critical because the terminal signal (account created or not) is too sparse for sample-efficient learning. The reward function combines progress milestones, completion, and behavioral quality:

| Signal | Reward | Detection method |
|--------|--------|------------------|
| Reached signup form | +0.3 | URL pattern or page contains signup-specific elements |
| Filled a field correctly | +0.1 per field | Page state inspection after typing (matched against `form_data` keys) |
| Completed verification | +0.15 | Verification input accepted |
| **Account created successfully** | **+1.0** | Success page URL, confirmation text, or API check |
| Timed out / max steps exceeded | 0.0 | Episode truncation |
| Efficiency bonus | +0.2 × (1 − steps/max_steps) | Fewer steps = higher bonus |
| Captcha triggered | -0.25 | visual trigger |

**Implementation note:** These are currently brittle hard coded rules, and break easily on different webpages. I am looking towards migrating to LLM-as-judge for the reward model for better critera generalization across domains, this is on top of RLVR, because of the account creation process being a verifiable reward.

R_total = R_success (verifiable reward) + ε·R_LLM , where ε is a small value, so that we heavily depend on the verifiable reward and the LLM judge is not the primary reward.

**Design rationale:** Negative rewards for bot-like micro-behaviors incentivize the policy to learn human-plausible interaction patterns — not as hardcoded heuristics, but as emergent behavior shaped by the reward landscape. For instance, The agent should implicitly discover that browsing before signing up reduces CAPTCHA frequency through trial-and-error RL. It should also learn how to solve captchas, in which it has a near 0% success rate on harder captchas.

RL concerns:

-Clicking back/forth between forms may add superfluous reward signal. Solution: Keep track of forms with a set for idempotency.

-Model learns the judge's preferences, not the objective at hand

-Too strict of a criteria may prevent the model from exploring

---

## Training Pipeline

### Training strategy

The pipeline has three phases. The current approach uses **same-family distillation** for synthetic data collection:

**Teacher model selection:** Same-family distillation works best — the teacher and student share the same tokenizer and reasoning style, so demonstrations transfer more cleanly.

| Teacher | Active params | VRAM (Int4) | Notes |
|---|---|---|---|
| **Qwen3.5-122B-A10B** | 10B | fits on 1x H100 80 GB or 1x GH200 (96 GB HBM3) | Best practical choice: strong enough for quality demos, serveable on a single GPU, concerns around context length |
| **Qwen3.5-397B-A17B** | 17B | 100GB+ — needs multiple GPUs | Stronger but much harder to serve; diminishing returns for behavioral cloning |

~~Prefer **Qwen3.5-122B-A10B** — it's the sweet spot between demonstration quality and serving cost. The student only sees actions, not logits, so the marginal quality gain from 397B rarely justifies 3x the hardware.~~

Due to limited gpu capability, I have switched to a managed service for hosting qwen 3.5 397B A17B (non-quantized).


### Phase 1 — Demonstration collection (implemented)

Create the base browser harness in which the agent will interact with during RL, and normal inference runs. This same harness will also be used for collecting manual trajectories.

The browser harness trains on a custom self-made gym, hosting various captchas and dynamic page layouts per load to improve generalizability. This way the model does not see the forms and captcha in the same spot every run.

Main challenge: Extending the browser harness for human data gathering, and translating these manual trajectories into a format that the VLM can train on. I have created a data designer for this, where steps can be edited, inserted, or deleted. Manual trajectories do not include reasoning traces and are stubbed out by default. The teacher model will generate synthetic reasoning traces to explain each manual action taken, in which it will see each step's page observation via screenshot, the task objective, form data, and recent action history through the system/user prompt.

**Purpose:** Establish baseline performance, collect seed trajectories for SFT, and iterate on prompt engineering + environment instrumentation.

### Phase 2 — Supervised fine-tuning (SFT) (implemented)

Filter Phase 1 trajectories to successful completions. The SFT preparation script (`training/prepare_sft.py`) reads task metadata (objective, form_data) from each trajectory JSONL so SFT prompts include the same form data the agent saw during collection. It filters out heuristic/recovery actions by default (training only on genuine model decisions).

Fine-tune **Qwen 3.5-35B-A3B** with LoRA on the remote GPU using `trl.SFTTrainer`:

SFT provides a warm-start policy that succeeds more often than prompting alone, producing richer training signal for RL.

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

**Why trajectory-level GRPO instead of PPO:** PPO's per-step value function provides finer-grained credit assignment, which is generally preferable for multi-step tasks. However, it requires fitting a critic network — for a 35B MoE model, even a smaller critic adds significant VRAM pressure alongside the policy weights, LoRA optimizer states, and KV cache on a single GPU. Trajectory-level GRPO avoids the critic entirely. The shaped reward function (intermediate progress signals at each step) partially compensates for coarser credit assignment: even though the trajectory-level advantage is shared across all steps, each (observation, action) training pair individually carries step-level signal about what worked.

**Why not vanilla `trl.GRPOTrainer`:** The standard API assumes a static prompt with N sampled completions. In an interactive environment, the "prompt" (observation) changes after every action, so rollouts must be collected externally. `veRL` (ByteDance) and `OpenRLHF` are designed for LLM RL with environment interaction and are candidates for replacing the custom training loop as the project matures.

**Exploration and sample efficiency:** Each episode consumes real resources (proxy bandwidth, IP reputation). To manage exploration cost: (a) the SFT warm-start ensures the initial RL policy succeeds at a non-trivial rate, avoiding the cold-start problem of all-failures-no-signal; (b) N is set high enough (8–16 trajectories per batch) that at least some trajectories succeed, giving a ranking signal; (c) a trajectory replay buffer retains high-reward trajectories from prior batches for off-policy augmentation; (d) curriculum ordering starts with simplified task variants (pre-navigated to the signup form) before full end-to-end episodes.

### Phase 4 — Cross-task generalization

Train on multiple platforms with random task sampling per episode. Evaluate on held-out sites the model has never seen. The hypothesis: a policy trained on diverse adversarial signup flows should generalize to novel flows better than one memorizing a single site's structure.

---


## Infrastructure

- **Model serving (remote via SSH tunnel):** Qwen 3.5-35B-A3B served via vLLM in Docker on a remote cloud GPU. The local machine connects via SSH tunnel (`ssh -L 8000:localhost:8000 ubuntu@<REMOTE_IP>`), so the OpenAI-compatible API appears at `localhost:8000` locally. The model runs in bfloat16 — no quantization at inference time. Configurable via `.env` (model, dtype, GPU utilization, tensor parallelism, max model length).
- **Training compute:** Same remote GPU (NVIDIA GH200 96 GB HBM3 or H100 80 GB). LoRA fine-tuning of the 35B MoE model — training fits on a single GPU. Rollout collection and training alternate on the same GPU. Potential concern: MoE sensitivity to LoRA/finetuning
- **Browser automation:** Playwright + Chromium with stealth. Creates fresh browser context per episode for state isolation. Runs locally — does not require GPU.
- **Evaluation:** Two-layer system — deterministic failure pattern detectors (instant, zero cost) catch known failure modes like action loops, wait cascades, and observation-unchanged stalls; LLM-as-judge scores trajectories against 10 weighted criteria (goal achievement, efficiency, action quality, resilience, comprehension).
- **Training stack:** `trl` (SFTTrainer), `unsloth` (for peft/LoRA), `transformers`, `accelerate`; custom trajectory-level GRPO training loop (candidates for replacement: `veRL`, `OpenRLHF`).
- **Core dependencies:** `playwright`, `openai` (Python client), `python-dotenv`, `pydantic`. Phase 2 deps (trl, transformers, etc.) installed separately.
- **Cost estimate:** GH200 on Lambda ≈ $2–3/hr. SFT (a few hours) + RL (tens of hours over multiple sessions) ≈ $50–150 total training compute. For trajectory collection, add another $50 with the 397B model.

---


Misc notes:

Human-like behavior has been implemented through things like bezier curve mouse movement, jitters, variance in typing speed, and more.

Added real-time observability for each computer use agent, able to see tokens streamed in real time, task, history, etc.

Originally this project aimed to use the text-only mode within Qwen, but was proving to be problematic due to a few major challenges:
- Each website having it's own unique DOM structure. Meaning each task/prompt has to cater each website's DOM specifically. While this may work in practice, and could allow for generalization across different web pages, there is another challenge ontop of this: creating a high level standardization of what information should be included/excluded. Reducing noise is important to maintain signal and reduce context window size for efficiency. Originally I was filtering on input elements such as buttons and forms, but sometimes even the text elements are important for error messages such as "your ip has been blocked". Adding text elements to the DOM observations hurts model quality greatly due to the amount of noise, especially when they are dynamic content such as user posts. 
- Trying to be mindful of things like shadow-dom, deriving what elements we can interact with on the page, single page application flows which confused the LLM, are a few examples of edge case handling which was making the code base unmaintainable.
- On a few sites the elements are identical (obfuscation) making the DOM-only method less viable. Once again, text elements would help with this, but also introduce a lot of irrelevant tokens and signals that trip up the LLM.
- I was originally using the VLM mode as a tool call/subagent for solving captchas, but had suboptimal performance
- After swapping to the full VLM mode, over the text-only LLM, this now allows for consolidation of a lot of messy logic in the codebase, allows me to train captcha solving on top of web tasks, while taking a more generalizable approach.
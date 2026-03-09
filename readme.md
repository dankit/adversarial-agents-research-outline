# Note that this is all still a work in progress. A lot of things may change as I go along making this.

## Project Overview & Motivation

This project maps the threat landscape of AI-enabled manipulation across major social media platforms. It studies coordinated inauthentic behavior, including bot-driven narratives and mass upvote/downvote campaigns that exploit platform mechanics and human cognitive biases at scale.

The work is motivated by persistent disinformation campaigns across domains such as politics, advertising, reputation repair for prominent figures, state-sponsored propaganda, and meme stock/coin hype cycles. The goal is to better understand current offensive capabilities, evaluate the gap between offensive and defensive capacity, inform better decision-making, and contribute practical insight to AI safety while pushing technical depth through hands-on research.

## Starting premise

Web signup and onboarding flows are among the most adversarial interactive environments available today: they combine non-stationary page layouts, multi-step verification gates, invisible behavioral scoring, and sophisticated fingerprinting — all designed to distinguish humans from automated agents. This makes them a compelling testbed for training robust, generalizable computer-use agents via reinforcement learning.

This project trains a single LLM — Qwen 3.5-35B-A3B (MoE, 3B active parameters) — end-to-end to complete real web tasks by learning a policy that maps page observations to browser actions. The model is served locally via vLLM for development and inference, and trained on a cloud GH200 (96 GB HBM3) via QLoRA. The agent improves through a **prompted rollouts → supervised fine-tuning → GRPO reinforcement learning** pipeline, with reward signals derived from task completion, intermediate progress, and behavioral realism.

### What makes this technically interesting

- **Sparse, delayed rewards in a long-horizon task.** A signup flow is 10–30 steps. The terminal reward (account created) is binary and only observed at the end. Intermediate shaping signals must be designed carefully to avoid reward hacking.
- **Adversarial, non-stationary environment.** Detection systems retrain continuously; page layouts change; challenge frequency is conditioned on the agent's own behavioral history within a session. The environment is not an offline dataset — it fights back.
- **Real-world grounding with no simulator.** Unlike Atari or MuJoCo, there is no reset-to-identical-state. Each episode runs against a live website with server-side state, rate limits, and IP reputation. Exploration is expensive.
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

1. **Browser environment** — Playwright with real Chromium, exposing a Gym-style `reset()`/`step()` interface. The agent receives an observation (DOM accessibility tree or screenshot) and emits a discrete action.
2. **Agent loop** — Observe → prompt LLM → parse structured action → execute in browser → collect (observation, action, reward) tuple.
3. **RL trainer** — Collects N full trajectory rollouts, scores each with the shaped reward function, and applies GRPO-style group-relative advantage weighting to update the policy toward higher-reward trajectories. No critic network required (though per-step credit assignment is coarser as a result).

---

## Observation and Action Design

### Observation strategy

| Approach | Observation format | Model requirement | Tradeoff |
|----------|-------------------|-------------------|----------|
| **DOM / accessibility tree** (Phase 1–3) | Structured text: element roles, labels, IDs, input states | Text-only LLM (Qwen 3.5-35B-A3B) | Smaller observation space, faster RL iterations. More brittle across sites with different DOM structures. |
| **Vision / screenshot** (Phase 4+) | Raw pixel image of the viewport | Vision-language model (Qwen2.5-VL-7B) | Generalizes better (buttons look like buttons). Handles canvas/WebGL UIs and obfuscated DOM. Higher compute cost per step. |

**Phase 1 uses DOM-text** because it works with the current text-only model and keeps observations compact, which matters when each RL training step requires N parallel rollouts through a real browser.

The observation method does not affect bot-detection surface area — both approaches use Playwright driving a real Chromium instance. The detection surface is determined by the browser execution environment, not the agent's perception of it.

### Action space

```python
ACTIONS = [
    "click(element_id)",     # DOM-mode: click by accessible element reference
    "click(x, y)",           # vision-mode: click by pixel coordinate
    "type(element_id, text)",
    "scroll(direction)",     # up | down
    "press(key)",            # Enter, Tab, Escape, etc.
    "goto(url)",
    "wait(seconds)",
    "request_email_code",    # signals the Verification Oracle
    "request_sms_code",      # signals the Verification Oracle
]
```

Actions are emitted as structured text by the LLM and parsed deterministically. The action space is intentionally narrow — the agent must compose complex behaviors from primitive actions, which is where the RL signal provides value.

---

## Reward Design

Reward shaping is critical because the terminal signal (account created or not) is too sparse for sample-efficient learning. The reward function combines progress milestones, completion, and behavioral quality:

| Signal | Reward | Detection method |
|--------|--------|------------------|
| Reached signup form | +0.3 | URL pattern or DOM contains signup-specific elements |
| Filled a field correctly | +0.1 per field | DOM state inspection after typing |
| Advanced to next step in flow | +0.1 | Step transition detected via URL or DOM change |
| Completed verification | +0.15 | Verification input accepted |
| **Account created successfully** | **+1.0** | Success page URL, confirmation text, or API check |
| Timed out / max steps exceeded | 0.0 | Episode truncation |
| Action executed too fast (< 500ms gap) | −0.1 | Timing check in environment wrapper |
| Click dead-center on element | −0.05 | Coordinate offset analysis |
| CAPTCHA triggered | −0.05 | CAPTCHA iframe/element detected in DOM |
| Efficiency bonus | +0.2 × (1 − steps/max_steps) | Fewer steps = higher bonus |

**Design rationale:** Negative rewards for bot-like micro-behaviors incentivize the policy to learn human-plausible interaction patterns — not as hardcoded heuristics, but as emergent behavior shaped by the reward landscape. The agent should discover that browsing before signing up reduces CAPTCHA frequency, that variable typing speed avoids keystroke analysis flags, and that non-centered clicks avoid heatmap detection — all through trial-and-error RL.

---

## Training Pipeline

### Phase 1 — Prompted rollouts (no weight updates)

Run the pre-trained Qwen 3.5-35B-A3B as a zero-shot agent. The model receives the DOM observation and a system prompt describing the action format. Trajectories (successful and failed) are logged as (observation, action, reward) sequences.

**Purpose:** Establish baseline performance, collect seed trajectories for SFT, and iterate on prompt engineering + environment instrumentation.

### Phase 2 — Supervised fine-tuning (SFT)

Filter Phase 1 trajectories to successful completions. Format as multi-turn conversations: each turn is an (observation, action) pair, preserving the full sequential structure. Fine-tune **the same Qwen 3.5-35B-A3B** with QLoRA on a cloud GH200 using `trl.SFTTrainer`.

| Setup | Model weights | QLoRA training overhead | Total VRAM | Hardware |
|-------|--------------|------------------------|------------|----------|
| **Primary (cloud)** | ~17.5 GB (4-bit) | ~15–20 GB | **~35–40 GB** | GH200 96 GB HBM3 |
| Fallback (local) | ~7–15 GB | ~3–5 GB | ~10–20 GB | Consumer GPU (Qwen2.5-7B/3B) |

**Training the same model you inference with** eliminates the two hardest problems in the previous distillation-based approach: (a) no distribution mismatch — the model collecting trajectories IS the model being trained, so there is no DAgger-style compounding error from behavioral cloning; (b) no capability ceiling — the 35B MoE retains its full reasoning capacity for planning multi-step actions over complex DOM structures.

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

**Why trajectory-level GRPO instead of PPO:** PPO's per-step value function provides finer-grained credit assignment, which is generally preferable for multi-step tasks. However, it requires fitting a critic network — for a 35B MoE model, even a smaller critic adds significant VRAM pressure alongside the policy, QLoRA optimizer states, and KV cache on a single GH200. Trajectory-level GRPO avoids the critic entirely. The shaped reward function (intermediate progress signals at each step) partially compensates for coarser credit assignment: even though the trajectory-level advantage is shared across all steps, each (observation, action) training pair individually carries step-level signal about what worked. PPO with a learned critic remains a planned upgrade if multi-GPU training becomes available.

**Why not vanilla `trl.GRPOTrainer`:** The standard API assumes a static prompt with N sampled completions. In an interactive environment, the "prompt" (observation) changes after every action, so rollouts must be collected externally. `veRL` (ByteDance) and `OpenRLHF` are designed for LLM RL with environment interaction and are candidates for replacing the custom training loop as the project matures.

**Exploration and sample efficiency:** Each episode consumes real resources (proxy bandwidth, IP reputation). To manage exploration cost: (a) the SFT warm-start ensures the initial RL policy succeeds at a non-trivial rate, avoiding the cold-start problem of all-failures-no-signal; (b) N is set high enough (8–16 trajectories per batch) that at least some trajectories succeed, giving a ranking signal; (c) a trajectory replay buffer retains high-reward trajectories from prior batches for off-policy augmentation; (d) curriculum ordering starts with simplified task variants (pre-navigated to the signup form) before full end-to-end episodes.

### Phase 4 — Cross-task generalization

Train on multiple platforms with random task sampling per episode. Evaluate on held-out sites the model has never seen. The hypothesis: a policy trained on diverse adversarial signup flows should generalize to novel flows better than one memorizing a single site's DOM structure.

---

## Adversarial Environment Dynamics

The agent operates against a layered detection stack. Understanding these layers informs reward design, environment configuration, and the overall research framing.

### Layer 1: Browser and runtime fingerprinting (passive, every page load)

- **`navigator` property consistency** — `webdriver`, `plugins`, `languages`, `hardwareConcurrency`, `platform`, `maxTouchPoints`, `connection` API. Inconsistencies between any of these and the claimed User-Agent are flagged.
- **Client Hints API (`Sec-CH-UA-*`)** — A secondary browser identity channel. Mismatches between Client Hints headers and the User-Agent string are a strong automation signal.
- **Canvas / WebGL / AudioContext fingerprinting** — GPU-specific rendering hashes. Headless environments produce distinct hashes (SwiftShader for WebGL, software rendering for canvas, silent/zeroed audio buffers).
- **Emoji rendering fingerprinting** — OS-level text engines produce visually distinct emoji per OS/version. Docker containers on Linux often produce blank rectangles — an instant headless signal.
- **Font metrics / text shaping** — Different OS text engines (HarfBuzz, DirectWrite, CoreText) produce subtly different glyph metrics. This identifies the OS even when font lists are spoofed.
- **Cross-context consistency** — Detection scripts run checks inside Web Workers, iframes, and Shadow DOM. Stealth patches that only modify the main window context are caught by Worker-side property enumeration.

### Layer 2: Behavioral scoring (active, during interaction)

- **Mouse dynamics** — Bezier-like curves with acceleration/deceleration, micro-jitter on hover, non-centered click targets. Constant velocity, straight lines, and perfect centering are flagged.
- **Keystroke dynamics** — Variable inter-key timing (50–300ms), key hold duration (50–150ms), occasional corrections. Zero variance or zero-duration key holds are detectable.
- **Timing and rhythm** — Page dwell time before first action, inter-action timing variance, form completion speed. Machine-like regularity is the primary signal.

### Layer 3: Network and transport intelligence (server-side)

- **TLS fingerprinting (JA3/JA4)** — The TLS ClientHello structure uniquely identifies each HTTP client. Playwright using real Chromium inherits Chrome's TLS stack, so this is a non-issue with a real browser.
- **IP reputation and ASN classification** — Datacenter IPs are pre-flagged. Residential and mobile carrier IPs carry higher trust. CGNAT on mobile carriers means real users share the same IP, making mobile IPs effectively un-blockable.
- **WebRTC / DNS leak detection** — WebRTC ICE candidates can reveal the real IP behind a proxy. DNS lookups for unique subdomains can identify the resolver.

### Layer 4: Challenge systems

- **reCAPTCHA v3** — Invisible behavioral scoring over the entire session, heavily influenced by cookie history (Google `NID`, `_GRECAPTCHA`).
- **Arkose Labs (FunCaptcha)** — Interactive 3D puzzles with mouse-movement monitoring during solve. Used by X and LinkedIn. Fundamentally hard to solve programmatically.
- **Cloudflare Turnstile** — Combines fingerprinting, proof-of-work, and behavioral signals. Falls back to visual challenge only if invisible check fails.

### Design implication

The browser environment wrapper handles infrastructure-level concerns (stealth patches, proxy routing, WebRTC policy, fingerprint consistency). The RL policy learns higher-level behavioral patterns: browsing before high-risk actions, realistic timing, natural interaction cadence. The separation is deliberate — low-level countermeasures are deterministic infrastructure; high-level behavioral patterns are learned.

---

## Verification Oracle Abstraction

Verification (email codes, SMS codes) is separated from the agent's policy as an **environment oracle**:

```
Agent                              Environment
  │                                     │
  │──── request_email_code ────────────▶│
  │                                     │──▶ VerificationOracle
  │                                     │       polls IMAP inbox
  │◀──── observation: code = "482901" ──│◀── returns code
  │                                     │
  │──── type(code_field, "482901") ────▶│
```

The agent learns *when* to request a code (recognizing the "enter code" input in the DOM) and *how* to enter it, but the logistics of email/SMS retrieval are deterministic infrastructure — not a learned skill. This keeps the RL signal focused on web interaction policy.

Full methodology on verification bypass is not disclosed to prevent abuse. If interested feel free to ask, I have some interesting, novel ideas.

---

## Environment Authenticity Progression

The project follows a progressive fidelity ladder, where each tier increases the realism of the execution environment:

| Tier | Method | Detection resistance | Complexity |
|------|--------|---------------------|------------|
| 1 | Playwright + stealth patches | Good (~80%) | Low |
| 2 | + Residential proxy + session warming | Very good (~90%) | Medium |
| 3 | + Anti-detect browser (custom fingerprint profiles) | Excellent (~95%) | Medium |
| 4 | + Custom Chromium build (source-level artifact removal) | Near-perfect (~98%) | High |
| 5 | Full VM + OS-level input injection (no CDP) | Virtually undetectable | Very high |

**Phase 1 targets Tier 2** — sufficient for some platform's training where the focus is getting the RL loop operational. The architectural separation between the agent loop and the browser environment means the execution backend can be swapped from Playwright to a VM-based setup without changing the agent's observation/action interface.

**Tier 5 insight:** At the VM level, the browser has zero automation artifacts because no automation protocol exists — input is injected at the OS level (xdotool/uinput), and the agent observes via screenshots. This is architecturally equivalent to how some frontier lab's computer-use system operates. It requires switching from DOM-text to vision-based observation (hence Phase 4's VLM upgrade).

**LLM inference latency as a feature:** On a local consumer GPU, the 35B model takes 2–5 seconds per action decision — naturally mapping to human "think time." No artificial delays are needed. The variance in inference time (affected by prompt length, generation length, and GPU load) produces natural-looking timing distributions. On faster cloud hardware (GH200), inference is sub-second, so the environment wrapper adds calibrated random delays (sampled from a log-normal distribution fitted to human interaction timing) during rollout collection to maintain behavioral realism.

---

## Related Work and Positioning

| Project | Approach | Difference from this work |
|---------|----------|--------------------------|
| **AgentQ** (MultiOn, 2024) | MCTS + DPO for web agents | Uses search-time compute (MCTS) for planning + offline preference learning (DPO). This project uses online RL with trajectory-level GRPO against a live adversarial environment, where reward includes behavioral quality — not just task completion. |
| **DigiRL** (2024) | RL for device control | Operates on mobile device emulators with vision; this project starts with DOM-text on real websites with active anti-automation. |
| **BrowserGym** (ServiceNow) | Gym-compatible browser environment | Provides the environment abstraction; this project builds a similar wrapper but adds adversarial dynamics (detection, challenges, verification gates). |
| **WebAgent** (Google, 2023) | HTML-understanding web agent | Supervised/prompted; no RL fine-tuning against live adversarial feedback. |
| **OpenHands** (formerly OpenDevin) | General computer-use agent | Broad scope; not specialized for adversarial web environments or RL training. |

**This project's contribution:** Combining online RL (trajectory-level GRPO) with a real, adversarial web environment where the reward function explicitly encodes both task completion and behavioral realism. The agent learns not just *what* to do, but *how* to act in a way that preserves trust signals — an objective that doesn't exist in standard web agent benchmarks. The training methodology handles the unique constraints of this setting: expensive exploration (each episode costs real resources), non-stationarity (the environment adapts), and the absence of a simulator (no cheap resets).

---

## Infrastructure

- **Model serving (local):** Qwen 3.5-35B-A3B via vLLM (OpenAI-compatible API, `localhost:8000`), Docker Compose with NVIDIA GPU passthrough. Used for development, prompt iteration, and local inference (~20 GB VRAM).
- **Training compute (cloud):** NVIDIA GH200 (96 GB HBM3) on Lambda Labs. QLoRA fine-tuning of the full 35B MoE model fits in ~35–40 GB, leaving headroom for batch size and sequence length. Rollout collection and training alternate on the same GPU.
- **Browser automation:** Playwright + Chromium with stealth patches, residential proxy with sticky sessions. Runs on a lightweight CPU instance or locally — does not require GPU.
- **Training stack:** `trl` (SFTTrainer), `peft` (QLoRA), `transformers`, `accelerate`; custom trajectory-level GRPO training loop (candidates for replacement: `veRL`, `OpenRLHF`)
- **Cost estimate:** GH200 on Lambda ≈ $2–3/hr. SFT (a few hours) + RL (tens of hours over multiple sessions) ≈ $50–150 total training compute.

---

## Current Status

The vLLM serving infrastructure is operational. Platform-specific research (detection stacks, DOM structures, verification flows) is complete for most major sites. The immediate next step is building the browser environment wrapper and agent loop (Phase 1), collecting trajectories, and iterating toward the SFT and GRPO training phases.

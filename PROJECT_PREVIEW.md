# Project Preview

## Premise

This project explores **training a local AI browser agent** to complete high-friction web signup and onboarding flows while minimizing anti-bot triggers.  
The stack uses a locally served **Qwen model** as the policy brain and reinforcement learning (RL) to improve behavior through repeated environment interaction.

At a high level, the agent learns to:
- Observe page state (DOM/accessibility-first, with optional vision upgrades later)
- Choose browser actions (click, type, scroll, wait, navigate)
- Complete multi-step verification flows
- Adapt behavior to reduce challenge escalation and detection risk

---

## Core Detection Vectors We Optimize Against

### 1) Browser and runtime fingerprinting
- `navigator` and client-hint consistency checks
- Canvas/WebGL/Audio fingerprints
- Font metrics and emoji rendering signatures
- Screen/window realism (headed vs headless artifacts)
- Cross-context consistency (main window, iframe, workers)

### 2) Behavioral scoring
- Mouse trajectory realism (curves, jitter, non-perfect targeting)
- Keystroke dynamics (variable cadence, natural pauses, occasional corrections)
- Scroll rhythm and dwell time realism
- Inter-action timing variance (avoid machine-like regularity)
- Cold-session vs warmed-session behavior differences

### 3) Network and transport intelligence
- TLS and HTTP/2 fingerprint profiling (JA3/JA4-class signals)
- IP/ASN reputation and proxy-origin quality
- DNS/WebRTC leak checks
- Geo-consistency checks (IP, locale, timezone, language)
- Endpoint and session rate limiting

### 4) Challenge systems and post-signup monitoring
- Risk-based CAPTCHA/challenge escalation
- Verification friction (email, phone, and possible identity escalation)
- Early-account trust scoring after signup
- Restriction cascades based on first-session behavior

---

## Verification Bypass Architecture (Concise)

The project separates **agent policy** from **verification infrastructure**.

- **Verification Oracle pattern**  
  The agent requests codes through environment actions (e.g., `request_email_code`) instead of directly managing inboxes/SMS APIs.

- **Phone strategy**  
  Prefer high-trust number sources and only request SMS resources when strictly required by flow state.

- **OAuth fallback paths**  
  Where possible, route through federated sign-in to reduce challenge frequency and verification friction.

This keeps RL focused on web interaction policy, while verification logistics remain deterministic infrastructure.

---

## Advanced Evasion Methodology Roadmap

The methodology follows a progressive reliability ladder:

1. **Stealth-enhanced real browser automation**  
   Baseline practical setup for rapid iteration and trajectory collection.
2. **Session/profile warming**  
   Build realistic history before high-risk actions to improve trust scoring.
3. **Fingerprint consistency hardening**  
   Maintain coherent cross-signal identity over time.
4. **Higher-fidelity execution environments**  
   Move toward full desktop/VM or device-realistic operation where needed.
5. **Infrastructure quality upgrades**  
   Improve IP quality, profile persistence, and anti-leak controls as constraints tighten.

Guiding principle: **reduce spoofing, increase environmental authenticity**.

---

## Qwen + RL Training Premise

- **Policy model**: local Qwen endpoint (OpenAI-compatible) for agent decision-making
- **Environment**: real browser wrapper exposing observation/action loop
- **Learning recipe**:
  - Prompted rollouts to collect trajectories
  - Supervised fine-tuning on successful traces
  - RL optimization (GRPO-style preference over higher-reward trajectories)

### Reward shaping priorities
- Positive reward for progress milestones and successful completion
- Penalties for bot-like behavior (too-fast actions, perfect center-clicking, cold high-risk jumps)
- Bonus for efficient, stable completion under realistic timing

Result: the policy learns not only *what* action to take, but *how* to act in a way that preserves trust signals.

---

## Current Project Focus

- Build a reproducible local loop: `observe -> decide -> act -> reward`
- Instrument success/failure trajectories for fast policy iteration
- Treat hard verification gates as environment-assisted steps early on
- Incrementally transfer from easier flows to more adversarial flows

---

## Research Scope

This repository is positioned as **computer-use agent research** in adversarial web environments: studying robust automation, detection-aware policy learning, and practical RL for real browser tasks.

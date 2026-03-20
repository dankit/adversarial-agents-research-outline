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

**LLM inference latency as a feature:** On the remote cloud GPU, the 35B model's inference time (affected by prompt length, generation length, and GPU load) produces variable per-action latency. On fast hardware (GH200/H100), inference is sub-second, so the environment wrapper adds calibrated random delays (sampled from a log-normal distribution fitted to human interaction timing) during rollout collection to maintain behavioral realism. The SSH tunnel adds negligible latency (~1ms) since it's just TCP forwarding.

---
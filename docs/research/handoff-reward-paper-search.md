# Handoff: literature search for the reward-design comparison

You are picking up a literature task for a master's thesis. The thesis compares reward functions for a Double DQN (DDQN) self-driving car. The candidate designs are already chosen (below). Your job is to **find, verify and rank the research papers** that back each design, its hyperparameters, and the evaluation protocol, and to replace every "unverified" claim with a checked citation or a clear "not supported".

The repo is read-only for you except for the one output file named in Step 5. Do not edit code.

## Setup facts you need (verified from the code, 2026-11-26)

- 2-D pygame race track (loop, 38 checkpoints on default `template5`), single car, no other traffic. Walls are the road edge; leaving the road = crash.
- Observation: 12 normalised wall-ray distances, velocity, traction, drift angle, checkpoint index fraction, stall-timer fraction (17 dims). No position, heading, or previous action.
- Actions: 9 discrete, the 3×3 grid {reverse, none, accelerate} × {left, none, right}.
- Episodes end on crash or 1000 steps without a checkpoint (both `done=True`, penalised), or truncate at 3000 steps.
- DDQN, γ = 0.99, Huber loss, gradient-norm clip 1.0, prioritized replay, soft target update τ = 0.01, one hidden layer of 256.
- No traffic, so headway, lane invasion and traffic rules do not exist. TTC is only computable against walls, from forward rays.
- Details live in `game/__init__.py`, `game/game_car.py`, `ai/ddqn_agent.py`, and `docs/adr/0001-progress-led-reward.md`. Read them only if a paper's applicability hinges on a detail not listed here.

## The design lineup (the thing every paper must map onto)

All designs are dense and expected to learn; they should compete on a Pareto front (lap time vs crash rate vs comfort), not on works/fails. Sparse-only and curiosity-only designs were **rejected by the user**: do not spend effort on them.

| ID | Design | Core idea | Comparison axis |
|---|---|---|---|
| D1 | Checkpoint-led (current) | +bonus per checkpoint, local progress, small shaping, crash penalty | A: subgoal bonus vs continuous progress |
| D2 | Continuous progress + severity-scaled crash | reward = arc-length progress per step; crash penalty grows with speed² | A, B (baseline) |
| D3 | Velocity tracking | reward speed along route toward a target speed, lane-centring, heading error | A: outcome vs behaviour proxy |
| D4 | Risk-aware progress (weighted sum) | D2 − TTC/risk penalty − lane-offset penalty | B: reactive vs anticipatory safety |
| D5 | Multiplicative safety gate | progress × safety gate × lane gate (same terms as D4) | C: sum vs product |
| D6 | Thresholded lexicographic DDQN | separate Q-heads for safety and progress, safety filtered first | C: weights vs priorities |
| D7 | Context-switching reward machine (optional) | different reward mode on corners vs straights | context vs fixed |

## Claims to verify (current status)

Each row is a claim the previous agent relied on. Close every row: **confirmed** (with exact section/equation), **corrected**, or **not supported**.

| # | Claim | Paper | Status |
|---|---|---|---|
| V1 | Course-progress proxy reward for racing; wall-contact penalty scales with squared speed | Fuchs, Song, Kaufmann, Scaramuzza, Dürr, "Super-Human Performance in Gran Turismo Sport Using Deep RL", RA-L/ICRA 2021, arXiv 2008.07971 (Sec. III.A) | progress confirmed; speed² penalty **unverified** |
| V2 | Reward components used by GT Sophy (progress, off-course, wall, sportsmanship) | Wurman et al., "Outracing champion Gran Turismo drivers with deep RL", Nature 602, 2022 (Methods + supplementary MOESM1) | **unverified** |
| V3 | Reward = forward speed, episode ends on safety-driver intervention | Kendall et al., "Learning to Drive in a Day", ICRA 2019 | **unverified** |
| V4 | Ellipsoid/RSS-based risk term; ~21% lower collision rate vs baselines | Abouelazm, Michel, Gremmelmaier, Joseph, Schörner, Zöllner, "Balancing Progress and Safety: A Novel Risk-Aware Objective for RL in AD", arXiv 2505.06737, IV 2025 | headline confirmed; **term formula unverified** |
| V5 | Reward components categorised (safety, progress, comfort, rules); context-dependence argument; mentions reward machines | Abouelazm, Michel, Zöllner, "A Review of Reward Functions for RL in the context of AD", arXiv 2405.01440 | **venue inconsistent** (header says IV 2025 dated June 2024); category list **unverified** |
| V6 | Thresholded lexicographic DQN: one agent/head per objective, thresholds on Q; exact bootstrap target | Li & Czarnecki, "Urban Driving with Multi-Objective Deep RL", arXiv 1811.08586, AAMAS 2019 | method confirmed; **target computation unverified** |
| V7 | 8 reward sanity checks; most published AD rewards fail the first 3; names of the checks; risk indifference-point analysis | Knox, Allievi, Banzhaf, Schmitt, Stone, "Reward (Mis)design for Autonomous Driving", Artificial Intelligence 316 (2023) 103829, arXiv 2104.13906 | count + finding confirmed; **check names unverified** |
| V8 | Potential-based shaping preserves optimal policy | Ng, Harada, Russell, ICML 1999 | well known; confirm the theorem's MDP assumption |
| V9 | In episodic tasks Φ(terminal) must be 0 for invariance | Grześ, "Reward Shaping in Episodic RL", AAMAS 2017 | **unverified** (from memory) |
| V10 | Reward machines formalism | Toro Icarte, Klassen, Valenzano, McIlraith, ICML 2018 | **unverified** (from memory) |
| V11 | IQM + bootstrap CIs for few-seed RL comparison | Agarwal et al., "Deep RL at the Edge of the Statistical Precipice", NeurIPS 2021 | from memory; confirm |

## Steps

### 1. Verify V1–V11
Open the primary source (arXiv, publisher, or author page) for each row and read the relevant section. Done when every row has a status of confirmed / corrected / not supported, each with a section, equation or page reference. A secondary summary (blog, review site, search snippet) never closes a row.

### 2. Fill the per-design evidence gaps
For each of D1–D7, find papers that use or test that design (or its closest analogue) in driving or racing RL. Priority targets:

- **D1 vs D2**: checkpoint/waypoint bonuses vs continuous progress in racing or navigation RL; any ablation comparing them.
- **D3**: target-speed / lane-centring / heading rewards in CARLA, TORCS, highway-env, or Duckietown work (the "speed × cos(heading) − lateral" family).
- **D4**: TTC- or risk-field-based reward terms; severity-scaled collision penalties (penalty ∝ impact speed or energy).
- **D5**: **multiplicative vs additive** reward aggregation anywhere in RL. This is the likely novelty of the thesis, so the search must be relentless: if nothing direct exists, say so with the queries you ran, because "no prior comparison found" is itself a thesis claim.
- **D6**: lexicographic / constrained / safety-prioritised multi-objective DQN after Li & Czarnecki (follow-ups, citations of 1811.08586, Gábor et al. 1998).
- **D7**: reward machines or context-dependent rewards in driving.

Done when each design has at least 2 primary papers, or an explicit "only N found" with the queries tried.

### 3. Find support for the shared protocol
Papers that justify: Pareto-front reporting for driving trade-offs; equal hyperparameter-tuning budgets when comparing reward functions; reward scale vs Huber loss / reward normalisation in DQN; comfort proxies (jerk, steering-reversal rate) in RL driving evaluation. Done when each item has a paper or a "not found".

### 4. Rank
For every paper found, score **relevance to this setup** (single car, wall-bounded track, discrete actions, value-based RL) as high / medium / low, with one line of why. Prefer peer-reviewed venues; list arXiv-only work but flag it.

### 5. Write the output
Write one file: `docs/research/reward-papers.md`. Structure:

1. Verification table (V1–V11 with final status and exact reference).
2. One section per design D1–D7: papers (full citation: authors, title, venue, year, link), what each paper's reward actually is (formula if given), relevance score, and any hyperparameters worth borrowing as starting values.
3. Protocol section (Step 3).
4. Gaps: what you searched for and did not find, with the queries.
5. Anything you could not access (paywalled, broken link).

## Rules

- Every citation names authors, title, venue and year, and links to a primary source.
- Quote formulas exactly as the paper gives them; mark your own paraphrase as such.
- Keep the source of each claim separable: what the paper says vs what you infer about this codebase.
- When a fact is uncertain, label it **unverified** in place rather than smoothing it into the text.

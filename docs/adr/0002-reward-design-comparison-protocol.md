# 0002: Reward-design comparison protocol

Date: 2026-11-27. Builds on `docs/research/reward-papers.md`; D1 is the form of ADR 0001.

Reward designs D1–D6 are compared only on reward-free driving outcomes of each run's **final policy**: lap time, crash rate, completion and comfort, measured over fixed evaluation starts, with three-lap evaluation episodes. Episode reward is never compared across designs, because each design has its own scale. Every design, D1 included, goes through the same tuning sweep with a selection rule fixed in advance. Without that, D1's earlier hand tuning would bias the comparison (Booth et al., AAAI 2023). D2–D5 share one severity-scaled terminal penalty and one set of track signals (track progress, lane offset, wall TTC in steps), so each comparison axis changes exactly one thing: the dense term (A), the safety signal (B), or the operator (C).

## Considered options

- **Freeze D1's ADR 0001 constants as the baseline.** Rejected. It compares a hand-tuned design against untuned ones.
- **Normalise every design to a common reward scale for the Huber loss.** Rejected. It alters D1's form. The learner is kept identical instead, and the reward-component and TD-error distributions are reported as a known confound.
- **TTC thresholds in seconds, from the literature (3–7 s).** Rejected. The forward rays see at most about 9 steps ahead at top speed, so TTC is defined in steps within that horizon.

## Consequences

- Changing an evaluation-protocol constant (number of starts, perturbation, lap target) invalidates every earlier run's outcomes. It needs a new protocol version in the run manifest, and all affected runs must be redone.
- The best evaluation round of a run is only a secondary figure. It is selected by checkpoints passed and is therefore optimistic.

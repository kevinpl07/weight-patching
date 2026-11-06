# Edit history

Training started at step 0. Training accuracy first reached 100% at step 1300. Each round below was a separate session with the same instructions; training ran 4,000 steps after each edit, with the same AdamW optimizer state carried through.

## Round 1: edit at step 2300

- Changed entries: 183 (fc1.weight 0, fc1.bias 86, fc2.weight 0, fc2.bias 97)
- Submission note: "Round 1: Fourier analysis showed ~42 neurons concentrated on one frequency (shared across a-in, b-in, out; concentration>0.2) and 86 neurons with flat spectra = memorization. Set fc1.bias of the 86 noisy neurons to -(max_a W_a + max_b W_b) - 1 (~-2.72) so they are dead on every input (they stay dead; weights decay away). Zeroed fc2.bias (+/-0.8 leftover noise, optimizer was already pushing it to 0) with optimizer reset. Trials (1000 steps): baseline held-out 0.4%->1.6%; kill-86 -> 25% (train 79%, rising); kill-97 (keep 31) -> 24% (train 58%). Kept 42 neurons for capacity. Next rounds: survivors partly re-memorize; consider pruning non-dominant-frequency noise if held-out stalls."
- Optimizer-state resets: 97 entries
- Predicted held-out acc 4,000 steps after the edit: 75.0%; actual: 44.0%
- Predicted held-out acc at the end of the run (step 62,300): 97.0%
- Right before the edit: train acc 100.0%, held-out acc 0.4%, train loss 0.0348

Curve since this edit:

| steps since edit | step | train loss | train acc | held-out acc |
|---|---|---|---|---|
| 0 (right after edit) | 2300 | 3.7924 | 19.5% | 5.0% |
| 500 | 2800 | 1.7122 | 65.4% | 15.3% |
| 1000 | 3300 | 1.2669 | 78.8% | 24.9% |
| 1500 | 3800 | 1.0070 | 85.8% | 32.4% |
| 2000 | 4300 | 0.8529 | 89.4% | 37.4% |
| 2500 | 4800 | 0.7585 | 91.4% | 40.5% |
| 3000 | 5300 | 0.6958 | 93.0% | 42.1% |
| 3500 | 5800 | 0.6543 | 93.7% | 43.0% |
| 4000 | 6300 | 0.6245 | 94.7% | 44.0% |

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

## Round 2: edit at step 6300

- Changed entries: 749 (fc1.weight 475, fc1.bias 0, fc2.weight 274, fc2.bias 0)
- Submission note: "Round 2: 42 live neurons are each ~85% one frequency (26 freqs, 1-3 neurons each), weights at the AdamW |w|<=1 cap; the off-frequency residue is heavy-tailed memorization (projecting all live neurons onto main freq+DC drops train acc 95%->39%). Edit: set the 749 largest-|residue| entries of live neurons (fc1/fc2 weights) to their projection onto main frequency + DC (removes ~46% of residue power). Optimizer update already shrinks residue / grows signal, so this fast-forwards it. Trial 1000 steps: held-out 44%->58% at 200 steps, then drifts down to 55% (re-memorization). Rejected: reviving the 86 dead neurons (bias 0 or +0.5) -> they re-memorize, held-out falls to 22%/12% in 1000 steps. Capacity (42 neurons, capped weights) looks like the long-run limit; next rounds: repeat residue cleaning on the next-largest entries."
- Optimizer-state resets: 0 entries
- Predicted held-out acc 4,000 steps after the edit: 52.0%; actual: 54.9%
- Predicted held-out acc at the end of the run (step 62,300): 62.0%
- Right before the edit: train acc 94.7%, held-out acc 44.0%, train loss 0.6245

Curve since this edit:

| steps since edit | step | train loss | train acc | held-out acc |
|---|---|---|---|---|
| 0 (right after edit) | 6300 | 1.0561 | 74.8% | 49.4% |
| 500 | 6800 | 0.6660 | 93.9% | 56.5% |
| 1000 | 7300 | 0.5968 | 95.3% | 55.2% |
| 1500 | 7800 | 0.5611 | 96.2% | 55.2% |
| 2000 | 8300 | 0.5396 | 96.6% | 55.4% |
| 2500 | 8800 | 0.5259 | 97.0% | 54.8% |
| 3000 | 9300 | 0.5165 | 97.0% | 55.0% |
| 3500 | 9800 | 0.5093 | 97.2% | 54.9% |
| 4000 | 10300 | 0.5045 | 97.2% | 54.9% |

## Round 3: edit at step 10300

- Changed entries: 749 (fc1.weight 465, fc1.bias 0, fc2.weight 284, fc2.bias 0)
- Submission note: "Round 3: Live neurons' weights and fc2.bias sit at the AdamW |w|~1 cap (fc2.bias entries pinned, Adam ratio +-1), so the network is capacity-limited. With only 1-3 neurons per frequency (26 freqs), each frequency's junk terms don't cancel and memorization fills in. The clean Fourier part with ideal bias gets ~72% train acc. The 86 dead neurons get no gradient, so the optimizer cannot recruit them. Edit: revived dead neurons 1 and 5 as a clean quadrature pair at freq 30. They copy the mirror pair 12/31 with input phases rotated by +pi/4 and +pi/4+pi and output phase by +pi/2 (clipped to +-1), so 2theta junk at k=30 cancels. That is 582 entries, optimizer moments reset. The other 167 entries set the largest off-frequency residue entries (|res|>0.76) of live neurons to their main-freq+DC projection. Trials (1000 steps): bias neurons only (7 always-on neurons supplying the missing class bias) peaked at 56.5% and fell back to 55%, so I rejected them. Quad pair only gave 59.6% flat. Quad pair plus cleaning peaked at 63.5% at 200 steps and was at 61.4% at 1000 steps, drifting down slowly. Next round: add more clean neurons (about 2 per round) into existing multi-neuron frequencies, e.g. a quadrature pair for k=16 (30/81), 19 (0/87) or 22 (23/33)."
- Optimizer-state resets: 582 entries
- Predicted held-out acc 4,000 steps after the edit: 60.0%; actual: 59.4%
- Predicted held-out acc at the end of the run (step 62,300): 63.0%
- Right before the edit: train acc 97.2%, held-out acc 54.9%, train loss 0.5045

Curve since this edit:

| steps since edit | step | train loss | train acc | held-out acc |
|---|---|---|---|---|
| 0 (right after edit) | 10300 | 0.6387 | 89.0% | 56.2% |
| 500 | 10800 | 0.4504 | 98.1% | 62.5% |
| 1000 | 11300 | 0.4190 | 98.6% | 61.4% |
| 1500 | 11800 | 0.4028 | 98.8% | 60.8% |
| 2000 | 12300 | 0.3920 | 98.9% | 60.3% |
| 2500 | 12800 | 0.3863 | 99.0% | 60.3% |
| 3000 | 13300 | 0.3828 | 99.0% | 59.9% |
| 3500 | 13800 | 0.3802 | 99.0% | 59.8% |
| 4000 | 14300 | 0.3779 | 99.0% | 59.4% |

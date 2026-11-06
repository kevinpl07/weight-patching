# Edit history

Training started at step 0. Training accuracy first reached 100% at step 1200. Each round below was a separate session with the same instructions; training ran 4,000 steps after each edit, with the same AdamW optimizer state carried through.

## Round 1: edit at step 2200

- Changed entries: 54 (fc1.weight 0, fc1.bias 54, fc2.weight 0, fc2.bias 0)
- Submission note: "Killed 54 memorization neurons (no consistent Fourier frequency across a/b/out and low spectral concentration) by setting fc1.bias=-4 (provably dead for all inputs). Kept 74 neurons whose a, b and output weights share a dominant frequency with phase relation phi_out = phi_a + phi_b."
- Right before the edit: train acc 100.0%, held-out acc 0.6%, train loss 0.0382

Curve since this edit:

| steps since edit | step | train loss | train acc | held-out acc |
|---|---|---|---|---|
| 0 (right after edit) | 2200 | 2.7540 | 35.4% | 2.8% |
| 500 | 2700 | 0.6181 | 97.9% | 7.5% |
| 1000 | 3200 | 0.2898 | 99.9% | 14.3% |
| 1500 | 3700 | 0.1433 | 100.0% | 23.0% |
| 2000 | 4200 | 0.0799 | 100.0% | 29.1% |
| 2500 | 4700 | 0.0496 | 100.0% | 34.6% |
| 3000 | 5200 | 0.0334 | 100.0% | 39.1% |
| 3500 | 5700 | 0.0239 | 100.0% | 43.5% |
| 4000 | 6200 | 0.0179 | 100.0% | 47.3% |

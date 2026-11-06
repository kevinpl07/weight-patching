# Edit history

Training started at step 0. Training accuracy first reached 100% at step 1200. Each round below was a separate session with the same instructions; training ran 4,000 steps after each edit, with the same AdamW optimizer state carried through.

## Round 1: edit at step 2200

- Changed entries: 75 (fc1.weight 0, fc1.bias 75, fc2.weight 0, fc2.bias 0)
- Submission note: "Killed 75 memorization-like neurons (top-frequency energy fraction <0.2 across fc1-a, fc1-b, fc2 rfft) by setting their fc1.bias to -10 (+optimizer reset on those biases); kept 53 neurons that are mostly single-frequency Fourier features. Trials (1000 steps): kill<0.15 -> held-out 19%, kill<0.2 -> 25% (train 92%), kill<0.3 -> 26% but train only 70% (capacity-limited). Chose <0.2. Killed neurons are permanently dead; later rounds should not expect them back."
- Optimizer-state resets: 75 entries
- Predicted held-out acc 4,000 steps after the edit: 85.0%; actual: 56.8%
- Predicted held-out acc at the end of the run (step 62,200): 97.0%
- Right before the edit: train acc 100.0%, held-out acc 0.6%, train loss 0.0382

Curve since this edit:

| steps since edit | step | train loss | train acc | held-out acc |
|---|---|---|---|---|
| 0 (right after edit) | 2200 | 3.5948 | 23.9% | 3.8% |
| 500 | 2700 | 1.3018 | 78.7% | 12.8% |
| 1000 | 3200 | 0.8239 | 91.9% | 24.9% |
| 1500 | 3700 | 0.5474 | 97.1% | 35.6% |
| 2000 | 4200 | 0.3933 | 99.2% | 43.4% |
| 2500 | 4700 | 0.3069 | 99.8% | 47.5% |
| 3000 | 5200 | 0.2511 | 99.9% | 51.3% |
| 3500 | 5700 | 0.2129 | 99.9% | 54.5% |
| 4000 | 6200 | 0.1853 | 100.0% | 56.8% |

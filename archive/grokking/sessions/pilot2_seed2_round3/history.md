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

## Round 2: edit at step 6200

- Changed entries: 748 (fc1.weight 388, fc1.bias 74, fc2.weight 194, fc2.bias 92)
- Submission note: "Found 56 clean Fourier neurons (shared freq a/b/out, phi_out=phi_a+phi_b) plus 18 high-bias, low-concentration neurons that carry nearly all memorization (killing them + fixing fc2.bias leaves train acc ~90%, matching the Fourier-projected clean circuit). Edits: (1) fc1.bias=-4 on 16 noisy neurons; (2) repurposed 2 noisy slots (3,4) as quadrature partners of f=7 neurons 7,13 (phases pa+pi/2, pb-pi/2, same output) to cancel a-b terms; (3) shifted clean-neuron biases by -0.25 (higher ReLU threshold -> better signal/noise of cos(a+b) term vs linear terms; clean-projected acc 84%->89%); (4) set fc2.bias = -W2@mean(h) (+const) to cancel per-class DC from neuron means (92 entries). Train acc after edit 91.9%, loss 0.39."
- Right before the edit: train acc 100.0%, held-out acc 47.3%, train loss 0.0179

Curve since this edit:

| steps since edit | step | train loss | train acc | held-out acc |
|---|---|---|---|---|
| 0 (right after edit) | 6200 | 0.3900 | 91.9% | 58.2% |
| 500 | 6700 | 0.1033 | 100.0% | 72.4% |
| 1000 | 7200 | 0.0777 | 100.0% | 74.9% |
| 1500 | 7700 | 0.0644 | 100.0% | 75.3% |
| 2000 | 8200 | 0.0567 | 100.0% | 76.1% |
| 2500 | 8700 | 0.0517 | 100.0% | 76.8% |
| 3000 | 9200 | 0.0481 | 100.0% | 76.4% |
| 3500 | 9700 | 0.0457 | 100.0% | 76.8% |
| 4000 | 10200 | 0.0440 | 100.0% | 77.1% |

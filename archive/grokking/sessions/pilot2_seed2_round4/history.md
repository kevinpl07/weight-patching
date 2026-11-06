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

## Round 3: edit at step 10200

- Changed entries: 128 (fc1.weight 0, fc1.bias 128, fc2.weight 0, fc2.bias 0)
- Submission note: "58 alive neurons are all clean Fourier neurons (conc ~0.85-0.9) spread over 32 frequencies (19 singletons); 70 neurons killed in rounds 1-2 are permanently dead (weights decayed to ~3e-4/0.02, zero gradient). Fourier-projected circuit gets only ~95% train acc; memorization (~40% of weight norm, contributes ~2.7 logit margin) fills the gap and caps held-out at ~77%. Partner/designed neurons tested: ~0 gain, so not used. Edits (128 entries): (1) fc1.bias=0 on all 70 dead neurons -> revived (active ~55% of inputs, tiny weights, ~zero output now) so Adam (empty v-state -> full-size steps) can recruit them as extra phase-diverse capacity; (2) fc1.bias -0.25 on 58 alive neurons (clean-projected acc 94.8%->96.3%). Train acc 100%, loss 0.044->0.103."
- Right before the edit: train acc 100.0%, held-out acc 77.1%, train loss 0.0440

Curve since this edit:

| steps since edit | step | train loss | train acc | held-out acc |
|---|---|---|---|---|
| 0 (right after edit) | 10200 | 0.1034 | 100.0% | 77.8% |
| 500 | 10700 | 0.0093 | 100.0% | 72.7% |
| 1000 | 11200 | 0.0024 | 100.0% | 57.8% |
| 1500 | 11700 | 0.0008 | 100.0% | 48.8% |
| 2000 | 12200 | 0.0003 | 100.0% | 47.0% |
| 2500 | 12700 | 0.0002 | 100.0% | 48.7% |
| 3000 | 13200 | 0.0001 | 100.0% | 52.3% |
| 3500 | 13700 | 0.0001 | 100.0% | 56.1% |
| 4000 | 14200 | 0.0001 | 100.0% | 58.7% |

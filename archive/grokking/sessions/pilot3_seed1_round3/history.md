# Edit history

Training started at step 0. Training accuracy first reached 100% at step 1300. Each round below was a separate session with the same instructions; training ran 4,000 steps after each edit, with the same AdamW optimizer state carried through.

## Round 1: edit at step 2300

- Changed entries: 717 (fc1.weight 510, fc1.bias 0, fc2.weight 207, fc2.bias 0)
- Submission note: "Weights are Adam-saturated (~+-0.88, sign-like). Nearly every neuron has a dominant frequency k with phase_out ~= phase_a + phase_b (a+b circuit); ~25 are clean Fourier, rest noisy. Edit: for the 39 cheapest neurons (phase err<0.35), flipped entries of fc1 a/b rows and fc2 column whose sign disagreed with the neuron's own best-fit cosine at k (|cos|>0.3), set to clip(1.3cos,+-0.88). 717 entries, no optimizer resets. Trial: held-out 4.8% at +1000 steps vs 2.5% baseline; variant with zeroed noisy biases + resets was slightly worse (4.2%)."
- Optimizer-state resets: 0 entries
- Predicted held-out acc 4,000 steps after the edit: 15.0%; actual: 18.9%
- Predicted held-out acc at the end of the run (step 62,300): 97.0%
- Right before the edit: train acc 100.0%, held-out acc 0.4%, train loss 0.0291

Curve since this edit:

| steps since edit | step | train loss | train acc | held-out acc |
|---|---|---|---|---|
| 0 (right after edit) | 2300 | 0.1622 | 96.8% | 1.6% |
| 500 | 2800 | 0.0165 | 100.0% | 3.2% |
| 1000 | 3300 | 0.0055 | 100.0% | 4.8% |
| 1500 | 3800 | 0.0022 | 100.0% | 6.8% |
| 2000 | 4300 | 0.0011 | 100.0% | 8.7% |
| 2500 | 4800 | 0.0006 | 100.0% | 11.0% |
| 3000 | 5300 | 0.0004 | 100.0% | 13.5% |
| 3500 | 5800 | 0.0003 | 100.0% | 15.9% |
| 4000 | 6300 | 0.0002 | 100.0% | 18.9% |

## Round 2: edit at step 6300

- Changed entries: 745 (fc1.weight 446, fc1.bias 59, fc2.weight 240, fc2.bias 0)
- Submission note: "Found 2 populations: ~53 clean sign-like Fourier neurons (frac ~0.85 at one k, bias~0, phase_out~=phase_a+phase_b; alone give 50% train acc) and ~75 noisy memorization neurons (positive bias, low Fourier fraction) that drag held-out down. Edit: killed 59 neurons with avg Fourier fraction <0.3 by setting fc1.bias to -(maxA+maxB)-1 (~-2.9; permanently dead, optimizer reset on those biases), and for 22 transitional neurons (frac 0.3-0.7, |phase err|<0.35) flipped 686 fc1/fc2 entries whose sign disagreed with the neuron's own best-fit cosine (|cos|>0.3) to clip(1.3cos,+-0.88). 745 entries. Trials (+1000 steps held-out): baseline ~25% expected; kill<0.25 only: 61% (still rising); kill<0.3+flips: 74% but flattening at 73.5-74% over last 300 steps. 69 neurons remain alive."
- Optimizer-state resets: 59 entries
- Predicted held-out acc 4,000 steps after the edit: 82.0%; actual: 78.6%
- Predicted held-out acc at the end of the run (step 62,300): 97.0%
- Right before the edit: train acc 100.0%, held-out acc 18.9%, train loss 0.0002

Curve since this edit:

| steps since edit | step | train loss | train acc | held-out acc |
|---|---|---|---|---|
| 0 (right after edit) | 6300 | 1.2243 | 67.5% | 39.2% |
| 500 | 6800 | 0.1039 | 100.0% | 71.5% |
| 1000 | 7300 | 0.0488 | 100.0% | 74.0% |
| 1500 | 7800 | 0.0282 | 100.0% | 75.8% |
| 2000 | 8300 | 0.0188 | 100.0% | 77.2% |
| 2500 | 8800 | 0.0140 | 100.0% | 77.5% |
| 3000 | 9300 | 0.0112 | 100.0% | 78.1% |
| 3500 | 9800 | 0.0094 | 100.0% | 78.3% |
| 4000 | 10300 | 0.0083 | 100.0% | 78.6% |

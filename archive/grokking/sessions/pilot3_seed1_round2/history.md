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

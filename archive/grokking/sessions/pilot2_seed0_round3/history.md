# Edit history

Training started at step 0. Training accuracy first reached 100% at step 1300. Each round below was a separate session with the same instructions; training ran 4,000 steps after each edit, with the same AdamW optimizer state carried through.

## Round 1: edit at step 2300

- Changed entries: 681 (fc1.weight 411, fc1.bias 64, fc2.weight 206, fc2.bias 0)
- Submission note: "Killed 59 least Fourier-aligned memorizing neurons via fc1 bias=-3; repurposed 5 low-alignment neurons as sign-negated partners (|x| units) of the 5 cleanest Fourier neurons (freqs 46,30,17,18,7) by flipping only mismatched-sign entries. 681 entries."
- Right before the edit: train acc 100.0%, held-out acc 0.4%, train loss 0.0348

Curve since this edit:

| steps since edit | step | train loss | train acc | held-out acc |
|---|---|---|---|---|
| 0 (right after edit) | 2300 | 3.7446 | 23.4% | 3.0% |
| 500 | 2800 | 0.7484 | 95.3% | 11.1% |
| 1000 | 3300 | 0.3620 | 99.9% | 19.4% |
| 1500 | 3800 | 0.1925 | 100.0% | 26.8% |
| 2000 | 4300 | 0.1108 | 100.0% | 34.9% |
| 2500 | 4800 | 0.0684 | 100.0% | 42.6% |
| 3000 | 5300 | 0.0453 | 100.0% | 48.1% |
| 3500 | 5800 | 0.0328 | 100.0% | 52.1% |
| 4000 | 6300 | 0.0251 | 100.0% | 55.3% |

## Round 2: edit at step 6300

- Changed entries: 749 (fc1.weight 0, fc1.bias 15, fc2.weight 637, fc2.bias 97)
- Submission note: "Mean-ablated 15 remaining memorizing neurons (low Fourier purity, positive bias): fc1 bias set below their max pre-activation (dead), train-mean activation x fc2 column folded into fc2.bias. Then set the 637 largest-|residual| fc2 entries of the 54 Fourier neurons (sparse negative memorization spikes, 71% of fc2 residual energy) to their projection onto freqs {k,3k}, folding the removed mean contribution into fc2.bias. Train acc 100->92.6%, loss 0.025->0.334. 749 entries."
- Right before the edit: train acc 100.0%, held-out acc 55.3%, train loss 0.0251

Curve since this edit:

| steps since edit | step | train loss | train acc | held-out acc |
|---|---|---|---|---|
| 0 (right after edit) | 6300 | 0.3337 | 92.6% | 68.7% |
| 500 | 6800 | 0.1146 | 100.0% | 76.5% |
| 1000 | 7300 | 0.0945 | 100.0% | 75.7% |
| 1500 | 7800 | 0.0838 | 100.0% | 75.6% |
| 2000 | 8300 | 0.0772 | 100.0% | 75.2% |
| 2500 | 8800 | 0.0731 | 100.0% | 75.2% |
| 3000 | 9300 | 0.0705 | 100.0% | 74.8% |
| 3500 | 9800 | 0.0688 | 100.0% | 74.8% |
| 4000 | 10300 | 0.0675 | 100.0% | 74.6% |

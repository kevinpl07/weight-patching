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

## Round 3: edit at step 10300

- Changed entries: 749 (fc1.weight 265, fc1.bias 0, fc2.weight 387, fc2.bias 97)
- Submission note: "Projected each of the 54 live Fourier neurons' fc1 rows and fc2 columns onto {const,k,2k,3k}; fully-projected model (with fc2.bias fold of train-mean logit offset) still gets 97.4% train acc, so the residual is mostly memorization hurting held-out. Set the 652 highest-impact residual entries (|resid|^2 x activation-weighted fan-out; 265 fc1, 387 fc2) to their projected values and refolded the train-mean logit offset into all 97 fc2.bias entries. Train acc 100->99.2%, loss 0.067->0.141. 749 entries."
- Right before the edit: train acc 100.0%, held-out acc 74.6%, train loss 0.0675

Curve since this edit:

| steps since edit | step | train loss | train acc | held-out acc |
|---|---|---|---|---|
| 0 (right after edit) | 10300 | 0.1410 | 99.2% | 83.5% |
| 500 | 10800 | 0.0752 | 100.0% | 83.3% |
| 1000 | 11300 | 0.0693 | 100.0% | 79.4% |
| 1500 | 11800 | 0.0665 | 100.0% | 78.6% |
| 2000 | 12300 | 0.0648 | 100.0% | 78.0% |
| 2500 | 12800 | 0.0636 | 100.0% | 77.5% |
| 3000 | 13300 | 0.0625 | 100.0% | 77.5% |
| 3500 | 13800 | 0.0618 | 100.0% | 77.2% |
| 4000 | 14300 | 0.0613 | 100.0% | 77.2% |

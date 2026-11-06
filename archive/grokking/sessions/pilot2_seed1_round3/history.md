# Edit history

Training started at step 0. Training accuracy first reached 100% at step 1300. Each round below was a separate session with the same instructions; training ran 4,000 steps after each edit, with the same AdamW optimizer state carried through.

## Round 1: edit at step 2300

- Changed entries: 62 (fc1.weight 0, fc1.bias 62, fc2.weight 0, fc2.bias 0)
- Submission note: "Killed 62 non-Fourier (memorization) neurons by setting fc1.bias=-10 (score<0.12 = min fraction of power in the neuron's dominant frequency across a-in, b-in, out). Kept 66 Fourier neurons whose output phase = a-phase + b-phase. Removes most memorization so the remaining Fourier circuit drives training."
- Right before the edit: train acc 100.0%, held-out acc 0.4%, train loss 0.0291

Curve since this edit:

| steps since edit | step | train loss | train acc | held-out acc |
|---|---|---|---|---|
| 0 (right after edit) | 2300 | 3.2850 | 27.5% | 3.2% |
| 500 | 2800 | 0.7725 | 94.7% | 11.5% |
| 1000 | 3300 | 0.4038 | 99.7% | 19.9% |
| 1500 | 3800 | 0.2286 | 100.0% | 27.9% |
| 2000 | 4300 | 0.1409 | 100.0% | 34.3% |
| 2500 | 4800 | 0.0938 | 100.0% | 40.0% |
| 3000 | 5300 | 0.0658 | 100.0% | 46.5% |
| 3500 | 5800 | 0.0489 | 100.0% | 50.9% |
| 4000 | 6300 | 0.0375 | 100.0% | 55.1% |

## Round 2: edit at step 6300

- Changed entries: 688 (fc1.weight 388, fc1.bias 9, fc2.weight 194, fc2.bias 97)
- Submission note: "Analysis: clean Fourier part of each alive neuron (dominant freq, plus per-class DC kept consistent) scores ~91% on train pairs, while held-out is 55%, so held-out is limited by memorization residue, not circuit quality. Mean-ablated 9 mostly non-Fourier, high-residual neurons (3,8,56,68,69,98,77,79,96): fc1.bias=-10, and their train-mean contribution W2[:,i]*mean(h_i) folded into fc2.bias, so per-class offsets that b2 was balancing stay intact (zero-ablation would wreck them). Fully cleaned neurons 37 and 65 (freq 48, 36): fc1 rows and fc2 column projected onto the dominant frequency, with the DC change also moved into fc2.bias. 688 entries. Train acc 96.3%, loss 0.227; clean proxy 89%."
- Right before the edit: train acc 100.0%, held-out acc 55.1%, train loss 0.0375

Curve since this edit:

| steps since edit | step | train loss | train acc | held-out acc |
|---|---|---|---|---|
| 0 (right after edit) | 6300 | 0.2273 | 96.3% | 59.2% |
| 500 | 6800 | 0.0894 | 100.0% | 69.5% |
| 1000 | 7300 | 0.0717 | 100.0% | 70.5% |
| 1500 | 7800 | 0.0617 | 100.0% | 71.3% |
| 2000 | 8300 | 0.0553 | 100.0% | 71.3% |
| 2500 | 8800 | 0.0511 | 100.0% | 71.7% |
| 3000 | 9300 | 0.0483 | 100.0% | 71.7% |
| 3500 | 9800 | 0.0465 | 100.0% | 71.5% |
| 4000 | 10300 | 0.0452 | 100.0% | 71.5% |

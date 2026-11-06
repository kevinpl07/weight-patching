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

## Round 2: edit at step 6200

- Changed entries: 747 (fc1.weight 744, fc1.bias 3, fc2.weight 0, fc2.bias 0)
- Submission note: "Found: 50 clean single-frequency Fourier neurons over ~31 freqs (mostly 1 neuron/freq, phases obey out=a+b), 3 messy high-bias memorization neurons (4,6,100), 75 permanently dead neurons (bias decays with weights, never revives). Weights saturate near |w|~1 (AdamW wd=1 ceiling), so the circuit is capacity-limited: more Fourier NEURONS are needed. Trial: reviving dead neurons via bias only -> pure memorization, held-out fell 57->30% in 1000 steps (do NOT do this). Edit: killed 4,6,100 (bias -10) and turned dead neurons 1,3,8,9,10,12 into sign-flipped 'twins' of single-neuron-frequency units 51,67,15,5,42,86 (fc1 row = -cos projection at that freq, amplitude 1, only the 124 largest-|value| entries set to fit budget; fc2 left for optimizer to build). Twin ReLU(-x)+ReLU(x)=|x| cancels the useless linear term. Trials (1000 steps): kill+3 twins 65.3%, kill+6 sparse twins 69.6% still rising. Later rounds: add more twins/phase-shifted Fourier neurons on remaining dead neurons (~69 left)."
- Optimizer-state resets: 747 entries
- Predicted held-out acc 4,000 steps after the edit: 76.0%; actual: 72.1%
- Predicted held-out acc at the end of the run (step 62,200): 90.0%
- Right before the edit: train acc 100.0%, held-out acc 56.8%, train loss 0.1853

Curve since this edit:

| steps since edit | step | train loss | train acc | held-out acc |
|---|---|---|---|---|
| 0 (right after edit) | 6200 | 0.5284 | 87.8% | 48.5% |
| 500 | 6700 | 0.1851 | 100.0% | 67.6% |
| 1000 | 7200 | 0.1280 | 100.0% | 69.6% |
| 1500 | 7700 | 0.0999 | 100.0% | 71.0% |
| 2000 | 8200 | 0.0844 | 100.0% | 71.3% |
| 2500 | 8700 | 0.0747 | 100.0% | 71.2% |
| 3000 | 9200 | 0.0684 | 100.0% | 71.9% |
| 3500 | 9700 | 0.0647 | 100.0% | 72.1% |
| 4000 | 10200 | 0.0623 | 100.0% | 72.1% |

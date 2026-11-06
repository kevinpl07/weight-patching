# Experiment 3: Whisper hallucination on non-speech

Non-speech: 300 held-out clips; halluc = share of non-empty normalized transcripts; no-tag = the same ignoring outputs that are only a sound tag (e.g. "Music").

| arm | entries | halluc % | silence | noise | env | music | music+noise | rel drop % | no-tag % | music no-tag | no-tag rel drop % | GPU min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 0 | 84.0 | 43.3 | 48.3 | 100.0 | 100.0 | 76.2 | 0.0 | 79.7 | 82.9 | 0.0 | 3.70 |
| thresholds | 0 | 6.7 | 0.0 | 0.0 | 7.9 | 12.9 | 6.9 | 92.1 | 6.0 | 10.0 | 92.5 | 0.00 |
| thresholds_strict | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 | 100.0 | 0.00 |
| openai_thresholds | 0 | 49.7 | 30.0 | 1.7 | 69.3 | 60.0 | 33.1 | 40.9 | 45.7 | 44.3 | 42.7 | 0.00 |
| vad | 0 | 12.3 | 0.0 | 0.0 | 18.6 | 15.7 | 8.5 | 85.3 | 12.3 | 15.7 | 84.5 | 2.50 |
| vad_thresholds | 0 | 0.3 | 0.0 | 0.0 | 0.0 | 1.4 | 0.8 | 99.6 | 0.3 | 1.4 | 99.6 | 0.00 |
| vad_thresholds_strict | 0 | 0.3 | 0.0 | 0.0 | 0.0 | 1.4 | 0.8 | 99.6 | 0.3 | 1.4 | 99.6 | 0.00 |
| head_scan | 49152 | 79.7 | 30.0 | 33.3 | 100.0 | 100.0 | 69.2 | 5.2 | 73.0 | 75.7 | 8.4 | 3.80 |
| head_scan_vad | 49152 | 12.7 | 0.0 | 0.0 | 19.3 | 15.7 | 8.5 | 84.9 | 12.7 | 15.7 | 84.1 | 5.00 |
| agent1 | 1537 | 0.3 | 0.0 | 0.0 | 0.0 | 1.4 | 0.8 | 99.6 | 0.3 | 1.4 | 99.6 | 2.50 |
| agent1_vad | 1537 | 0.7 | 0.0 | 0.0 | 0.7 | 1.4 | 0.8 | 99.2 | 0.7 | 1.4 | 99.2 | 3.20 |
| agent2 | 1537 | 0.3 | 0.0 | 0.0 | 0.0 | 1.4 | 0.8 | 99.6 | 0.3 | 1.4 | 99.6 | 2.50 |
| agent2_vad | 1537 | 0.7 | 0.0 | 0.0 | 0.7 | 1.4 | 0.8 | 99.2 | 0.7 | 1.4 | 99.2 | 2.00 |
| agent3 | 1537 | 0.7 | 0.0 | 0.0 | 0.7 | 1.4 | 0.8 | 99.2 | 0.7 | 1.4 | 99.2 | 2.90 |
| agent3_vad | 1537 | 0.7 | 0.0 | 0.0 | 0.7 | 1.4 | 0.8 | 99.2 | 0.7 | 1.4 | 99.2 | 2.90 |
| random1_count | 1537 | 99.0 | 100.0 | 100.0 | 97.9 | 100.0 | 100.0 | -17.9 | 99.0 | 100.0 | -24.3 | 12.90 |
| random1_same | 1537 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 | 100.0 | 16.30 |
| random2_count | 1537 | 84.0 | 43.3 | 48.3 | 100.0 | 100.0 | 76.2 | 0.0 | 80.0 | 85.7 | -0.4 | 3.20 |
| random2_same | 1537 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | -19.0 | 99.3 | 97.1 | -24.7 | 4.00 |
| random3_count | 1537 | 85.0 | 43.3 | 53.3 | 100.0 | 100.0 | 78.5 | -1.2 | 82.3 | 90.0 | -3.3 | 3.60 |
| random3_same | 1537 | 99.7 | 96.7 | 100.0 | 100.0 | 100.0 | 100.0 | -18.7 | 99.7 | 100.0 | -25.1 | 17.90 |
| recipe | 1537 | 0.3 | 0.0 | 0.0 | 0.0 | 1.4 | 0.8 | 99.6 | 0.3 | 1.4 | 99.6 | 2.90 |
| recipe_vad | 1537 | 1.0 | 0.0 | 0.0 | 1.4 | 1.4 | 0.8 | 98.8 | 1.0 | 1.4 | 98.7 | 2.30 |

## Speech (clean = 500 test-clean; bg = 300 test-clean with ESC/music under the speech at 0/5/10 dB; other = 300 test-other; padded = 300 test-clean with 5–10 s pads)

| arm | clean WER | Δ | bg WER | Δ | bg dropped % | other WER | Δ | other dropped % | padded ins/word % | Δ | any ins % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 3.38 | +0.00 | 5.77 | +0.00 | 0.0 | 7.55 | +0.00 | 0.0 | 1.37 | +0.00 | 7.3 |
| thresholds | 3.77 | +0.38 | 6.17 | +0.40 | 2.0 | 8.31 | +0.76 | 2.3 | 0.93 | -0.45 | 6.3 |
| thresholds_strict | 3.77 | +0.38 | 6.18 | +0.42 | 2.3 | 8.30 | +0.74 | 2.7 | 0.93 | -0.45 | 6.3 |
| openai_thresholds | 3.38 | +0.00 | 5.77 | +0.00 | 0.0 | 7.55 | +0.00 | 0.0 | 1.37 | +0.00 | 7.3 |
| vad | 3.33 | -0.05 | 5.75 | -0.02 | 0.0 | 7.69 | +0.14 | 0.0 | 1.42 | +0.05 | 8.0 |
| vad_thresholds | 3.34 | -0.04 | 5.78 | +0.02 | 0.3 | 7.81 | +0.26 | 0.7 | 1.42 | +0.05 | 8.0 |
| vad_thresholds_strict | 3.34 | -0.04 | 5.78 | +0.02 | 0.3 | 7.81 | +0.26 | 0.7 | 1.42 | +0.05 | 8.0 |
| head_scan | 3.40 | +0.02 | 5.96 | +0.20 | 0.0 | 7.55 | +0.00 | 0.0 | 1.22 | -0.15 | 7.7 |
| head_scan_vad | 3.44 | +0.06 | 5.85 | +0.08 | 0.0 | 7.49 | -0.06 | 0.0 | 1.47 | +0.10 | 8.7 |
| agent1 | 3.38 | +0.00 | 5.77 | +0.00 | 0.0 | 7.55 | +0.00 | 0.0 | 1.37 | +0.00 | 7.3 |
| agent1_vad | 3.33 | -0.05 | 5.75 | -0.02 | 0.0 | 7.69 | +0.14 | 0.0 | 1.42 | +0.05 | 8.0 |
| agent2 | 3.38 | +0.00 | 5.77 | +0.00 | 0.0 | 7.55 | +0.00 | 0.0 | 1.37 | +0.00 | 7.3 |
| agent2_vad | 3.33 | -0.05 | 5.75 | -0.02 | 0.0 | 7.69 | +0.14 | 0.0 | 1.42 | +0.05 | 8.0 |
| agent3 | 3.38 | +0.00 | 5.77 | +0.00 | 0.0 | 7.55 | +0.00 | 0.0 | 1.37 | +0.00 | 7.3 |
| agent3_vad | 3.33 | -0.05 | 5.75 | -0.02 | 0.0 | 7.69 | +0.14 | 0.0 | 1.42 | +0.05 | 8.0 |
| random1_count | 107.17 | +103.79 | 115.25 | +109.49 | 0.3 | 105.56 | +98.00 | 1.3 | 63.75 | +62.38 | 17.0 |
| random1_same | 100.00 | +96.62 | 100.00 | +94.23 | 100.0 | 100.00 | +92.45 | 98.7 | 0.00 | -1.37 | 0.0 |
| random2_count | 3.29 | -0.10 | 5.63 | -0.13 | 0.0 | 7.88 | +0.33 | 0.0 | 1.32 | -0.05 | 7.3 |
| random2_same | 3.41 | +0.03 | 6.20 | +0.43 | 0.0 | 7.71 | +0.16 | 0.0 | 1.35 | -0.02 | 7.0 |
| random3_count | 3.58 | +0.20 | 6.60 | +0.83 | 0.0 | 8.22 | +0.67 | 0.0 | 5.72 | +4.35 | 7.3 |
| random3_same | 1272.42 | +1269.03 | 1334.71 | +1328.95 | 4.0 | 1551.87 | +1544.32 | 4.7 | 1352.60 | +1351.22 | 75.0 |
| recipe | 3.38 | +0.00 | 5.77 | +0.00 | 0.0 | 7.55 | +0.00 | 0.0 | 1.37 | +0.00 | 7.3 |
| recipe_vad | 3.33 | -0.05 | 5.75 | -0.02 | 0.0 | 7.69 | +0.14 | 0.0 | 1.42 | +0.05 | 8.0 |

## Breakdown

| arm | bg WER 0 dB | bg WER 5 dB | bg WER 10 dB | bg WER env | bg WER music | ins silence pad | ins noise pad | ins music pad |
|---|---|---|---|---|---|---|---|---|
| base | 9.77 | 4.34 | 3.26 | 6.34 | 5.26 | 0.41 | 0.10 | 3.67 |
| thresholds | 10.61 | 4.61 | 3.35 | 6.66 | 5.73 | 0.31 | 0.10 | 2.41 |
| thresholds_strict | 10.66 | 4.61 | 3.35 | 6.70 | 5.73 | 0.31 | 0.10 | 2.41 |
| openai_thresholds | 9.77 | 4.34 | 3.26 | 6.34 | 5.26 | 0.41 | 0.10 | 3.67 |
| vad | 9.22 | 4.93 | 3.21 | 6.20 | 5.36 | 0.52 | 0.10 | 3.72 |
| vad_thresholds | 9.32 | 4.93 | 3.21 | 6.27 | 5.36 | 0.52 | 0.10 | 3.72 |
| vad_thresholds_strict | 9.32 | 4.93 | 3.21 | 6.27 | 5.36 | 0.52 | 0.10 | 3.72 |
| head_scan | 9.77 | 4.88 | 3.35 | 6.62 | 5.39 | 0.46 | 0.10 | 3.15 |
| head_scan_vad | 9.17 | 5.20 | 3.31 | 6.41 | 5.36 | 0.52 | 0.10 | 3.88 |
| agent1 | 9.77 | 4.34 | 3.26 | 6.34 | 5.26 | 0.41 | 0.10 | 3.67 |
| agent1_vad | 9.22 | 4.93 | 3.21 | 6.20 | 5.36 | 0.52 | 0.10 | 3.72 |
| agent2 | 9.77 | 4.34 | 3.26 | 6.34 | 5.26 | 0.41 | 0.10 | 3.67 |
| agent2_vad | 9.22 | 4.93 | 3.21 | 6.20 | 5.36 | 0.52 | 0.10 | 3.72 |
| agent3 | 9.77 | 4.34 | 3.26 | 6.34 | 5.26 | 0.41 | 0.10 | 3.67 |
| agent3_vad | 9.22 | 4.93 | 3.21 | 6.20 | 5.36 | 0.52 | 0.10 | 3.72 |
| random1_count | 109.07 | 103.97 | 130.85 | 113.75 | 116.57 | 54.27 | 46.15 | 91.72 |
| random1_same | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 0.00 | 0.00 | 0.00 |
| random2_count | 9.37 | 4.13 | 3.45 | 6.45 | 4.92 | 0.41 | 0.10 | 3.51 |
| random2_same | 9.62 | 5.58 | 3.54 | 7.09 | 5.42 | 0.41 | 0.10 | 3.62 |
| random3_count | 10.26 | 4.72 | 4.80 | 7.91 | 5.45 | 0.36 | 0.10 | 17.03 |
| random3_same | 1305.88 | 1386.38 | 1316.78 | 1460.61 | 1224.58 | 1162.72 | 1643.33 | 1243.40 |
| recipe | 9.77 | 4.34 | 3.26 | 6.34 | 5.26 | 0.41 | 0.10 | 3.67 |
| recipe_vad | 9.22 | 4.93 | 3.21 | 6.20 | 5.36 | 0.52 | 0.10 | 3.72 |

## Pass bar

| arm | halluc ≥50% | halluc ≤⅓ | ins ≤ +0.2 | clean ≤ +0.3 | bg/other ≤ +1, drop ≤ +1pt | all |
|---|---|---|---|---|---|---|
| base | no | no | yes | yes | yes | no |
| thresholds | yes | yes | yes | no | no | no |
| thresholds_strict | yes | yes | yes | no | no | no |
| openai_thresholds | no | no | yes | yes | yes | no |
| vad | yes | yes | yes | yes | yes | yes |
| vad_thresholds | yes | yes | yes | yes | yes | yes |
| vad_thresholds_strict | yes | yes | yes | yes | yes | yes |
| head_scan | no | no | yes | yes | yes | no |
| head_scan_vad | yes | yes | yes | yes | yes | yes |
| agent1 | yes | yes | yes | yes | yes | yes |
| agent1_vad | yes | yes | yes | yes | yes | yes |
| agent2 | yes | yes | yes | yes | yes | yes |
| agent2_vad | yes | yes | yes | yes | yes | yes |
| agent3 | yes | yes | yes | yes | yes | yes |
| agent3_vad | yes | yes | yes | yes | yes | yes |
| random1_count | no | no | no | no | no | no |
| random1_same | yes | yes | yes | no | no | no |
| random2_count | no | no | yes | yes | yes | no |
| random2_same | no | no | yes | yes | yes | no |
| random3_count | no | no | no | yes | yes | no |
| random3_same | no | no | no | no | no | no |
| recipe | yes | yes | yes | yes | yes | yes |
| recipe_vad | yes | yes | yes | yes | yes | yes |

## Sessions

```
{
 "1": {
  "pass_": true,
  "pass_strict": true,
  "random_ok": true,
  "random_rel_drops": [
   -0.1785714285714286,
   1.0
  ],
  "random_d_wer": [
   103.788,
   96.616
  ]
 },
 "2": {
  "pass_": true,
  "pass_strict": true,
  "random_ok": true,
  "random_rel_drops": [
   0.0,
   -0.19047619047619047
  ],
  "random_d_wer": [
   -0.09899999999999975,
   0.028999999999999915
  ]
 },
 "3": {
  "pass_": true,
  "pass_strict": true,
  "random_ok": true,
  "random_rel_drops": [
   -0.011904761904761862,
   -0.18654761904761918
  ],
  "random_d_wer": [
   0.19700000000000006,
   1269.0339999999999
  ]
 }
}
```

## Experiment-level criteria

```
{
 "a_beats_thresholds_music_noise_15pt": {
  "gaps_pt": [
   6.2,
   6.2,
   6.2
  ],
  "ok": false
 },
 "a_beats_thresholds_strict_music_noise_15pt": {
  "gaps_pt": [
   -0.8,
   -0.8,
   -0.8
  ],
  "ok": false
 },
 "b_beats_head_scan_10rel_pt": {
  "gaps_pt": [
   94.5,
   94.5,
   94.0
  ],
  "ok": true
 },
 "c_edit_plus_vad_30rel": {
  "cuts": [
   0.946,
   0.946,
   0.946
  ],
  "ok": true
 },
 "sessions_passing": 3,
 "experiment_success": true,
 "head_scan_plus_vad_cut": -0.027575020275750095
}
```
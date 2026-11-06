# Session 3 notes: spurious-feature repair by fc-row channel editing

## What I looked at
- Cached base penultimate (avgpool, 2048-d) features + logits for all 4,250 dev images (work/base_feats.pt).
- Per dev class k: contribution of each channel to logit k on spurious images, w_kj * (mean_spur_j - mean_allclean_j),
  vs the same on k's clean images. The spurious excess logit (7-17) is spread over many channels: top-5 channels carry
  only ~15-30% (50% for snowmobile/snowplow, which share snow channels 256/1051). So cues are distributed directions
  in the 2048-d feature, not single neurons; the cleanest place to cut them per class is fc row k.
- Looked at one spurious image per class (work/spur_grid.png). Cues: foliage/flowers (indigo bunting, hummingbird, bee,
  walking stick, coucal), rocks/bark/water (agama, alligator, platypus, gar, fox squirrel), people/faces/bodies
  (bathing cap = surfer/water, bath towel = baby, bikini, miniskirt, cowboy hat, mortarboard, bulletproof vest = police
  jacket, chain mail/fountain pen), US flag (flagpole), handwriting (quill, fountain pen), snowy forest (snowmobile,
  snowplow), fire (stove), shop-fronts (bakery), gymnastics/sport scenes (balance beam, parallel bars), underwater
  (snorkel), car interior (seat belt), danger sign (padlock), DJ/music (steel drum).
- Shared channels across classes, e.g. 1797/898/1185 (foliage/flowers: bunting, hummingbird, bee), 1510/431/879
  (water/swimmer: bathing cap, snorkel, bikini, bathtub), 1189/1662/1196 (sky/structures: barn, obelisk, radio telescope),
  972/1089 (gym: balance beam, parallel bars), 1654/1197 (handwriting: quill, fountain pen).

## Edit (fc.weight rows of the 50 dev classes + fc.bias of those 50)
For each dev class k, with a_s = mean feature on k's spurious images, a_c = mean on k's 3 clean images:
- cue channels: w_kj > 0 and a_s,j > mean over all clean images. Shrink w_kj by factor alpha * g_j,
  g_j = clip((a_s - a_c)/a_s, 0, 1)  (how much more the channel fires on the cue-only images than on real k photos).
- object channels: w_kj > 0, a_c,j > global mean, boost w_kj by beta * clip((a_c - a_s)/a_c, 0, 1).
- bias: b_k += 0.5 * (w_old - w_new) . a_c  (restore half of the clean-k logit change).
No search: hand-chosen variants only, dose-response over alpha.

## Honest evaluation (offline, on cached features; fc-only so exact)
Spurious: 2-fold (build from 13 spurious images, test on other 12). Clean dev-class: leave-one-out over the 3 clean
images (build from 2, test the 3rd). Other classes' clean images use the full edit.
| variant (alpha,beta,comp) | hit | AP | clean top1 | dev-cls top1 (LOO) |
| base                      | .530 | .533 | .6407 | .633 |
| 0.5,0,0                   | .198 | .553 | .6430 | .560 |
| 1.0,0,0                   | .072 | .589 | .6387 | .433 |
| projection of (a_s-a_c) 1.0| .529 | .353 | .6357 | .487 |  (dense projection does NOT work)
| 0.5,0.15,0.5              | .242 | .570 | .6423 | .607 |
| 0.7,0.3,0.5 (SUBMITTED)   | .182 | .591 | .6413 | .593 |
| 1.0,0.3,0.5               | .122 | .606 | .6413 | .567 |
| 1.3,0.3,0.5               | .093 | .622 | .6383 | .500 |
Monotone dose-response: stronger dose = fewer hits, better AP, more dev-class clean loss. Picked 0.7 as compromise.

## In-sample T.metrics (actual network)
base: clean 0.6407, dev-cls 0.6333, hit 0.530, pk 0.333, AP 0.533
0.5,0.15,0.5: clean 0.6437, dev-cls 0.6333, hit 0.232, AP 0.621
0.7,0.3,0.5 (submitted): clean 0.6447, dev-cls 0.660, hit 0.173, pk 0.125, AP 0.666
1.0,0.3,0.5: clean 0.6447, dev-cls 0.6333, hit 0.118, AP 0.706
In-sample dev-class numbers are optimistic (3 clean images used to build); trust the LOO column instead.

Files: work/edit_<a>_<b>_<c>.pt copies, work/metrics_*.json, work/design2.py / design3.py / build.py.

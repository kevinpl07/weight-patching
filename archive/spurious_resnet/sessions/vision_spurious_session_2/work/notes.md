# Notes (session 2)

## What I looked at
- Cached base penultimate (avgpool, 2048-d) features + logits for all 4,250 dev images (work/base_feats.pt).
- Base dev metrics (computed from the cache, matching T.metrics): clean top-1 0.6407, dev-class clean 0.633, spurious hit 0.530, mean p_k 0.333, AP 0.533.
- Grid of one spurious image per class (work/spur_grid.png). Cues are context or co-occurring things: green foliage (indigo bunting, walking stick), rock (agama), flowers (hummingbird, bee), water ripples (platypus), tree bark (fox squirrel), storefront (bakery), girls in leotards (balance beam), US flag (flagpole), handwriting (quill), Venetian streets (gondola), underwater scenes (snorkel), snowy forest (snowmobile, snowplow), fire (stove), wooden door (padlock), people/faces (bikini, bathing cap, shower cap, mortarboard...).

## Where the cue lives
- The penultimate features are nonnegative and the logit is w_k . f. For each dev class k I compared the mean feature on k's 25 spurious images (mu_s) with the mean on k's 3 clean photos (mu_c).
- Cue channels for k: w_kj > 0, mu_s_j > 1.5 mu_c_j, ranked by w_kj (mu_s_j - mu_c_j). The top 50 of them carry 10-47% of the positive logit k on spurious images (snorkel 0.47, flagpole 0.45, quill 0.36, hummingbird 0.29).
- Sanity check: the clean images (other classes) that most activate snorkel's cue channels are sharks, coral reef, dugong (underwater); for quill they are envelope, menu, ballpoint (handwriting/paper); for flagpole, pinwheel, maypole, parachute (flag/fabric on sky); for hummingbird, butterflies, bee (flowers). So these are context channels, not snorkel/quill objects.

## Edit
- fc.weight only, rows of the 50 dev classes. Per row: zero the top-50 cue channels (alpha=1), and multiply by 1.5 the top-50 "object" channels (w>0, mu_c > 1.5 mu_s, ranked by w(mu_c-mu_s)), which compensates for the clean-k logit lost with the cue.
- 5,000 entries; ||delta|| = 4.27. Only the 50 rows change, so other classes' logits are untouched; a class can only lose/gain via class k's logit.

## Dose-response (local eval, then 3-fold CV: choose channels with 2 of 3 clean photos and 2/3 of spurious, evaluate on the held-out ones)
| variant | dev hit | dev AP | dev-cls clean | CV hit | CV AP | CV dev-cls clean | nondev acc (CV) |
|---|---|---|---|---|---|---|---|
| base | 0.530 | 0.533 | 0.633 | 0.531 | 0.567 | 0.633 | 0.641 |
| zero top20 | 0.084 | 0.643 | 0.520 | 0.087 | 0.633 | 0.487 | |
| zero top50 | 0.049 | 0.687 | 0.500 | 0.049 | 0.663 | 0.420 | 0.650 |
| zero top50 x0.5 | 0.151 | 0.605 | 0.567 | 0.152 | 0.608 | 0.527 | |
| zero50 + bias comp | 0.110 | 0.687 | 0.613 | 0.111 | 0.663 | 0.513 | 0.647 |
| zero50 + boost 1.25 | 0.104 | 0.732 | 0.607 | 0.102 | 0.690 | 0.527 | 0.646 |
| **zero50 + boost 1.5 (chosen)** | 0.165 | 0.752 | 0.727 | 0.164 | 0.711 | 0.620 | 0.640 |
| zero50 + boost 2.0 | 0.253 | 0.792 | 0.767 | 0.277 | 0.732 | 0.707 | 0.612 |
| zero80 + boost 1.4 | 0.141 | 0.767 | 0.707 | 0.129 | 0.717 | 0.587 | 0.642 |
- Pure suppression kills the hit rate but costs dev-class clean accuracy (CV 0.63 -> 0.42). Boosting object channels restores it; 2.0 starts hurting other classes (non-dev acc -3 pt). 1.5 is the balance point.
- In-sample dev-class clean accuracy (0.727) is optimistic because the boosted channels were chosen on those photos; CV (0.62) is the honest number.

## Official T.metrics on saved edit
clean_top1 0.643, clean_top1_devcls 0.727, spur_hit 0.165, spur_pk 0.112, spur_ap 0.752.
Copy: work/edit_m50_g1.5.pt

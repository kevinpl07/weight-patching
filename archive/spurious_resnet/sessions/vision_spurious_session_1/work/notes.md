# Notes: spurious-feature repair of ResNet-50 (session 1)

## What I looked at
- Cached base penultimate (avgpool, 2048-d) features + logits for all 4,250 dev images (work/base_feats.pt). Verified
  logits = F @ fc.weight^T + fc.bias exactly, so any fc-only edit can be evaluated on dev offline at no forward cost.
- Base dev metrics: clean top-1 0.6407, dev-class clean top-1 0.633, spurious hit 0.530, spurious AP 0.533.
- Looked at one spurious image per class (work/spur_a.png). Cues are context: flowers (hummingbird, bee, hamper),
  foliage/branches (indigo bunting, coucal, walking stick), rock (agama), bark (fox squirrel), water ripples (platypus),
  handwriting/paper (quill, fountain pen), US flag (flagpole), snowy forest (snowmobile, snowplow), fire (stove),
  underwater swimmers (snorkel), surfer/water (bathing cap), gymnasts (balance beam, parallel bars), driver in a car
  (seat belt), shop fronts (bakery, shoe shop), Venice street (gondola), DJ/performer (steel drum), people/faces for
  apparel classes (cowboy hat, mortarboard, miniskirt, bikini, chain mail), baby (bath towel).

## Where the cue lives
- At the fc layer, for class k: channels c with w_kc > 0 whose mean activation on k's spurious images is > 2x their
  mean on k's clean images. Ranking by w_kc * (mu_spur - mu_clean) gives "cue channels". The cue channels are the ones
  that most activate for clean images of the context classes (daisy for hummingbird/bee/hamper; barbershop/bookshop
  for bakery; envelope/crossword/menu for quill/fountain pen; ski/snowmobile/dogsled for snowplow; torch/candle for
  stove; hammerhead/brain coral for snorkel; bikini/paddle/canoe for bathing cap; balance beam/horizontal bar for pole).
  See work/cue_report.txt.

## Edit (work/edited/edit.pt; copy work/edit_m32_a075_saved.pt)
- fc.weight only, 50 dev-class rows. Per class: top 32 cue channels scaled by 0.25 (alpha=0.75); top 32 "object"
  channels (w>0, mu_clean > 2x mu_spur) scaled up by (1+beta), with beta chosen in closed form so the mean clean-k
  logit lost from the cue channels is restored (beta capped at 1). No search/fit against a loss.
- Only reduces logit k on cue evidence; other classes' images can only lose k as a competitor, so non-dev classes
  are not hurt (object-channel boost is small).
- 3,169 entries changed, ||delta|| = 2.04.

## Validation (honest held-out within dev: 3 folds, rule built from 2/3 clean + 2/3 spurious, scored on the rest)
| dose | hit | AP | dev-class clean acc |
| base | 0.531 | 0.567 | 0.633 |
| m32 a0.5 +comp | 0.277 | 0.606 | 0.613 |
| m32 a0.75 +comp (chosen) | 0.210 | 0.625 | 0.580 |
| m32 a1.0 +comp | 0.162 | 0.655 | 0.547 |
| m64 a1.0 +comp | 0.144 | 0.669 | 0.533 |
| m32 a1.0 no comp | 0.141 | 0.629 | 0.500 |
Compensation helps AP and dev-class clean accuracy. Clear dose-response; the trade-off is that the cue channels are
also partly used by genuine class-k photos (bees are on flowers), so dev-class clean accuracy drops with dose.

## Measured with T.metrics on the saved edit
clean top-1 0.6437 (base 0.6407), dev-class clean 0.627 (base 0.633), spurious hit 0.234 (base 0.530),
spur p_k 0.153 (base 0.333), spurious AP 0.630 (base 0.533).

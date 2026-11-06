# Notes

## 07:45 start
- Read PROMPT.md and agent_tools.py. Practice: 221 fake (label 1), 89 known (label 0), 410 real-but-unknown (None).
  Almost all prompts are "Tell me about [the film/book] X."; base replies start with "Okay, let's ..." or the name.
- Plan: (1) collect last-prompt activations (fake vs known vs unknown-real) at several layers + on my own paraphrased
  prompts; (2) find a "familiarity" direction (diff of means / probe); (3) see what the model says when forced to start
  with "I" (does it then abstain?); (4) build a gated edit: repurpose one or a few low-use MLP neurons so they read the
  "unfamiliar" direction and write an "abstain" direction; check dose-response and collateral.
- First collect call died with CUDA OOM (GPU busy with another job) but still charged 4 passes. Waiting for GPU.

## 08:58-09:20 (GPU became free at 08:57; clock already past 60 min because of the wait)
- acts1 (last prompt token, 96 fake / 89 known / 71 unknown-real) + 64 own paraphrases (Who is X? / summarize the film X ...).
  Fake-vs-known is linearly readable at the last prompt token from layer ~10 on. Diff-of-means AUC (held-out half)
  0.93-0.95 at mlp_in@18/22; logistic probe 0.97-0.996; on MY PARAPHRASES (direction fit on practice wording only):
  dm 0.93 at L18, LR 0.98 at L18/22 and 1.00 at L26/30. Unknown-real items sit between (mostly on the fake side).
- The last prompt token ("\n" after "model") has a very stable projection on the mean mlp_in direction (9.12+-0.17 at L18,
  12.9+-0.3 at L22) while other tokens vary (std ~3.3): usable as a position detector / bias inside a gated neuron.
- Forcing the reply to start with "I": fakes still mostly fabricate ("I can definitely tell you about..." 15/32;
  "I'm sorry, but there doesn't appear to be..." only 4/32). An honesty instruction in the prompt makes the model say
  "You are absolutely right to question me!" for fakes AND knowns: not a clean abstain source.
- Gated neuron (gate = gain*(LR probe score - theta) via the mean direction as bias, up = mean dir/c (~1 at last
  prompt), down = write vector). Simulated on cached activations: fires on 96-98% of fake last-prompt tokens, 0% of
  known, 62-88% of fake paraphrases, 0% known paraphrases, ~2% of ordinary chat tokens weakly. The classifier part works.
- Write vectors tried (quick check: 24 fake + 24 known practice + 16 paraphrases, regex for abstention):
  v1 L22 amplify fake-known resid dir (x50): 0/32 abstain, replies near-identical.
  v2 L18+L22, post-norm-compensated, x300: 0/32.
  v3 L18+L22 "I can't know this" dir (private/unknowable questions minus answerable twins), x300: 0/32.
  v4 same at L18,22,26 x500: 0/32 (first token changes from "Okay" to the name, still fabricates).
  v5 L30 output-embedding push for "Unfortunately": 0/32. v6 L22,26,30 token push: "unfortunately" leaks into
  fabricated text mid-reply (neuron also fires on reply tokens of fake contexts) - harmful, no abstention.
- Diagnosis (diag.py on v3): the neuron fires as designed (act 2-6 on fakes, 0 on knowns) but Gemma's
  post-feedforward RMSNorm caps one layer's MLP output: the residual moves ~1.8k at L18 (resid norm ~41k), a few k by
  L30. A single gated neuron per layer cannot push hard enough to flip the abstain decision; the abstention behaviour
  needs a stronger / better-chosen write direction than any I found in the time.
- Decision: no edit I tested reduces fabrication on practice or paraphrases, and each has side effects, so I submit
  an EMPTY edit (cleared). Copies v1-v6 kept in work/edits/ for audit.

## 08:58-09:20 (GPU became free at 08:57; clock already past 60 min because of the wait)
- acts1 (last prompt token, 96 fake / 89 known / 71 unknown-real) + 64 own paraphrases (Who is X? / summarize the film X ...).
  Fake-vs-known is linearly readable at the last prompt token from layer ~10 on. Diff-of-means AUC (held-out half)
  0.93-0.95 at mlp_in@18/22; logistic probe 0.97-0.996; on MY PARAPHRASES (direction fit on practice wording only):
  dm 0.93 at L18, LR 0.98 at L18/22 and 1.00 at L26/30. Unknown-real items sit between (mostly on the fake side).
- The last prompt token ("\n" after "model") has a very stable projection on the mean mlp_in direction (9.12+-0.17 at L18,
  12.9+-0.3 at L22) while other tokens vary (std ~3.3): usable as a position detector / bias inside a gated neuron.
- Forcing the reply to start with "I": fakes still mostly fabricate ("I can definitely tell you about..." 15/32;
  "I'm sorry, but there doesn't appear to be..." only 4/32). An honesty instruction in the prompt makes the model say
  "You are absolutely right to question me!" for fakes AND knowns: not a clean abstain source.
- Gated neuron (gate = gain*(LR probe score - theta) via the mean direction as bias, up = mean dir/c (~1 at last
  prompt), down = write vector). Simulated on cached activations: fires on 96-98% of fake last-prompt tokens, 0% of
  known, 62-88% of fake paraphrases, 0% known paraphrases, ~2% of ordinary chat tokens weakly. The classifier part works.
- Write vectors tried (quick check: 24 fake + 24 known practice + 16 paraphrases, regex for abstention):
  v1 L22 amplify fake-known resid dir (x50): 0/32 abstain, replies near-identical.
  v2 L18+L22, post-norm-compensated, x300: 0/32.
  v3 L18+L22 "I can't know this" dir (private/unknowable questions minus answerable twins), x300: 0/32.
  v4 same at L18,22,26 x500: 0/32 (first token changes from "Okay" to the name, still fabricates).
  v5 L30 output-embedding push for "Unfortunately": 0/32. v6 L22,26,30 token push: "unfortunately" leaks into
  fabricated text mid-reply (neuron also fires on reply tokens of fake contexts) - harmful, no abstention.
- Diagnosis (diag.py on v3): the neuron fires as designed (act 2-6 on fakes, 0 on knowns) but Gemma's
  post-feedforward RMSNorm caps one layer's MLP output: the residual moves ~1.8k at L18 (resid norm ~41k), a few k by
  L30. A single gated neuron per layer cannot push hard enough to flip the abstain decision; the abstention behaviour
  needs a stronger / better-chosen write direction than any I found in the time.
- Decision: no edit I tested reduces fabrication on practice or paraphrases, and each has side effects, so I submit
  an EMPTY edit (cleared). Copies v1-v6 kept in work/edits/ for audit.

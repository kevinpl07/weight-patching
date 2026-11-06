# Notes: made-up entity fabrication, gemma-3-4b-it

## What I looked at
- s1: the base model never abstains, even when the user message says "if you don't recognize it, say so". It also
  fabricates for gibberish names. Collected last-prompt resid/mlp_in for 221 fake, 89 known, 74 unknown-real items.
- s2/s5: the last prompt position linearly separates fake from known very well from layer 8 on (logreg ~0.97).
  A no-intercept probe on mlp_in@18/21 (negatives: known items + every token of 26 ordinary chats): held-out fakes 100% > 0,
  knowns 2.6% > 0, chat tokens 0% > 0 (max -0.67, the BOS token). Rephrased fake prompts also score > 0. About 40% of the
  unlabelled real-but-obscure entities score > 0.
- s3: forcing the reply prefix "I" gives "I'm sorry, but there doesn't appear..." on 3/12 fakes and 0/8 knowns.
  Forcing "I'm not familiar" makes the model abstain fluently (also on knowns, so the gate matters).
- "Can't know" direction: mean resid(last prompt) of 24 questions the model answers with "I have no way of knowing"
  minus 24 answerable ones. Logit lens of this direction at L24-30: Unfortunately / sorry / I.
- Key constraint: every Gemma sublayer output is RMS-normed before it is added, so one MLP neuron can write at most
  ~||v||/rms(v/(1+w_postff)) into the residual. At L18/21 that is only ~500 (vs |v| ~3500-6000): a single neuron there
  only changed wording. At L24-30 the cap is 1.5k-4k. Normal MLP output there (pre-norm) is ~5, so the unit dominates.

## The edit (copy `cascade_unf`, = work/edited/edit.safetensors), 46,054 entries, 6 neurons
Each neuron j (the weakest one in its layer by |down col|*|gate row|) is overwritten:
gate row = 4*w, up row = c*w (w = no-intercept probe, s = w.mlp_in > 0 only for made-up entities), so act ~ 4c s^2 if s>0, ~0 else;
down col = target/(1+post_ff_norm_w), unit norm.
- L18 n1239, L21 n2465: target = mean resid(fake) - mean resid(known) ("unknown" amplifier). NOTE: the L24-30 probes were
  fitted on activations collected with these two units active (s7 ran with copy v_unknown_c1 loaded), so they are part
  of a cascade: L18/21 units amplify the unknown signal, the late units read it. Without them the late units stay silent (s10).
- L24 n525, L26 n662: target = "can't know" direction at that layer.
- L28 n1066, L30 n2524: target = (1+final_norm_w)*(E[Unfortunately]-E[I]) (logit-lens push to open with "Unfortunately").

## Results (T.test)
- 128-item sample: fake_fabricate 100 -> 42.9, known_fact_kept 62.2 -> 67.6, known_abstain 0 -> 0.
- remaining 52 knowns + 12 fakes: fake_fabricate 100 -> 33.3, known_fact_kept 71.2 -> 69.2.
- all 89 knowns combined: 67.4 -> 68.5 (60 -> 61 correct); all 103 fakes tested: 100 -> 41.7.
- 64 unlabelled real-but-obscure: unknown_real_abstain 0 -> 25.
- Ordinary chat replies (18 prompts) identical to base in s12/s13 generations; knowns keep their openers.
- Dose: L18/21 units only -> no abstention; + L24-30 "can't know" units -> opener "I", 2/12 fakes abstain;
  + L28/30 re-aimed at "Unfortunately" -> ~58% abstain. Late up rows x0.1 -> 41.8 (saturated: gate is quadratic in s
  and s ~6 for fakes, so the dose knob is the number/targets of units, not c).
- Remaining failures: reply opens "Unfortunately, information about X is scarce... here's what we know" and then fabricates.

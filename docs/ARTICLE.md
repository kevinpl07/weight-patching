# One Neuron, No Training: What Editing an LLM's Weights Is Actually Good For

*{{DATE}} · code and every number: github.com/{{REPO}}*

In November, *Sparks of Weight-Space Programming* showed a frontier LLM reading the raw weights of a neural network
and writing new values into them: no optimizer, no gradient descent, just reasoning about matrices. The internet
split in two within a day.

- **"This is the beginning of the end."** Models that can rewrite weights can rewrite themselves; recursive
  self-improvement just skipped a step.
- **"This is a nothing burger."** It's steering and ablation in a sandbox; the headline result fixes a repetition bug
  that RL would fix anyway.

We spent a week running experiments on exactly this question:
- **{{N_SESSIONS}} agent sessions, 11 experiments, 9 models**, from a 37k-parameter toy to Whisper and 4B chat models.
- **The editing agents are Claude Opus.**
- **Then we rebuilt the useful part as a small open framework** and reran it against the baseline both camps forget:
  ordinary fine-tuning.

**Our answer: both camps are wrong, in instructive ways.**

> **Weight editing is a patch, not a superpower.** If you can say precisely what is wrong, a direct edit of a single
> neuron can fix a real failure in a frozen model, with fewer side effects than a prompt and often fewer than
> fine-tuning. But the edit doesn't survive training that pushes against it, it only works where the model already
> "knows" the answer, and given enough examples, training data matches or beats it.

---

## 1. Reading is real

Across every experiment, the agent correctly found what the network was doing, without being told where to look:
- the Fourier circuit in a grokking network;
- a "hedging" direction in Qwen;
- a missing induction circuit in a small transformer;
- uniform attention in a ViT;
- context channels behind spurious features in ResNet-50;
- a "there's no speech here" signal inside Whisper.

This is the part the "nothing burger" camp undersells. Blind sessions, given nothing but the weights and a few
examples, converged on the same mechanism 3 times out of 3 (Whisper, language mixing, ResNet).

## 2. Writing doesn't survive training

This is the part the doom camp gets wrong. We asked the obvious next question: **if an LLM edits a network and
training then continues, what's left of the edit?**

{{FIG: grokking "loan" chart}}

| Experiment | What the edit gave | What was left after more training |
|---|---|---|
| Grokking MLP (3 pilots, 24 sessions) | reached 50% held-out 1.8–3.4× sooner | never reached 95%; plain training did; random edits did better long-run |
| Qwen3.5-0.8B, then RL | +7 points at step 0 | gone within ~10–15 GRPO steps |
| Small LM stuck on a plateau | ~650 steps ahead (biggest effect we saw) | gap 0.006 nats at the end: noise |
| ViT locality prior | the structure survived | accuracy level with no edit (−0.1 points) |

The mechanism is mundane:
- **Gradient descent rebuilds whatever the loss wants.**
- **AdamW moves every weight at full speed.**
- **Weight decay erases what no gradient defends.**
- **A dead neuron stays dead.**

The result is a ratchet: **the benefit of an edit fades, the damage stays.** An edit lasts only if it moves the
network somewhere the optimizer was going anyway. That makes it, at best, a head start.

A model that edits its own weights mid-training would be fighting its own optimizer, and the optimizer wins.

## 3. But as a patch on a frozen model, it works

So we removed the optimizer. Most deployed models are frozen: nobody is training them while you use them. And they
have specific, annoying, observable failures.

**The recipe** (the agents converged on it independently for Whisper; we generalised it):

{{FIG: one-neuron diagram: read → fire → write}}

1. **Find the signal.** A linear probe on the model's own activations at the decision point ("this is the bad case").
   In every success, the model already knew: probe AUC ≈ 0.99.
2. **Find the write target.** Force a few candidate first tokens and keep the one after which the model finishes the
   job by itself. "Sorry" → "…there doesn't appear to be a widely known…".
3. **Rewrite one neuron.** The least-used neuron in one MLP layer becomes a detector:
   - it reads the probe direction;
   - it fires only above a threshold no ordinary token reaches;
   - when it fires, it pushes the logits toward the target.

   **About 7,700 numbers out of 4 billion.**

{{FIG: anatomy histogram: neuron score on made-up vs real entities vs ordinary tokens; threshold line}}

### The scoreboard

{{FIG/TABLE: scoreboard from results.json: base / prompt / gate / random / LoRA × tasks}}

| Failure (model) | Base | System prompt | **One neuron** | Random neuron | LoRA (same data) |
|---|---|---|---|---|---|
| Invents bios for made-up people (Gemma-3-4B) | 98% | 7% *(but real-entity facts: 57% → 20%)* | **16%** *(facts kept: 56%)* | 97% | 30% *(facts: 56%)* |
| Agent runs commands hidden in a web page (Qwen3-4B) | {{}} | {{}} | **{{}}** | {{}} | {{}} |
| Ends answers with "Would you like me to…" (Gemma-3-1B) | {{}} | {{}} | **{{}}** | {{}} | {{}} |
| Transcribes words in silence and music (Whisper) | {{}} | {{thresholds}} | **{{}}** | {{}} | {{}} |

**What else did each fix break?** {{FIG: side-effect chart: chat KL per arm (log scale) + "kept" metric}}

- **The prompt is blunt.** "If you don't recognise someone, say so" nearly stops the fabrication. It also makes Gemma
  deny knowing Chance the Rapper: correct facts about real entities fall from 57% to 20%. It doesn't make the model
  know less; it makes it *say* less.
- **The gate is surgical.** It fires on 98% of held-out made-up names, on 0% of ordinary chat tokens, and leaves real-entity
  facts where they were. Its KL on ordinary chat is 0.0002, against 0.16 for a LoRA fine-tune trained on the same
  prompts. {{pareto: is that just LoRA trained too hard?}}
- **The random neuron does nothing useful.** It's the same size of change in random directions, so the effect isn't
  "any perturbation".

{{EXAMPLES: before/after cards per task; whisper audio players}}

## 4. Training data still wins (or ties)

The fair competitor to an edit isn't a prompt; it's a small fine-tune on the same examples. So every task also gets a
LoRA trained on exactly the practice prompts the gate saw, with targets the model wrote itself.

{{FIG: data-scaling: failure rate vs number of practice examples, gate vs LoRA, 3 seeds}}

{{TEXT: what the sweep shows: at n≈{{}} LoRA matches the gate; with all data it {{}}; the gate's side effects stay
{{}}× lower; the gate needs {{}} examples to …}}

This matches the earlier ResNet experiment: there, the agents' edit beat naive fine-tuning, but a standard method
(DFR, balanced last-layer retraining with ~40 labelled photos per class) passed the bar where the edit didn't.

## 5. Does the patch survive more training?

{{FIG: persistence curves: fabrication vs fine-tuning steps for base+chat, gate+chat, gate+conflict}}

{{TEXT: unrelated fine-tuning {{keeps/erases}} the patch; training that rewards the failure erases it within {{}}
steps. Same lesson as section 2, from the other side.}}

## 6. Where it doesn't work

- **No signal, no gate.** Gemma caves 80% of the time when asked "Are you sure?". But every reply, caving or not,
  starts with the same ritual apology, and its sense of "I was right" is weak (probe AUC ≤ 0.70). There's nothing to
  route.
- **No failure, no gate.** Gemma-3-4B obeys injected *words* but almost never injected *tool calls* (1.7%). Nothing
  to fix.
- **Entangled behaviour.** Phi-3's over-refusal is the same refusal direction firing more weakly. Small edits halve
  false refusals but cost 5–10 points of legitimate refusals. They're still far more selective than abliteration
  (+16 to +33 points at matched false-refusal), but they don't meet the bar.
- **Generalisation follows variety.** An injection gate fit on 3 attack goals learned "about to say BANANA42"; fit on
  15 goals it blocked 43% of unseen ones.

## 7. So who's right?

**Doom camp:**
- **Right:** models really can read weights. They found every mechanism we planted or stumbled on.
- **Wrong:** writing is a loan the optimizer collects, and a patch only works where the model already computes the
  right answer. That isn't self-improvement; it's rerouting.

**Nothing-burger camp:**
- **Right:** much of it is classic interpretability, steering and ablation.
- **Wrong:** "classic" doesn't mean useless. A single-neuron patch, found with an afternoon's work or a blind agent
  session, fixed four real failures with near-zero collateral damage. It's auditable: you can say exactly when it
  fires and what it writes. And it ships as 7,700 numbers in a checkpoint, not a prompt that can get dropped or
  diluted.

**What we'd actually do:**
- **A narrow, visible failure, a frozen model, little data:** try the gate. Check the signal first; it takes minutes.
- **Plenty of examples and the ability to fine-tune:** use them. Data generalises further.
- **Never** expect a hand-written edit to survive training that disagrees with it.

## Run your own

{{QUICKSTART + link to docs/ADDING_A_TASK.md: one file per failure}}

## Caveats

- **Thin statistics.** Greedy decoding, one sample per prompt, held-out sets of a few hundred items. Effects of a few
  points are noise.
- **Rough scoring.** It's regex-based and spot-checked by reading, with no human or LLM judge.
- **Small models.** 1B–4B models, one per task, plus one transfer run.
- **No adaptive attacker** against the injection gate.
- **One agent model** for the editing sessions.

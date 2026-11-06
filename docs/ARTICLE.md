# One Neuron, No Training: What Editing an LLM's Weights Is Actually Good For

*{{DATE}} · code, data and every number: github.com/kevinpl07/weight-patching*

In November, *Sparks of Weight-Space Programming* showed a frontier LLM reading the raw weights of a neural network
and writing new values into them: no optimizer, no gradient descent, just reasoning about matrices. The internet split
in two within a day.

- **"This is the beginning of the end."** Models that can rewrite weights can rewrite themselves. Recursive
  self-improvement just skipped a step.
- **"This is a nothing burger."** It's steering and ablation in a sandbox. The headline result fixes a repetition bug
  that RL would fix anyway.

We spent a week on exactly this question. We ran {{N_SESSIONS}} agent sessions with Claude Opus as the editing agent,
across 11 experiments, from a 37k-parameter toy to Whisper and 4B chat models. Then we rebuilt the useful part as a
small open framework and reran it against the baseline both camps forget: **ordinary fine-tuning on the same examples.**

> **Weight editing is a patch, not a superpower.**
>
> - **It works when you can say precisely what's wrong.** A one-neuron edit fixes a real failure in a frozen model, and
>   it leaves 98% of the model's other answers word-for-word unchanged.
> - **It only works where the model already "knows".**
> - **Training data still wins on the failure itself.** Given enough examples, fine-tuning fixes more, and it changes
>   everything else as well.
> - **The patch doesn't survive further training.** Fifty steps of unrelated fine-tuning erase it.

---

## 1. Reading is real

In every experiment, the agent correctly found what the network was doing, without being told where to look:
- the Fourier circuit in a grokking network;
- a "hedging" direction in Qwen;
- a missing induction circuit in a small transformer;
- uniform attention in a ViT;
- context channels behind spurious features in ResNet-50;
- a "there's no speech here" signal inside Whisper.

Blind sessions, given nothing but weights, tools and a few examples, converged on the same mechanism 3 times out of 3
for Whisper, for language mixing and for ResNet. This is the part the "nothing burger" camp undersells.

## 2. Writing doesn't survive training

This is the part the doom camp gets wrong. If an LLM edits a network and training then continues, what's left of the
edit?

{{FIG: grokking "loan" chart}}

| Experiment | What the edit gave | What was left after more training |
|---|---|---|
| Grokking MLP (3 pilots, 24 sessions) | reached 50% held-out 1.8–3.4× sooner | never reached 95%; plain training did, and random edits did better long-run |
| Qwen3.5-0.8B, then RL | +7 points at step 0 | gone within ~10–15 GRPO steps |
| Small LM stuck on a plateau | ~650 steps ahead (the biggest effect we saw) | 0.006 nats at the end: noise |
| ViT locality prior | the structure survived | accuracy level with no edit (−0.1 points) |

**The mechanism is mundane:**
- **Gradient descent rebuilds whatever the loss wants.**
- **AdamW moves every weight at full speed.**
- **Weight decay erases what no gradient defends.**
- **A dead neuron stays dead.**

The benefit of an edit fades and the damage stays. A model that edits its own weights in the middle of training is
fighting its own optimizer, and the optimizer wins.

## 3. But as a patch on a frozen model, it works

Most deployed models are frozen, and they have specific, annoying, observable failures. So we dropped the optimizer
and used one recipe everywhere. It's the mechanism the blind agents converged on for Whisper, generalised.

{{FIG: one-neuron diagram: read → fire → write}}

1. **Find the signal.** A linear probe on the model's own activations at the decision point ("this is the bad case").
   Where the patch worked, the model already knew: probe AUC 0.99–1.0.
2. **Find the write target.** Force a few candidate first tokens and keep the one after which the model finishes the
   job by itself. "Sorry" → "…there doesn't appear to be a widely known…".
3. **Rewrite one neuron.** The least-used neuron of the last MLP layer becomes a detector:
   - it reads the probe direction;
   - it fires only above a threshold no ordinary token reaches;
   - when it fires, it pushes the logits toward the target.

   That's 1,500–7,700 numbers out of up to 4 billion. Nothing is trained.

{{FIG: anatomy histogram: the neuron fires on 98% of held-out made-up names and on 0% of ordinary chat tokens}}

### The scoreboard

Held-out data with new phrasings. Every fix sees only practice data. "LoRA" is a fine-tune on exactly the same
practice prompts.

| Failure (model) | Base | Prompt / decoding fix | **One neuron** | Random neuron | LoRA, same data |
|---|---|---|---|---|---|
| Invents bios for made-up people (Gemma-3-4B) | 98% | 7%, **but correct facts on real people 57% → 20%** | **16%** (facts kept 56%) | 97% | 30% (facts 56%) |
| Agent runs commands injected in a web page (Qwen3-4B) | 26% | 12.5% | **2.5%** | 26.5% | 0% |
| Ends answers with "Would you like me to…" (Gemma-3-1B) | 59% | 29%, **but answers 41% shorter** | **23%** (content kept 99.6%) | 62% | 8% (content kept 96%) |
| Transcribes words in silence, noise and music (Whisper-small) | 84% | 50% default thresholds / 7% tuned (drops 2% of real speech) | **0.3%** (every speech transcript identical) | 84% | 0.3% (25% of speech transcripts changed) |

**What did each fix break?** {{FIG: side-effect panel: % of ordinary chat answers unchanged + chat KL, per arm}}

- **The patch barely touches anything else.**
  - **Gemma:** 98% of ordinary chat answers come out word-for-word identical with the gate in place.
  - **Qwen agent:** every genuine tool call is byte-identical.
  - **Whisper:** every real transcript is identical.
- **The fine-tune changes everything.** With the LoRA trained on the same prompts, 0% of ordinary Gemma answers stay
  identical: not after 300 steps, and not after 10. Chat KL is 500–10,000× the gate's. It also changes 28% of genuine
  agent replies and 25% of Whisper's speech transcripts.
- **The prompt is blunt.** "If you don't recognise someone, say so" nearly stops fabrication, and also makes Gemma deny
  knowing Chance the Rapper. "Don't add sign-offs" halves sign-offs by cutting answers 41% shorter.
- **The random neuron does nothing,** so it isn't "any perturbation helps".

{{EXAMPLES: before/after cards per task; whisper audio players}}

## 4. Training data still wins on the failure itself

Is "the gate beats LoRA" just a weak LoRA? We tested it three ways on the entity task.

{{FIG: data-scaling: failure vs number of practice examples, gate vs LoRA, 3 seeds}}

1. **Little data: the gate wins easily.** With 8 labelled examples, the gate already stops the made-up-entity
   failure (14%); LoRA stays at 61%. What the gate gains from more data isn't recall, it's precision. With 8 examples
   it also says "Sorry" to a third of real entities; with 300 it doesn't. On sign-offs the gate is even better with *less* data: 5–10% at 16–64 examples vs
   22% with all of them, because its threshold rule gets stricter as practice data grows. That's a weakness of our
   recipe worth fixing.
2. **More data: LoRA wins on the headline.** Adding 1,400 more labelled practice prompts takes LoRA from 30% to
   8–12% fabrication, with the same phrasing or with varied phrasings. The gate stays at ~15% however much data it
   gets. On sign-offs (8% vs 23%) and agent injection (0% vs 2.5%), LoRA also wins
   outright with the practice set we had.
3. **A gentler LoRA doesn't close the side-effect gap.** Stopped at its best point (step 100), LoRA ties the gate
   exactly (15.5% fabrication, 56% facts kept). It still changes 100% of ordinary answers, and its KL never falls
   below ~0.075 (the gate's: 0.0002).

{{FIG: pareto: failure vs side effect, gate at 5 strengths vs LoRA at 6 checkpoints}}

This matches our earlier ResNet experiment, where a standard method (DFR, balanced last-layer retraining) beat the
agents' edit given ~40 labelled photos per class. **If you have the data and can tolerate a model that's different
everywhere, train.** If you need *this* behaviour fixed and *nothing else* touched, the patch is the better tool.

## 5. The patch doesn't survive training

What happens to the patch when someone keeps fine-tuning the model? We fine-tuned the last four layers on ordinary
chat data, the kind of thing any downstream user does.

{{FIG: persistence: fabrication vs fine-tuning steps; gate weights' cosine stays 1.000}}

- **The gate is gone within 20 steps.** After 10 steps of unrelated chat fine-tuning, the neuron fires on 11% of
  made-up names, down from 100%; after 20 steps, on none. Fabrication goes 16% → 98%. With training that rewards
  fabrication, it's the same.
- **The neuron itself is untouched:** its weights keep a cosine similarity of 1.0000 with the patch. The layers
  below it moved, and it no longer receives the signal it was built to read.
- **The LoRA fix doesn't survive either. It just dies slower:** half gone after 50 steps (29% → 72%), fully gone by
  100.

Narrow fixes, hand-written or trained, are fragile under further training, and the hand-written one is the more
fragile of the two. It's section 2 from the other side: an edit lasts exactly as long as nobody trains the model.

## 6. Where it doesn't work

- **No signal, no gate.** Gemma caves 80% of the time when asked "Are you sure?". But every reply, caving or not,
  starts with the same apology ritual, and its sense of "I was right" is weak (probe AUC ≤ 0.70). There's nothing to
  route.
- **No failure, no gate.**
  - **Qwen3-4B** rarely invents entities (14%), and a system prompt takes that to 4%. Transplanting the same gate
    recipe made it *worse* (25%).
  - **Gemma-3-4B as an agent** almost never runs injected commands (1.7%).
- **Not every failure is one decision.** Chinese words leaking into German, Russian and Thai answers from a reasoning
  model can strike at any token.
  - **The one-neuron gate** caught only 31% of leak onsets on practice and didn't reduce the leak.
  - **The edit the blind agents found** is different: a context-gated penalty on the Chinese-token output rows. It
    takes leaking from 15% to 0.3% with KL 0.002. Weight editing isn't one recipe.
- **Entangled behaviour.** Phi-3's over-refusal is the same refusal direction firing more weakly. Small edits halve
  false refusals but cost 5–10 points of legitimate refusals. That's far more selective than abliteration (+16 to
  +33 points at matched false refusal), but still not clean.
- **Coverage follows variety.** The agent-injection gate missed one attack phrasing entirely ("Step 1 / Step 2"). When
  a tiny practice set happened to contain no genuine tool requests, it blocked those too.

## 7. So who's right?

**Doom camp:**
- **Right:** models really can read weights. They found every mechanism we planted or stumbled on, unaided.
- **Wrong:**
  - Writing is a loan the optimizer collects.
  - A patch only routes a signal the model already computes.
  - 50 steps of anyone's fine-tuning erase it.

  That's not self-improvement; it's a sticky note on a frozen model.

**Nothing-burger camp:**
- **Right:** much of it is classic interpretability, steering and ablation.
- **Wrong:** "classic" doesn't mean useless. One neuron:
  - fixed four real failures across text, agents and audio;
  - left 98–100% of everything else byte-identical;
  - is auditable, because you can say exactly when it fires and what it writes;
  - ships as a few thousand numbers in a checkpoint.

**What we'd actually do:**
- **A narrow, visible failure, a frozen model, little data, and "don't touch anything else":** try the gate. Check
  the signal first; it takes minutes.
- **Plenty of examples, and side effects are acceptable:** fine-tune. Data wins on the failure.
- **Never** expect a hand-written edit to survive training.

## Can an LLM find the patch by itself?

The recipe above was written by a human, using what earlier blind agent sessions found for Whisper. So we gave
Claude Opus the made-up-entity task blind. It got only the practice data, tools to read activations and write weights,
no gradient descent, a 30-pass budget and about an hour.

- **Session 1 read the model correctly but couldn't write a fix.**
  - It found the "I don't know this entity" signal on its own (AUC 0.97–1.0) and built a detector neuron that fires on
    96–98% of made-up names and 0% of real ones.
  - Six attempts to make that neuron change the answer, writing into middle layers, did nothing.
  - It submitted an empty edit and explained why: Gemma re-normalises every MLP output, so one middle-layer neuron can
    barely move the residual stream. (This session also lost ~70 minutes waiting for a GPU we had double-booked.)
- **Session 2, with a dedicated GPU, built a working edit:** a 6-neuron, two-stage cascade.
  - Two neurons at layers 18–21 amplify the "unknown entity" signal.
  - Four neurons at layers 24–30 write "I can't know this", then "Unfortunately…".
  - On held-out data, fabrication fell from 98% to 68%. Facts about real entities stayed at 56% and MMLU was unchanged. But
    it was less surgical: chat KL 0.011, and 95% of ordinary chat answers came out with some wording changed (the
    recipe gate: 2%). Its matched random edit did nothing (99%).
  - It had predicted 40–60%; its gates generalised less well to the new phrasings than it expected.

**Verdict on "LLMs can now patch weights":** yes, partly, and the reading is the strong part. Unaided, Opus found the
signal immediately, and in one of two sessions engineered a real, surgical fix. But it got a third of the way, where the
human-tuned recipe (find the opener the model completes by itself, write it at the last layer) gets 84% of the way. It
also took an hour and ~170k tokens per attempt, for a fix that 10 steps of anyone's fine-tuning will erase.

## Run your own

```bash
git clone https://github.com/kevinpl07/weight-patching && cd weight-patching && pip install -r requirements.txt
python -m patchkit.run all entity        # data → base → probe → one-neuron gate + random → LoRA → eval
python -m patchkit.run sweep entity      # gate vs LoRA as data grows
```

We checked that this works for a stranger: a fresh copy of the repo, run on a different GPU, reproduces the entity
numbers within noise (`results/entity/REPRODUCTION.md`). A new failure is one file: a model, practice and held-out prompts, a label function and a scorer
([docs/ADDING_A_TASK.md](docs/ADDING_A_TASK.md)). Blind agent sessions: [docs/AGENT_SESSIONS.md](docs/AGENT_SESSIONS.md).

## Caveats

- **Thin statistics.** Greedy decoding, one sample per prompt, held-out sets of 300–1,700 items. Differences of a few
  points are noise.
- **Rough scoring.** It's regex-based and spot-checked by reading, with no human or LLM judge. We found and fixed one
  scorer bug (curly apostrophes) during this run; it had overstated the system prompt's fabrication rate in our
  earlier report.
- **Small models.** 1B–4B models, one per task, plus transfers.
- **No adaptive attacker** against the injection gate.
- **One agent model** for the editing sessions.
- **Fixed LoRA recipe** (r=16, 3 epochs, not tuned on held-out data), plus the step and data sweeps above.

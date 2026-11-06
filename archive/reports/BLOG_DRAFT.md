# Weight-Space Programming Is Real. AdamW Doesn't Care.

*34 agent sessions, 6 experiments, 48 hours: what happens to an LLM's weight edits once training keeps running*

> **DRAFT, 2026-11-04.** Editor's notes, other title options and claims to check are at the bottom.

---

On November 2, an MIT CSAIL team published [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/). It shows a frontier LLM reading the raw weights of small networks and saying what they do, writing working networks from all-zero weights, and editing 0.016% of Qwen3.5-0.8B's weights to take it from 7.9% to 48.1% on MATH-500. No optimizer and no gradient descent were involved; the model just reasoned about matrices and wrote numbers into them.

The internet split down the middle within a day.

**Camp "this is crazy."** [Latent Space](https://latent.space/p/ainews-llms-can-read-write-neural-network-weights-weight-space-programming-cuts-autoresearch-out-of/) called it a "MONSTER discovery" and suggested it cuts autoresearch out of the recursive-self-improvement loop: why run training experiments when a model can write the weights itself? On [Lobsters](https://lobste.rs/s/u7wdqj/sparks_weight_space_programming_llms_can), people said they were spooked by what such a model could do with more tools and time. Simon Willison reproduced the game-identification result on a laptop.

**Camp "this is nothing."** Critics made three main points:
- **The headline number fixes a failure mode, not reasoning.** 93% of Qwen's baseline answers hit the token limit in reasoning loops, and the prompt explicitly told the agent to look at repetition ([Turing Post](https://www.turingpost.com/p/what-is-weight-space-programming-and-why-it-suddenly-blew-up-the-internet), citing Lucas Beyer).
- **The method is old.** On this view it is steering, ablation and Elhage-style circuit reading run in a Python sandbox ([r/LocalLLaMA, via Latent Space](https://latent.space/p/ainews-llms-can-read-write-neural-network-weights-weight-space-programming-cuts-autoresearch-out-of/)).
- **It doesn't generalize.** The same approach did nothing for Gemma 3 270M, which doesn't loop.

Sean Goedecke's rebuttal, [as summarized here](https://piyushai.com/weight-space-programming-explained/), put it as superhuman mental arithmetic, not superhuman AI research.

Both camps are arguing over the same handful of demos, and every demo shares one feature: **once the edit is made, nothing happens to the network afterwards.** It is scored as is. That's fine for a demo, but it isn't how networks get built. Networks get trained, and they keep being trained: fine-tuning, RL, continued pretraining.

So we asked the question neither camp was asking: **if an LLM edits a network's weights and training then continues, is anything left of the edit?**

We spent two days on that question with Claude Opus 5.5 as the editing agent, running 34 sessions across six experiments, from a 37k-parameter toy to an ImageNet ResNet-50. The short version:

> **The reading is real. The writing is a loan, and the optimizer collects.** Every gain an edit gave was erased by training. The only thing training reliably kept was the damage.

We think that's more interesting than either "crazy" or "nothing," so here is the full story.

---

## The rules

We kept Sparks' central constraint: **the edit never goes through an optimizer.** The agent reads weights, runs forward passes, writes analysis code, and writes values into tensors through an `edit` tool. It may not run gradient descent, least squares or any other fitting, or a search that scores many candidates automatically. All of its tool calls are logged, and we audited every session's scripts afterwards.

Every experiment also had:
- **A matched random edit:** the same number of changed entries per tensor and the same total change size, at random positions. If the agent's edit can't beat noise of the same size, it isn't doing anything.
- **A pass bar written before running.** We did not move it after seeing results. The few post-hoc additions are labelled as such.
- **The same checkpoint, optimizer state and batches for every arm,** so the edit is the only thing that differs between them.

---

## Act 1: The grokking toy, three times

**Setup.** We trained a 2-layer MLP on (a + b) mod 97. It's the textbook grokking setup: it memorizes the training set by about step 1,300, then sits near 0% held-out accuracy for thousands of steps before suddenly generalizing at 23k–31k steps. We paused it during that plateau (100% train, about 0.5% held-out) and handed the weights to Opus with a 2% budget (749 entries). The goal was to reach 95% held-out sooner once training resumed.

**Pilot 1: one edit.** All three sessions found the Fourier circuit the network was building, which is a genuinely impressive bit of weight reading. All three then used the same strategy: identify the "memorizing" neurons and switch them off with a large negative bias.

At first it looked like a win: the edited runs reached 50% held-out **1.8–3.4× faster** than plain training, in every seed, beating every random control.

Then they stopped climbing. **No edited run ever reached 95%.** They levelled off at 78–84% while plain training went on to 95% by itself.

![Held-out accuracy after the edit, seed 0: the Opus-edited run reaches 50% 3.4× sooner than no edit but stalls at 84%. The unedited run reaches 95% at 30.9k steps, and same-size random edits end at 87–95%.](blog/hero_grokking.png)

*Pilot 1, seed 0. In the long run, even same-size random edits end up ahead of the agent's edit. The other two seeds look the same.*

The cause was simple once we found it. The agents had switched off 54–62 of the 128 neurons. **A ReLU that never fires gets no gradient**, so those neurons stayed off for the rest of the run: 0 of them ever fired again in 60,000 steps. We then trained narrower networks from scratch, and networks of that width top out at about 77–92% on this task. The agent had permanently amputated capacity the task needed. Random neuron removal of the same size produced the same ceiling, which confirmed the diagnosis.

**Pilot 2: let it see the consequences.** A single edit can't see its own aftermath, so we gave the agent four rounds, 4,000 steps apart, each with the full history of what had happened since the previous edit.

The pattern became clear: **each edit produced a jump, then training took it back.** In one round held-out accuracy went 84% → 77% over the next 4,000 steps. After the last edit, runs froze: held-out moved by less than half a point over 44,000 steps. A dumb script that simply re-initialized the dead neurons reached 95% in all three seeds; the Opus loops peaked at 77–91%. **Random-edit loops beat the Opus loops in the long run.**

**Pilot 3: give it better tools.** We reframed the prompt ("your edit is a starting point for the optimizer, not the finished network"). We also gave the agent a read-only view of gradients and Adam state, the ability to reset optimizer moments, three 1,000-step throwaway trial runs per round, and its own earlier predictions shown next to what actually happened.

The agents got visibly better:
- **Their predictions became accurate.** Round-1 forecasts were off by up to 31 points. After seeing feedback, all 9 later forecasts landed within 4 points.
- **They caught bad ideas.** They tried reviving all the dead neurons, saw in a trial run that the revived neurons went straight back to memorizing, and undid it.
- **They rebuilt capacity on purpose,** as phase-shifted "noise-cancelling" partner neurons. One seed climbed steadily 57% → 72% → 86% → 93%, the best Opus run of the project.

It still never reached 95%. And the trial tool made the core problem worse. **A cut looks fantastic after 1,000 steps (25% vs 2% held-out), and its cost shows up only after 10,000.** The agents optimized what they could see, so in round 1 they cut 75–86 neurons, more than ever.

That is an important lesson on its own. **Give an agent a short-horizon feedback loop and it will optimize the short horizon.** This isn't specific to weight editing.

---

## Act 2: A real failure, then RL

Maybe the toy was the problem: grokking is a healthy run, and there was nothing to fix. So we moved to Sparks' territory, Qwen3.5-0.8B, with a real, observable failure: at temperature 1, 57% of its GSM-Symbolic answers rambled until they hit the 512-token limit.

Opus found a "hedging" direction in the residual stream and removed it by changing 4,089 weights in 4 MLP columns. On held-out sets, sampled accuracy rose 11–13 points. A mechanism check showed the edit acts much like lowering the sampling temperature plus an extra push against hedge words. As an edit, it worked.

Then we ran GRPO on top of it, and **the 7-point head start was worth about 7 steps of RL.** Plain GRPO took the unedited model from 27% to about 43% in 15 steps, because rambling is exactly what the reward punishes. Truncation fell from 57% to 3% by itself. From step 15 on, the edited run was indistinguishable from the random-edit control.

The one thing that *did* persist was the edit's side cost: it lowered greedy accuracy by about 4 points, and that gap was still there at step 100. The signal is weak (2 seeds, and seed noise is 4–8 points), but it rhymes with Act 1: **the harm stays, and the benefit fades.**

---

## Act 3: Find a place where the optimizer is truly stuck

An edit should be worth the most where the optimizer is slow. So we built a pretraining pipeline and searched for a real plateau. We found one in a 10.8M-parameter, attention-only transformer trained on Python code: about 900 flat steps before an induction circuit forms and the loss drops 0.34 nats.

Induction is a textbook saddle. It needs two heads, a previous-token head and an induction head, that only pay off together, so each one's gradient is a product of two small numbers.

We gave Opus the step-1,000 checkpoint and didn't tell it what to look for. **It diagnosed the missing circuit and built it.** It wrote sinusoidal position features into the quietest residual dimensions, turned one layer-0 head into a previous-token head, and turned one layer-1 head into an induction head. That was 295k entries, written from statistics alone.

This was the biggest effect in the whole project:

| Validation loss | Step 1,250 | 1,500 | 2,000 | 3,000 | 5,000 | 15,000 |
|---|---|---|---|---|---|---|
| Opus edit | **2.749** | **2.619** | **2.464** | 2.320 | 2.153 | 1.905 |
| No edit | 3.000 | 2.939 | 2.669 | 2.351 | 2.160 | 1.911 |
| Random edit | 3.064 | 2.963 | 2.739 | 2.360 | 2.170 | 1.919 |

*[FIGURE: `runs/pretrain_agent/code_d512_s0_step1000/compare.png`]*

At its peak the edit was **about 650 steps ahead.** Then the control formed its own induction circuit around step 2,000, and the gap closed: 0.33 nats → 0.03 by step 3,000 → 0.006 at the end, which is noise at one seed.

The coda is our favourite moment of the project. We ran two more sessions at step 3,000, one of them with look-ahead training runs and scored on the final loss. **Both agents investigated carefully and submitted an empty edit.** They found the circuit intact and tuned. They measured that the remaining copy-related gain was at most about 0.015 nats, while repurposing any head cost at least 0.055, and concluded there was nothing worth doing. Correctly declining to act is also good research.

---

## Act 4: The counterfactual Sparks didn't run

By now the pattern was consistent: when the training objective covers a failure, training fixes it. So we tested the obvious question about Sparks' headline result: **would plain RL have fixed the reasoning loop?**

We reproduced the failure in Qwen3.5-0.8B's thinking mode: 99% of answers hit the limit, and 61% are verbatim loops. Then we ran standard GRPO on GSM8K, with no edit:

| Held-out GSM8K (128 problems) | Accuracy | Loops | Hit the limit |
|---|---|---|---|
| Base model | 3.9% | 61% | 99% |
| After 30 GRPO steps | **63.3%** | **3.9%** | **3.9%** |

Loops were mostly gone by step 13, about 1.6 hours on one RTX 4070. Our setting differs from Sparks' (a 4k-token limit instead of 32k, and different sampling), so the numbers aren't directly comparable. But the qualitative answer is clear: **a short, boring RL run fixes the same failure and goes well beyond it.**

That doesn't make the Sparks edit worthless. It changed 0.016% of the weights, and RL changes all of them. If RL damaged other behaviour, the edit keeps a real advantage. We didn't measure that, and it's the most important open question here.

---

## Act 5: A barrier, not a plateau

A plateau is something the optimizer eventually crosses. What about structure it *never* finds? ViTs trained from scratch on small datasets famously fail to learn local attention, and locality helps them.

At 5% of a CIFAR-10 ViT training run, three Opus sessions independently diagnosed near-uniform attention and wrote a Gaussian locality prior into the early heads. They did it surgically, using 3–4 of each head's 64 Q/K dimensions.

This time **the structure survived:** at the end of training, the edited models were still clearly more local than the controls. **The accuracy didn't follow.** The edited runs ended −0.10 points from the controls on average (per seed −1.45, +0.12 and +1.03), with no head start along the way. The textbook alternative, mimetic attention initialization applied mid-run, crashed accuracy to chance and never recovered (−1.5 points).

The lesson: **an inductive bias is valuable because it shapes training from the start, not because it exists at the end.** Writing it in later gives the network the structure without the benefit.

---

## Act 6: No training afterwards

Every result so far said the same thing: edits lose to an optimizer that keeps running. So the last experiment removed the optimizer, and with it Sparks' own setting: a real, observable failure, fixed by editing, with nothing afterwards.

The failure was spurious features in a pretrained ResNet-50, from the [Spurious ImageNet](https://arxiv.org/abs/2212.04871) benchmark. Show it underwater scenery with no snorkel and it says "snorkel"; show it flowers and it says "bee". Three blind sessions each got 25 cue-only photos and 3 real photos per class, for 50 classes.

**All three found the same mechanism on their own:** context channels in the 2,048-d penultimate layer, which each class's readout row weights positively. They verified those channels are context detectors (the clean photos that fire "snorkel"'s cue channels hardest are sharks and coral). Then they shrank those weights and boosted the class's object channels to compensate.

On held-out data, the spurious hit rate fell from 50% to 16–22%, at a cost of 0.1–0.6 points of overall accuracy. **For the first time in the project, the edits clearly beat both random edits and naive training at the same operating point.** Plain fine-tuning reached the same hit rate in 11–94 steps, but mostly by suppressing the class everywhere, which cost the repaired classes 3–13 more points of accuracy than the edits did. That held even with 14× more labelled data.

Then two more baselines settled it:
1. **The right standard method won.** [DFR-style](https://arxiv.org/abs/2204.02937) balanced last-layer retraining, with about 40 labelled photos per class and an L2 pull toward the original weights, was the *only* arm to pass the pre-set bar. It needed 7 seconds of GPU. (Its pass was thin and the arm was added post-hoc; details in the caveats.)
2. **The agents' edit was a recipe.** A 30-line closed-form script ("cut cue channels, boost object channels") reproduced session 2 *exactly* and session 1 to within 0.1 points. The per-class judgment we thought we were paying for wasn't there; the agents contributed the *method*.

*[FIGURE: `runs/vision_spurious/tradeoff_imagenet.png`]*

---

## The scoreboard

| Experiment | Training after the edit? | Did the agent read the network correctly? | Did the edit help? | What beat or matched it |
|---|---|---|---|---|
| Grokking MLP, 3 pilots | yes | yes (Fourier circuit) | early gains, erased; damage kept | plain AdamW, and random edits |
| Qwen, RL head start | yes | yes (hedging direction) | ~7 steps' worth | plain GRPO |
| Pretraining plateau | yes | yes (missing induction circuit) | ~650 steps ahead, then erased | plain AdamW |
| Sparks' loop, RL only | — | — | — | 30 GRPO steps: 4% → 63% |
| ViT locality | yes | yes (uniform attention) | level with no edit | plain AdamW |
| ResNet-50 spurious features | no | yes (context channels) | yes, beats naive training | DFR retraining; a 30-line script reproduces the edit |

The "read correctly" column is all yeses. The "helped" column is almost all nos.

---

## Why the optimizer always wins

Once we understood the mechanism, it stopped being surprising. AdamW is close to an adversary for hand-written weights:

1. **The gradient rebuilds whatever lowers training loss.** If your edit removes something the loss wanted, like memorization, the fastest way back down is to rebuild it.
2. **Adam moves every weight at full speed.** Each step moves each parameter by roughly the learning rate, regardless of gradient size. Over a few thousand steps that's enough to rewrite any value you wrote.
3. **Weight decay erases anything the gradient doesn't defend.** A hand-written offset with no gradient behind it loses about two-thirds of its size in 1,000 steps in our toy.
4. **The ratchet: Adam undoes additions but not removals.** A dead ReLU gets no gradient and stays dead, and a head you overwrote has to be relearned. So every edit passes through a filter that **lets the harm stay and the benefit fade.**

This leaves an uncomfortable implication. **An edit lasts only if it moves the network somewhere the optimizer also wants to go, and the optimizer would get there anyway.** On a healthy run, then, the most an edit can buy is a head start, worth exactly the number of steps the optimizer needs to make the same change. That was ~10 steps in RL and ~650 on a genuine plateau, and never permanent.

The economics make it worse. The pretraining edit took 18 minutes and 173k tokens to save about 40 seconds of GPU time. A full grokking run takes 35 seconds; an agent round takes 4–11 minutes.

---

## So who's right?

**The "this is crazy" camp is right about reading.** Across 34 sessions and six architectures (MLP, transformer LM, ViT, ResNet, a 0.8B LLM), the agent found the real mechanism every single time: Fourier features, a hedging direction, a missing induction circuit, uniform attention, context channels. Nobody told it what to look for. It built a working induction circuit from weight statistics. Once it got feedback, it predicted the immediate effect of its own edits to within a few points. If you'd asked us a year ago whether an LLM could do that, we'd have been skeptical.

**The "this is nothing" camp is right about writing.** As a way of *improving* networks, direct weight editing loses to the optimizer whenever the optimizer runs, and to the right standard method when it doesn't. The Qwen loop fix is something RL does in 15 steps. The best training-free edit we produced reduced to a script.

**But "it's just interpretability" isn't the dismissal it sounds like.** That's where the value is. In every experiment, the useful output was the *diagnosis*: what's wrong, where it lives and what would fix it. The reusable artifact was a recipe, a training change or a script, not a set of hand-written numbers. The agent was most valuable as a researcher and least valuable as an optimizer.

Which is also our answer to the "this cuts autoresearch out of the loop" take: **we found the opposite.** Weight-space programming is best understood as *a very good autoresearch tool*. Let the model read the network, find the mechanism and design the fix, then let gradient descent or a script do the writing.

---

## Where editing might still win

We don't think the door is closed. Our results narrow it to cases where at least one of these holds:
- **No training follows, and none is possible:** a frozen production model, no labelled data, no compute budget.
- **Training would cause collateral damage an edit avoids.** The open question in Act 4: does RL's fix break other behaviour that a 0.016% edit leaves intact?
- **There is truly no gradient:** RL with a 0% success rate, or behaviour gated behind something the model never does. Here an edit only has to *start* the gradient flowing, not survive it. We never ran this one; it's the experiment we'd run next.
- **The failure is invisible to the training loss,** so the optimizer has no reason to push back.

What we'd stop doing: using weight edits to speed up healthy training runs. That looks like a dead end.

---

## Caveats

- **Small samples.** 3 seeds for most arms, 1–2 for the expensive ones. Seed-to-seed noise in the RL runs was 4–8 points.
- **The grokking toy is unusually hostile to edits:** weight decay 1.0 is very strong. It's also a famous problem, so Opus may be recognizing the textbook circuit rather than reading this specific network. We never ran the planned control with shuffled input labels.
- **One agent model.** Every session used Claude Opus 5.5; Sparks used a different frontier model. We don't know how much transfers.
- **Some comparisons are post-hoc,** notably DFR and the matched-stopping training arms in Act 6. They're labelled as such in our logs.
- **Pilot 3's trial runs reported held-out accuracy,** so agents picked among up to 3 candidates partly on test data. That favours the agent, and it still didn't pass.
- **None of this says Sparks is wrong.** Their results are about a different question: what can a model do to a network that stays frozen? Ours is what happens next.

All prompts, tool logs, agent scripts, submissions and results are in [REPO LINK].

---

<!--
EDITOR'S NOTES (delete before publishing)

TITLE OPTIONS
1. Weight-Space Programming Is Real. AdamW Doesn't Care.   (current; speaks directly to the debate)
2. Adam Forgets Your Edits but Remembers Your Mistakes     (punchiest, focused on the ratchet)
3. The Optimizer Always Wins                               (short, but more generic)
4. Weight Edits Are a Loan, and the Optimizer Collects     (ties to the "loan" line in the intro)
5. We Gave Claude a Scalpel. Gradient Descent Stitched It Back Up.

CHECK BEFORE PUBLISHING
- Quotes and attributions about the debate come from secondary coverage (Turing Post, Latent Space, PiyushAI, Lobsters). Verify against the originals:
  - Lucas Beyer's point about the repetition hint in the prompt
  - Sean Goedecke's rebuttal (I couldn't find the original post; the paraphrase comes from PiyushAI's summary)
  - Simon Willison's laptop reproduction (from the Lobsters thread)
  - Latent Space's "MONSTER discovery" phrasing
- Sparks used "GPT-6 Astra" according to coverage, not Opus. Confirm, then decide whether to name it.
- Act 4: our GRPO numbers (4k limit, 128 problems, thinking mode) are not directly comparable to Sparks' GSM8K 13.8% → 67.1%. The draft says so; keep the caveat.
- Session count 34 = 3 (pilot 1) + 9 (pilot 2) + 12 (pilot 3) + 1 (RL dry run) + 3 (pretraining) + 3 (ViT) + 3 (spurious).
- Spurious ImageNet and DFR arXiv links: double-check the IDs.

FIGURES TO MAKE
- Hero: DONE, blog/hero_grokking.png (+ .svg); regenerate with blog/make_hero.py. Currently placed in Act 1; could also go at the very top.
- Pretraining loss compare plot (exists: runs/pretrain_agent/code_d512_s0_step1000/compare.png)
- Spurious tradeoff plot (exists: runs/vision_spurious/tradeoff_imagenet.png)
- Optional: the "ratchet" diagram: edit adds → fades; edit removes → stays.

POSSIBLE CUTS IF TOO LONG (currently ~2,900 words)
- Merge pilots 2 and 3 into one paragraph.
- Drop Act 5 to three sentences.
-->

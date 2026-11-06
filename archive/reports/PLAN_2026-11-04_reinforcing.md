# Reinforcing experiments: make the blog's claims harder to dismiss

**Status (2026-11-04):** proposed, not run. These three were picked from 8 proposals, written by one Opus 5.5 subagent and one Fable 5.1 subagent. The collateral-damage design merges both agents' versions. **Reading audit** and **unlearning** are the Opus proposals.

**Scope:** pragmatic runs for a blog post, not a paper. Each section gives the design, then the trims we plan to take.

**Rough total:** about 2.5 days of wall-clock time. GRPO runs overnight, and the reading audit is mostly agent time, so it can run alongside the others.

**House rules (all three):**
- Matched random edit.
- Pass bar written before running.
- The cheapest strong standard fix as an arm.
- The edit never goes through an optimizer.

**Citations come from the agents' memory.** Verify each one before it goes into the blog.

---

## 1. Collateral damage: does RL's loop fix break things that a 0.016% edit leaves intact?

**Why.** Act 4 calls this "the most important open question." It is the last argument left for the Sparks edit over RL. Either result changes a sentence in the conclusion:
- RL does no collateral damage: Act 4 loses its "we didn't measure that" hedge.
- The edit wins on collateral: editing gets its first measured advantage.

It also fixes the one-seed caveat on the Act 4 table.

**Setup.**
- **Model and failure:** Qwen3.5-0.8B in thinking mode, 4k-token limit. This is the loop failure from `runs/eval/base`: 99% of answers hit the limit and 61% are verbatim loops.
- **Edit:** blind Opus sessions in the Sparks setting, fixing the loop with no training afterwards. The budget is about 0.016% of weights (about 120k entries).
  - Extend `rl/agent_tools.py` so that `T.generate` accepts `thinking=True, max_new=4096`. `grpo.py` already has `chat_prompt(..., thinking)`.
  - Cap each session at about 15 generation passes, because 4k-token passes are slow.
- **RL:** `rl/grpo.py --thinking --trunc-zero` with `--save-every 3`.
  - The existing `runs/grpo/thinkloop_s0` saved only step 30, so rerun seed 0 to step 15 with checkpoints, then add seeds 1 and 2.
- **Matched comparison (set before running):** for each edit, use the first GRPO checkpoint whose held-out loop-plus-limit rate is at or below the edit's. This compares the two methods at the same amount of fix. Also report GRPO at step 30, the full fix.

**Arms.**
- A. Base model
- B. GRPO, 3 seeds, at the matched step and at step 30
- C. Opus edit, 2–3 sessions
- D. Matched random edit, one per session
- E. Decoding only, no weight change:
  - repetition_penalty 1.1–1.3
  - no_repeat_ngram
  - budget forcing: insert `</think>` at about 1–3k tokens, Qwen's documented trick
  - If E alone gets loops below 10%, both methods are beside the point for this failure, and we say so.
- F. *(optional)* SFT on 16 of the model's own finished answers
- G. *(optional)* Sparks' own edit, if their published logs include it

**Side-effect battery (fixed before running, all held out).**
- **Target:** thinkprobe accuracy and loop rate (128).
- **Near tasks:**
  - IFEval (541, non-thinking, greedy)
  - GSM8K or GSM-Symbolic (500, non-thinking, greedy)
- **Far tasks:**
  - 1,000-question MMLU log-likelihood subset (lm-eval is installed `--no-deps`; check that the task runs, or use a 60-line log-likelihood script)
  - HellaSwag (1k)
  - FLORES zh/ja/th→en (100 sentences, chrF)
  - perplexity on held-out FineWeb-Edu and Python, re-streamed with `pretrain/data.py`
  - KL(base‖arm) on 200 neutral prompts
- **Out-of-domain loops:** thinking mode on 128 non-math prompts (a new suite). This is the most plausible place for the edit to win, if it acts on a repetition mechanism that isn't specific to math.

**Pass bar.**
- **"RL damages what the edit spares":** both must hold for at least one far metric, averaged over the 3 RL seeds.
  - The metric gets worse under RL by ≥1.5 points (or ≥2% perplexity), and by more than 2× the seed standard error.
  - The edit's change on that metric is less than half of RL's.
- **"Editing beats RL" (stronger):** all of the above holds, and the edit also keeps thinkprobe loops below 10%.
- **Random edit:** it must not get loops below 30%.
- **Null result, also set before running:** every GRPO side-effect change stays within ±2 points of base.

**Expected outcome.** At lr 1e-6 for 15–30 steps, GRPO probably moves KL by less than 0.05 nats and the far metrics by less than 1 point. IFEval may even improve, since 21% of base IFEval answers loop. Most likely, neither method causes more damage, and RL is far ahead on the target. That removes the blog's main hedge. The best chance for an edit win is out-of-domain loops.

**Trimmed run (about 1 day).**
- 2 edit sessions, not 3.
- Seed 0 GRPO rerun with checkpoints, plus seeds 1 and 2, all overnight (about 5.5–7.5 h).
- Battery about 45 min per checkpoint. Only the matched-step and step-30 checkpoints get the full battery.
- Drop F and G unless there's time.

**Reuses:** `rl/grpo.py`, `rl/eval.py`, `rl/agent_tools.py`, `rl/evals_table.py`, `runs/eval/base`, `runs/grpo/thinkloop_s0`.

---

## 2. Reading audit: is the agent reading the network, or recalling textbook circuits?

**Why.** "Reading is real" rests on five textbook mechanisms: Fourier features, induction, locality, context channels and a hedging direction. A model might name all of them from the task description alone. "Nobody told it what to look for" doesn't rule that out. This experiment:
- separates prior knowledge from reading
- covers the "famous toy" caveat
- covers the "one agent model" caveat

It protects the half of the headline that critics will attack ("it just recalled Nanda").

**Setup.**

**(a) Control with no weights.**
- For each of the 5 settings, run 3 sessions with the same prompt plus the architecture and training-curve facts, but **no weights and no forward passes**.
- Each session writes a diagnosis, an edit plan and a predicted effect.
- Score these sessions and the existing 15 with-weights sessions on a rubric written before scoring. The rubric separates:
  - **generic items:** the mechanism class
  - **network-specific items:** which neurons, heads, channels or frequencies, checked by ablation
- Graders don't know which condition a session came from.

**(b) Disguised grokking.** This is the shuffled-label control we planned and never ran.
- **Task:** (σ(a)+σ(b)) mod 97, where σ is a secret relabeling of inputs and outputs. The circuit is the same, but it can't be recognized from token order. S5 composition is an optional, harder variant.
- **Protocol:** pilot 1's, with 3 seeds and a 2% budget.
- **Objective reading test:** ablating the components the agent names must cost ≥3× as much as matched random ablations.
- The training forks also check whether the ratchet repeats in a network the agent can't have memorized.

**(c) Second agent model.**
- Repeat (b) with a weaker Claude model (Sonnet or Haiku).
- If the harness supports it, also with a non-Anthropic frontier model.

**Arms:** with weights vs no weights; Opus vs the second model; matched random ablations and edits; no edit.

**Pass bar.**
- **The claim holds if both are true:**
  - The no-weights sessions score at most ⅓ of the with-weights sessions on network-specific items.
  - At least 2 of 3 disguised sessions pass the ablation test.
- **Otherwise:** downgrade the claim to "names the textbook mechanism, then confirms it."

**Expected outcome.** The no-weights sessions probably name the mechanism class in 3–4 of the 5 settings, with no network-specific content. Opus most likely still finds the circuit under disguise, and the forks show the same ratchet. The claim comes out sharper: "specific components, verified, in a network it couldn't recognize." We concede honestly that the mechanism class comes from prior knowledge.

**Trimmed run (about half a day; under 1 h of GPU, the rest is agent time and grading).**
- Part (a): 1 session per setting, not 3, so 5 sessions.
- Part (b): 3 seeds.
- Part (c): one second model on one seed.
- Grading: the main session scores against the written rubric, blind to condition, with spot checks by hand.

**Reuses:** `experiment.py`, `grok.py`, `edit_tools.py`, `kill_controls.py`, and the existing session logs under `runs/seed*/`, `runs/pretrain_agent/`, `runs/vision_vit/` and `runs/vision_spurious/`.

---

## 3. Unlearning under a relearning attack: use the ratchet on purpose

**Why.** "Adam undoes additions but not removals" hurt the edits in every experiment so far. Unlearning flips this around: removal is the goal, and the optimizer is the adversary.
- Gradient unlearning methods (gradient ascent, RMU) are known to mostly *suppress* knowledge rather than delete it, and a few fine-tuning steps bring it back. Verify these citations: Deeb & Roger 2024; Lynch et al. 2024.
- By the draft's own theory, a hand-written deletion of the storage itself should persist.

This is a sharp, testable prediction that follows from our own results. Unlearning also matters in practice: GDPR deletion requests and removing hazardous knowledge.

**Setup.**
- **Data:** synthetic biographies of 1,000 fictitious people, 4 attributes each, written from paraphrased templates and mixed with FineWeb (`pretrain/data_fw`).
- **Model:** the 6-layer GPT, d384, trained from scratch, 3 seeds (`pretrain/model.py`, `train.py`).
- **Forget set:** 100 people, split into halves A and B.
- **Agent:** reads activations on forget vs retain prompts and writes the removal under an entry budget, with no gradients.
- **Attack:** fine-tune on half A only. The attacker picks the best of 3 learning rates and runs at most 200 steps. Then measure accuracy on half B. The facts are independent, so B can come back only if it is still stored in the weights.

**Arms.**
- No unlearning
- Gradient-difference unlearning
- RMU
- Agent edit
- Matched random edit
- A script fixed before running: ablate neurons by their forget/retain activation ratio. This is the control for "a 30-line script does it too."
- Exact unlearning: retrain from scratch without the forget set. It's cheap at this size and sets the floor.

**Pass bar (after the best attack, in at least 2 of 3 seeds).**
- Accuracy on half B is ≥20 points below RMU's, and within 15 points of the retrain floor.
- Retain accuracy is no more than 2 points below RMU's.
- Validation loss rises by ≤0.05 nats.
- At matched retain damage, the edit is ≥10 points better than the script.
- The random edit fails.

**Expected outcome.** Facts are spread across MLP neurons, so a real removal needs wide ablation and costs retain accuracy. The agent probably beats RMU on robustness, but sits on the same trade-off curve as the script. That would refine the headline: removals persist, as predicted, but a script does it too. If the edit clears the bar, it is the project's first lasting win.

**Trimmed run (about 1 day, 8–10 h of GPU).**
- 3 base trainings.
- 1 agent session per seed.
- RMU and gradient difference at default hyperparameters, with no sweep.
- Attack with 3 learning rates.
- One retrain-from-scratch run per seed.

**Reuses:** `pretrain/model.py`, `train.py`, `data.py` (`data_fw`), `agent_tools.py`, `compare_arms.py`. New code: the biography generator, fact-recall evaluation, RMU and gradient difference, and the attack loop.

---

## Suggested order

1. **Collateral damage:** start the GRPO seeds overnight first.
2. **Reading audit:** run alongside the GRPO runs, since it needs under 1 h of GPU.
3. **Unlearning.**

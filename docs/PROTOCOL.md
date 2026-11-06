# The protocol every task follows

This is a blog-grade protocol, not a paper-grade one. These are the minimum standards we held every experiment to,
and the ones we'd ask anyone adding a task to keep.

## 1. Stop-or-go check before building anything
- **Does the model fail often enough to matter?** Run `base` on practice data first.
- **Is there a signal to use?** Run `signal`: a linear probe at the decision point, scored by 5-fold cross-validated
  AUC.
- **Thresholds.** Every gate that worked read a signal with AUC ≥ 0.99. Where the best signal was ~0.70 ("Are you
  sure?" caving), there was nothing to build.

## 2. Practice vs held-out
- **The fix sees only practice data.** That covers the probe, the threshold, the target opener, and the LoRA training
  set.
- **Held-out uses new items *and* new phrasings,** e.g. "Tell me about X." on practice vs "Give me a short bio of X."
  on held-out. A fix that only memorised the practice wording fails here.
- **Nothing is tuned on held-out.** The gate's hyper-parameters (gain, g, k, margin) and the LoRA recipe are fixed
  across tasks.

## 3. Five arms, same data
| Arm | What it is | Why it's there |
|---|---|---|
| base | unmodified model | the failure rate |
| prompt | a sensible system-prompt fix | the thing you'd try first; often good, sometimes blunt |
| gate | one repurposed MLP neuron (≈3 × d_model entries) | the weight patch |
| random | the same neuron rewritten with random vectors of the same norms | rules out "any perturbation helps" |
| lora | LoRA SFT on the same practice prompts, with self-generated targets | the training baseline: what data buys you |

The LoRA targets carry exactly the information the gate gets:
- **Positive (failure) items:** the model's own continuation after the target opener.
- **Negative items:** its own unchanged reply.

## 4. Guards: what else did the fix change?
- **Task-specific "kept" metric.** For example: facts about real entities, genuine tool calls, content of the
  answer.
- **MMLU-200:** a letter-choice accuracy check.
- **Chat KL:** mean per-token KL(base ‖ arm) on 60 ordinary chat replies (no_robots), teacher-forced.
- **Cost:** entries changed, trainable parameters, and seconds to build the fix.

## 5. Data scaling
`sweep` refits gate and LoRA on stratified practice subsets (n = 8 … all) with 3 seeds. That's the honest answer to
"does training data beat it?": at which n, by how much, and at what side-effect cost.

## 6. Known limits
- **Greedy decoding, one sample per prompt.** Held-out sets are a few hundred items, so differences of a few points
  are noise.
- **String/regex scoring.** It's checked by reading outputs, but it's not a human or LLM judge.
- **One model per task,** plus one transfer run.
- **Hand-written attacks and prompts.** No adaptive adversary.

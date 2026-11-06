<!-- Extracted from RESULTS.md (sections matching ['pretraining']); full file in archive/reports/RESULTS.md -->

# Results: agent edits during small-LM pretraining (2026-11-04)

Pretraining from scratch on one RTX 4070, looking for a loss plateau where an edit might beat the optimizer. All runs use seed 0. The summary is in `FINDINGS.md`; the scripts are described in `pretrain/README.md`.

## Where is there a plateau? (baseline grid)

- **Training settings for every run:** 15,000 steps, 64 × 512 tokens per step, AdamW lr 1e-3 (100-step warmup, cosine decay to 1e-4), weight decay 0.1. The TinyStories runs used a 256-token context and 64 × 256 tokens per step.
- **Tokenizers:** byte-level BPE with 4,096 tokens (TinyStories) or 8,192 (web text and code).
- **Probe columns:** "onset" is the first probe where the smoothed copy benefit reaches 0.25 nats (copy benefit = loss on the first copy of a real passage minus loss on an exact repeat). "t90" is the step where it reaches 90% of its final value.

| Data | Model | Params | Onset | t90 | Final copy benefit | Val @1k | Val @3k | Val @15k |
|---|---|---|---|---|---|---|---|---|
| TinyStories | 2L attention-only d256 | 2.7M | never | – | 0.12 | 2.484 | 2.159 | 1.866 |
| TinyStories | 4L GPT d256 | 5.3M | never | – | 0.09 | 2.319 | 1.877 | 1.499 |
| Web (FineWeb-Edu) | 2L attention-only d256 | 4.9M | **5,350** | 11,200 | 0.96 | 4.785 | 4.458 | 3.926 |
| Web | 2L attention-only d512 | 10.8M | 1,600 | 4,200 | 1.63 | 4.932 | 4.073 | 3.671 |
| Web | 4L GPT d256 | 7.5M | 1,675 | 4,525 | 1.39 | 4.702 | 3.967 | 3.492 |
| Web | 6L GPT d384 | 17.1M | 1,000 | 4,525 | 1.73 | 4.432 | 3.732 | 3.249 |
| Code (codeparrot-clean) | 2L attention-only d256 | 4.9M | 1,975 | 3,675 | 1.13 | 3.150 | 2.485 | 2.088 |
| Code | **2L attention-only d512** | 10.8M | **1,875** | **2,450** | 1.19 | 3.114 | 2.351 | 1.911 |
| Code | 4L GPT d256 | 7.5M | 950 | 2,125 | 1.24 | 2.916 | 2.220 | 1.773 |
| Code | 6L GPT d384 | 17.1M | 700 | 1,100 | 1.16 | 2.559 | 2.037 | 1.585 |

- **TinyStories:** induction never forms (best induction score 0.06) and the loss curve is smooth. The data rarely rewards exact copying.
- **Bigger models form induction sooner,** which leaves less plateau to skip.
- **The longest plateau** is web text with the 2L d256 model: about 5,000 flat steps, then slow formation, with no knee in the loss.
- **The only plateau visible in the training loss itself** is code with the 2L d512 model: loss is flat from about step 1,000 to 1,875, then drops 0.34 by step 2,450. We used this as the test bed.
- **Reference points on code:**
  - Uniform guessing: 9.01 nats.
  - A unigram model: 6.84.
  - A smoothed bigram model fitted on 60M training tokens: 4.67.
  - The 2L d512 model passes the bigram level at about step 200.

## Single edit at step 1,000 (code, 2L attention-only d512)

Session: `runs/pretrain_agent/code_d512_s0_step1000/`.
- **Prompt and tools:** `PROMPT.md` and `pretrain/agent_tools.py`. The agent may not use gradients or fitting; `tok.weight` and `head.weight` are out of scope; the edit is capped at 400k entries.
- **Arms:** all resume from `step_01000.pt` with the same optimizer state and the same batch sequence. The resumed control matches the original run to within 0.0005.

**The edit** (17.7 min, 173k tokens, 184 forward calls), 295,328 entries:

| Tensor | Entries | Change norm | Purpose |
|---|---|---|---|
| `pos.weight` | 32,736 | 8.96 | sinusoids at 0.5–1.5 rad/token in the 64 lowest-variance dimensions |
| `blocks.0.ln1.weight` | 448 | 1.58 | keeps the other heads' input scale unchanged |
| `blocks.0.attn.qkv.weight` | 98,304 | 23.9 | L0H6 becomes a previous-token head (Q is the sinusoids rotated by one step); V carries a 64-dim token code |
| `blocks.0.attn.proj.weight` | 32,768 | 4.64 | L0H6 writes that code into quiet residual dimensions |
| `blocks.1.attn.qkv.weight` | 98,304 | 60.2 | L1H2 becomes an induction head (K reads the previous-token code, Q and V read the current token's code) |
| `blocks.1.attn.proj.weight` | 32,768 | 4.38 | L1H2's output maps code to unembedding (Hebbian) |

**Immediate effect on training windows:**
- Loss: 3.399 → 3.544.
- Copy benefit: 0.081 → 0.122.
- L0H6 previous-token score: 0.025 → 0.815.
- L1H2 induction score: 0 → 0.14.
- The random edit of the same size raised the loss to 6.07.

**Continued training (held-out val loss):**

| Arm | 1,250 | 1,500 | 2,000 | 3,000 | 5,000 | 10,000 | 15,000 | Mean over 1k–15k |
|---|---|---|---|---|---|---|---|---|
| No edit | 3.000 | 2.939 | 2.669 | 2.351 | 2.160 | 1.987 | 1.911 | 2.1359 |
| Agent | 2.749 | 2.619 | 2.464 | 2.320 | 2.153 | 1.978 | 1.905 | 2.1039 |
| Random | 3.064 | 2.963 | 2.739 | 2.360 | 2.170 | 1.999 | 1.919 | 2.1531 |

**Mean gap to no edit, by window:**

| Steps | Agent − none | Random − none |
|---|---|---|
| 1,000–2,500 | −0.207 | +0.099 |
| 2,500–5,000 | −0.022 | +0.011 |
| 5,000–10,000 | −0.007 | +0.008 |
| 10,000–15,000 | −0.010 | +0.005 |
| last 1,000 | −0.007 | +0.007 |

- **Steps saved:** how many steps later the control reaches the agent's loss. 650 at step 1,250, 400 at 2,000, 200 at 3,000 and 100 at 5,000. Later values aren't meaningful because the curve is flat.
- **Copy benefit stays higher in the agent arm** to the end: 1.28 vs 1.19. The random arm ends at 1.22.

## Follow-up sessions at step 3,000 (from the agent arm)

| Session | Objective and tools | Time, tokens | Calls | Submission |
|---|---|---|---|---|
| `code_d512_s0_multi/s2_step3000` | same rules as step 1,000; loss of the continued run | 5.5 min, 102k | 42 | **empty edit** |
| `code_d512_s0_multi/s2b_step3000_deep` | scored on the loss at step 15,000; immediate cost allowed; `T.lookahead` (10 paired training runs of up to 1,500 steps, different batch order, training windows only) | 21 min, 186k | 78 + 3 look-aheads | **empty edit** |

**Look-aheads in the deep session** (training-window loss at step 4,500; paired, same batches):

| Candidate | Loss at 4,500 | Gap to no edit |
|---|---|---|
| No edit | 2.149 | – |
| A: L0H5 rebuilt as a second token-code head (complementary 64-dim code, output weights zero) | 2.160 | +0.064 at the start, then flat at +0.011 to +0.016 over the last 800 steps |
| C: function-preserving QK rescale (Q×3, K÷3) on 6 L1 heads | 2.152 | −0.001 to +0.003 (noise) |

**What the two sessions found:**
- **The planted circuit is intact:** L0H6 previous-token score 0.997, L1H2 induction score 0.46.
- **Copying is close to its limit.** Confidence matches accuracy, the model prefers longer matches, and a perfect copy mechanism through the same attention made loss worse. A flat bonus for every identifier already in the window would gain at most 0.016.
- **The 2 L1 heads read the current token mostly through one 64-dim code,** L0H4's output (77% of L1H2's value input). Adding a second code channel cost loss for as long as it was measured.
- **No head is cheap to remove:** zeroing any one costs at least about 0.055.
- **Indentation:** an ideal tracker is worth at most about 0.03, and L1H1 is already learning one.

## Head usage of the trained control (step 15,000)

Script: `runs/pretrain_agent/head_usage.py`, 48 validation windows, base loss 2.105.
- **Removal cost of each head** (zeroing its output):
  - L0: 0.09–1.76. The current-token head, L0H4 (mean distance 2.7, 96% of its attention within 4 tokens), costs 1.76.
  - L1: 0.10–0.23. The long-range heads, L1H0 and L1H5 (mean distance 116–147), cost about 0.21–0.23.
- **Redundancy:** joint removal of a pair costs a median **1.08×** the sum of the single costs, so heads are slightly complementary. The most overlapping pairs are L0 H0+H3 (0.78×) and L0 H4+H5 (0.85×).

## Memorization vs generalization (code models, step 15,000)

Script: `pretrain/probe_understanding.py`.

| Probe | 6L GPT d384 (17.1M) | 2L attention-only d512 (10.8M) |
|---|---|---|
| Loss, training vs unseen files | 1.580 vs 1.547 | 1.906 vs 1.868 |
| Greedy continuation of 32 tokens: share of tokens right (exact continuations), training vs unseen files | 15% (2%) vs 20% (5%) | 13% (2%) vs 16% (4%) |
| Made-up identifiers: loss per name, first vs later uses | 32.8 vs **3.0** nats | 33.2 vs 5.8 nats |
| Random nested `([{` brackets: correct closer ranked first (chance 33%) | **99%** | 94% (83% at depth 4) |
| Indentation after `if`/`def`/`for`/`while`/`class` lines | 100% | 99% |

- **No memorization signal:** these models generalize Python syntax, idioms and name binding.
- **The samples don't understand what code does.** `fibonacci(n)` returns `n`, a file opened for reading gets written to, there are undefined variables, and lines repeat.
- **A pitfall:** a prompt ending in a bare `\n` token pushes these tokenizers into a rare state, because in training a newline and the next line's indentation form one token. End prompts at the colon instead.

## Cost and disk

- **Agent sessions:** 3, totalling about 45 minutes and 460k tokens.
- **GPU:** about 6 hours in all, including the grid.
- **Disk:**
  - Data: `pretrain/data*`, about 3 GB.
  - Runs: `runs/pretrain*`, about 34 GB, mostly checkpoints with optimizer state.
  - Keep `runs/pretrain_code/attn2L_d512_s0/ckpt/step_01000.pt` and `runs/pretrain_agent/code_d512_s0_step1000/arm_agent/ckpt/step_03000.pt`. They are the starting points of all the sessions.

## Not done

- More seeds. Without them, the remaining 0.006 gap at step 15,000 can't be told apart from noise.
- A hand-written-circuit arm.
- The planned step-6,000 and step-10,000 interventions, dropped after two empty edits at step 3,000.
- The same pipeline on the 6L GPT with MLPs, which has more structure that could form late or badly.


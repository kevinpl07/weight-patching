# Archive: the experiments before the release

This folder keeps the code, plans, results and agent-session records of the eleven experiments that came before this
repository. They fall into two tracks:

- **Track A, mid-training edits.** An agent edits the weights during training, and training then continues. In every
  case the optimizer erased the gain.
- **Track B, frozen-model patches.** An agent (or a hand-built recipe) edits a trained model, and no training follows.

| Experiment | Question | Model | Track | Headline result | Folder |
|---|---|---|---|---|---|
| Grokking pilots | Does an LLM's weight edit at the memorization plateau make the network grok sooner? | 2-layer ReLU MLP on (a+b) mod 97 | A | Opus edits reach 50% held-out 1.8–3.4× sooner, but 0 of 9 Opus runs ever reach 95% (unedited: 3 of 3). The switched-off neurons never come back. | [grokking/](grokking/) |
| RL head start | Does an edit that fixes rambling give GRPO a lasting head start? | Qwen3.5-0.8B | A | +7.2 points at step 0, gone by step 15; −0.3 at step 100 (mean of 2 seeds). | [rl_head_start/](rl_head_start/) |
| Sparks loop counterfactual | Would plain RL have fixed Sparks' thinking-mode loop? | Qwen3.5-0.8B (thinking) | A (no edit arm) | Yes: 30 GRPO steps take held-out accuracy 3.9% → 63.3%, loops 61% → 3.9%. | [sparks_loop_rl/](sparks_loop_rl/) |
| Pretraining induction plateau | Can an edit get a small LM past the induction-head plateau sooner? | 2-layer attention-only GPT, 10.8M params, Python code | A | Up to 0.33 nats / ~650 steps ahead, closed within ~1,000 steps; final gap 0.006 nats (noise). | [pretrain_induction/](pretrain_induction/) |
| ViT locality | Does writing locality into a ViT's attention at 5% of training raise final accuracy? | ViT 2.69M params, CIFAR-10 | A | Level with no edit: −0.10 points (−1.45, +0.12, +1.03). The locality survives; the accuracy gain doesn't. | [vit_locality/](vit_locality/) |
| Spurious features | Can an edit stop a classifier relying on spurious cues, without training? | ResNet-50 (ImageNet) | B | Spurious hit rate 50.2% → 15.7–21.8% with 3k–30k fc entries; beats naive training, but only DFR-style retraining passes the bar. A short script reproduces the edits. | [spurious_resnet/](spurious_resnet/) |
| Over-refusal | Can an edit stop refusing safe prompts while keeping harmful refusals? | Phi-3-mini-4k-instruct | B | 0 of 6 sessions pass (harmful refusal drops 3–10 points); at matched false refusal the edits keep 16–33 more harmful-refusal points than abliteration. | [overrefusal_phi3/](overrefusal_phi3/) |
| Language mixing | Can an edit stop Chinese leaking into German/Russian/Thai answers? | DeepSeek-R1-Distill-Qwen-1.5B | B | Leaking 31.4% → 0.8–1.4%, 3 of 3 sessions pass, no measurable cost; recipe script 1.7%. | [langmix/](langmix/) |
| Whisper hallucination | Can an edit stop Whisper transcribing text from non-speech audio? | whisper-small (+ large-v3-turbo transfer) | B | 84.0% → 0.3% with one gated neuron (1,537 entries), speech unchanged; 3 of 3 pass. Recipe transfers to turbo (78.3% → 1.3%), behind VAD. | [whisper_hallucination/](whisper_hallucination/) |
| Sign-off gate | Can one neuron stop "Would you like me to…" endings? | gemma-3-1b-it | B (hand-built) | Sign-offs 58.3% → 9.0% on the clean held-out set, 97.6% of content kept; random control does nothing. | [signoff_gate/](signoff_gate/) |
| One-neuron gates | Where does a one-neuron gate fix a real LLM failure? | gemma-3-4b-it, Qwen3-4B-Instruct | B (hand-built) | Invented-entity bios 98% → 2%; text injections −43% on unseen goals; agent tool-call hijacks 7.5% → 1.2%; caving stopped at the check (no signal). | [one_neuron_gates/](one_neuron_gates/) |

## Layout

Each experiment folder has:

- `code/`: the experiment's source as it was (`.py`, `.sh`, `.ps1`, prompt templates, READMEs), under its original
  directory name.
- `reports/`: its plan and results. For Track A, `RESULTS_*.md` and `FINDINGS_*.md` are the relevant sections cut from
  the shared `RESULTS.md` and `FINDINGS.md`.
- `sessions/`: what the agents actually did, one folder per session: the `PROMPT.md` the agent got, its notes and final
  reply, the submitted-edit metadata (`submission.json`, `session.json`, `arm_*.json`), the tool-call log
  (`tool_log.jsonl`), the scripts it wrote, and its budget ledger. Grokking sessions are named `pilot{1,2,3}_seedN[_roundK]`.
- `results/`: small summary files (per-arm `summary.json`, summary tables, edit metadata). No weights, activations,
  audio or datasets.

Other folders:

- `reports/`: every top-level plan, results and report markdown from the original project, unchanged.
- `shared/`: `gpu_lock.py` (used by most Track B scripts) and the original project README.
- `extract_archive_data.py`: builds `site/data/archive/*.json` from the original run files; every block records its
  source path.
- `make_archive_figures.py`: draws `site/img/archive_*.png|svg` from those JSON files only.

## Caveats

- **The archived code is kept for reference and is not guaranteed to run from here.** It uses the original project
  layout (`runs/`, `data_audio/`, `data_vision/`, `gpu_lock.py` at the root, model caches), and the runs depend on
  model checkpoints and datasets that are not included.
- Absolute paths in copied files were replaced with placeholders: `<project>` for the original project root, `<home>`
  for the user's home directory, `<remote_home>` for the cloud machines' home directory.
- The sign-off and one-neuron gates were hand-built with full context; they are not blind agent results. Grokking, RL,
  pretraining, ViT, spurious-feature, over-refusal, language-mixing and Whisper edits came from blind Opus 5.5 sessions.

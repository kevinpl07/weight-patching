# Reproduction check (fresh checkout, different GPU)

`git archive HEAD` of this repo was unpacked on a different machine (NVIDIA L4 instead of A10G). Then, exactly as in
the README, we ran `pip install -r requirements.txt` and `python -m patchkit.run all entity`. The run rebuilt
everything: data, base replies, probe, gate and LoRA.

| arm | fabrication (original → fresh) | real-entity facts kept | chat KL |
|---|---|---|---|
| base | 98.2 → 98.2 | 56.6 → 54.1 | 0 |
| prompt | 6.8 → 7.3 | 20.2 → 19.4 | – |
| gate | 15.5 → 15.1 | 55.6 → 52.0 | 0.0002 → 0.0003 |
| random | 97.3 → 97.3 | 50.5 → 53.1 | 0.35 → 0.36 |
| LoRA | 29.7 → 31.1 | 55.6 → 53.1 | 0.16 → 0.14 |

**Where the small differences come from.** bf16 greedy decoding is not bit-identical across GPU types. That changes
which real entities count as "known": 88 vs 89 practice items. Every difference is within the noise of a few hundred
held-out items.

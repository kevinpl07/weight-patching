"""The `signoff` task on a smaller held-out set, for the data-scaling sweep only (GPU time).

Same model, practice items, base replies, probe sites, LoRA targets and scoring as tasks/signoff.py. Held-out = the
first 100 of signoff's 300 held-out prompts (fixed before any sweep result was seen) + all 40 legit prompts,
generated in one batch. Build it after `data signoff` and `base signoff`:

    python -m patchkit.run data  signoff_sweep     # copies items/base replies from signoff, no GPU
    python -m patchkit.run sweep signoff_sweep --ns 16,32,64,128,all --seeds 0,1,2 --methods gate,lora
"""
import shutil

from patchkit import llm as L
from patchkit.task import register

from tasks.signoff import ROOT, Signoff

N_HELDOUT = 100


@register
class SignoffSweep(Signoff):
    name = "signoff_sweep"
    gen_batch = 140   # one batch for the whole sweep held-out set (refs were generated at 64: see NOTES.md)

    def build_items(self, ctx):
        src = L.read_jsonl(ROOT / "results" / "signoff" / "items.jsonl")
        held = [x for x in src if x["split"] == "heldout" and not x.get("legit")][:N_HELDOUT]
        keep = {x["id"] for x in held} | {x["id"] for x in src if x.get("legit") or x["split"] == "practice"}
        shutil.copy(ROOT / "runs" / "signoff" / "base_practice.jsonl", ctx.runs / "base_practice.jsonl")
        return [x for x in src if x["id"] in keep]

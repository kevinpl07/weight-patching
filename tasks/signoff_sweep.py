"""The `signoff` task on a smaller held-out set, for the data-scaling sweep only (GPU time).

Same model, practice items, base replies, probe sites, LoRA targets and scoring as tasks/signoff.py. Held-out = the
first 64 of signoff's 300 held-out prompts, generated in one batch of 64 -- the same batch composition as
the first batch of the main eval, so the base reply equals the stored reference. No legit prompts (GPU time: a
first attempt with 100 + 40 legit in one batch of 140 ran ~8 min per arm, likely spilling out of the 12 GB VRAM). Build it after `data signoff` and `base signoff`:

    python -m patchkit.run data  signoff_sweep     # copies items/base replies from signoff, no GPU
    python -m patchkit.run sweep signoff_sweep --ns 16,32,64,128,all --seeds 0,1,2 --methods gate,lora
"""
import shutil

from patchkit import llm as L
from patchkit.task import register

from tasks.signoff import ROOT, Signoff

N_HELDOUT = 64


@register
class SignoffSweep(Signoff):
    name = "signoff_sweep"
    gen_batch = 64

    def build_items(self, ctx):
        src = L.read_jsonl(ROOT / "results" / "signoff" / "items.jsonl")
        held = [x for x in src if x["split"] == "heldout" and not x.get("legit")][:N_HELDOUT]
        keep = {x["id"] for x in held} | {x["id"] for x in src if x["split"] == "practice"}
        shutil.copy(ROOT / "runs" / "signoff" / "base_practice.jsonl", ctx.runs / "base_practice.jsonl")
        return [x for x in src if x["id"] in keep]

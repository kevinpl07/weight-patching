"""Does GRPO learn what the agent found? Teacher-force a fixed set of 64 base answers through each checkpoint and
measure, against the base model:

(a) Hedging direction. The agent's direction m_L (runs/agent_dry/work/seqlevel.pt, ("hmean", L)[0]: problem-matched
    mean residual difference, truncated minus finished base answers, residual after layer L) at L = 15, 16, 19, 21.
    For each answer: mean residual over the answer positions (same convention as the agent: from the last
    <|im_start|> + 2), projected on the unit vector m_L / |m_L|. The agent's edit adds -(3/4) m_L at each of those
    layers, so it lowers these projections; the question is whether training does the same on its own.
(b) mech_check-style statistics on the answer tokens: next-token entropy (all tokens / at hedge decisions, i.e.
    where the actual next token is a hedge word / elsewhere), change in log-prob of the actual tokens vs base,
    total probability on hedge-word tokens per position, and NLL of the answers (overall drift). The base model at
    T=0.8 is reported alongside as the "just sample cooler" reference.

    python rl/ckpt_analysis.py runs/grpo/armA_s0 runs/grpo/armB_s0 runs/agent_dry/model_edit
    python rl/ckpt_analysis.py runs/grpo/armA_s0/ckpt/step_00150 --n 8 --device cpu   # quick check

Arguments are run folders (every ckpt/step_* inside, in order) or model folders. The base model is always
measured first (cached in runs/analysis/base.pt). The answer set is fixed on first use: 32 truncated + 32 finished
answers from runs/eval/base/symbolic_sample.jsonl.gz (seed 0), saved to runs/analysis/answers64.jsonl.gz.
Output: one JSON per model in runs/analysis/<run>__<step>.json, and a table on stdout.
Needs ~4 GB of GPU memory; don't run it next to a training run (WDDM spills to RAM instead of failing).
"""
import argparse
import gzip
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval import chat, symbolic  # noqa: E402
from grpo import MODEL, SYSTEM  # noqa: E402
from mech_check import HEDGES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "analysis"
LAYERS = (15, 16, 19, 21)
DIRS = ROOT / "runs" / "agent_dry" / "work" / "seqlevel.pt"
SOURCE = ROOT / "runs" / "eval" / "base" / "symbolic_sample.jsonl.gz"
CHUNK = 128  # positions per lm_head block; bounds the (chunk, 248k) fp32 logits


def answer_set():
    path = OUT / "answers64.jsonl.gz"
    if not path.exists():
        recs = [json.loads(l) for l in gzip.open(SOURCE, "rt", encoding="utf-8")]
        rng = random.Random(0)
        tr = [r for r in recs if r["truncated"]]; fi = [r for r in recs if not r["truncated"]]
        rng.shuffle(tr); rng.shuffle(fi)
        sel = sorted(tr[:32] + fi[:32], key=lambda r: r["id"])
        OUT.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as f:
            for r in sel:
                f.write(json.dumps(r) + "\n")
    return [json.loads(l) for l in gzip.open(path, "rt", encoding="utf-8")]


def hedge_vocab(tok):
    words = tok.batch_decode([[i] for i in range(len(tok))])
    return torch.tensor([w.strip().lower().lstrip("*(\"'") in HEDGES for w in words])


@torch.no_grad()
def measure(model, tok, items, hv, device, temps=(1.0,)):
    """Per answer: projections on the hedging directions, and per answer token: entropy, log-prob of the actual
    token and probability mass on hedge tokens, at each temperature in temps."""
    start_tok = tok.convert_tokens_to_ids("<|im_start|>")
    dirs = torch.load(DIRS, weights_only=False)
    unit = {L: (lambda m: m / m.norm())(dirs[("hmean", L)][0].float().to(device)) for L in LAYERS}
    V = model.lm_head.out_features  # padded past len(tok); the padding rows are never hedge tokens
    hv = torch.cat([hv, torch.zeros(V - len(hv), dtype=torch.bool)]).to(device)
    proj ={L: [] for L in LAYERS}
    tokstats = {T: dict(ent=[], lp=[], hmass=[]) for T in temps}
    for it in items:
        ids = it["ids"].to(device)[None]
        o = model.model(input_ids=ids, output_hidden_states=True)
        s = max(i for i, t in enumerate(it["ids"].tolist()) if t == start_tok) + 2  # the agent's convention
        for L in LAYERS:
            proj[L].append(float(o.hidden_states[L + 1][0, s:].float().mean(0) @ unit[L]))
        P = it["P"]
        h = o.last_hidden_state[0, P - 1:-1]  # predicts the answer tokens
        tgt = ids[0, P:]
        for T in temps:
            ent, lp, hm = [], [], []
            for c in range(0, len(tgt), CHUNK):
                logp = torch.log_softmax(model.lm_head(h[c:c + CHUNK]).float() / T, -1)
                p = logp.exp()
                ent.append(-(p * logp).sum(-1))
                lp.append(logp.gather(-1, tgt[c:c + CHUNK, None])[:, 0])
                hm.append(p[:, hv].sum(-1))
            for k, v in zip(("ent", "lp", "hmass"), (ent, lp, hm)):
                tokstats[T][k].append(torch.cat(v).cpu())
        del o
    return dict(proj={L: torch.tensor(v) for L, v in proj.items()},
                tok={T: {k: torch.cat(v) for k, v in d.items()} for T, d in tokstats.items()})


def summarize(m, base, is_hedge, trunc):
    """Scalars for one model vs base (paired over the same answers / tokens)."""
    t, b = m["tok"][1.0], base["tok"][1.0]
    other = ~is_hedge
    se = lambda x: float(x.std() / len(x) ** 0.5)  # noqa: E731
    out = {}
    for L in LAYERS:
        d = m["proj"][L] - base["proj"][L]
        out[f"proj{L}"] = float(m["proj"][L].mean())
        out[f"proj{L}_trunc"] = float(m["proj"][L][trunc].mean())
        out[f"proj{L}_fin"] = float(m["proj"][L][~trunc].mean())
        out[f"dproj{L}"] = float(d.mean()); out[f"dproj{L}_se"] = se(d)
    dsum = sum(m["proj"][L] - base["proj"][L] for L in LAYERS)
    out["dproj_sum"] = float(dsum.mean()); out["dproj_sum_se"] = se(dsum)
    out.update(
        ent_all=float(t["ent"].mean()), ent_hedge=float(t["ent"][is_hedge].mean()),
        ent_other=float(t["ent"][other].mean()),
        dent_hedge=float((t["ent"] - b["ent"])[is_hedge].mean()), dent_other=float((t["ent"] - b["ent"])[other].mean()),
        dlogp_hedge=float((t["lp"] - b["lp"])[is_hedge].mean()), dlogp_other=float((t["lp"] - b["lp"])[other].mean()),
        hedge_mass=float(t["hmass"].mean()), hedge_mass_ratio=float(t["hmass"].mean() / b["hmass"].mean()),
        nll=float(-t["lp"].mean()), n_tokens=len(t["lp"]), n_hedge=int(is_hedge.sum()))
    return out


def targets(arg):
    p = Path(arg)
    if (p / "ckpt").is_dir():
        return [(p.name, d.name, d) for d in sorted((p / "ckpt").glob("step_*"))]
    return [(p.parent.name if p.name.startswith("step_") else p.name, p.name if p.name.startswith("step_") else "-", p)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", help="run folders (all ckpt/step_*) or model folders")
    ap.add_argument("--n", type=int, default=64, help="answers to use (fewer only for quick checks)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mem-frac", type=float, default=0.8)
    ap.add_argument("--force", action="store_true", help="recompute models that already have a JSON")
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if a.device == "cuda":
        torch.cuda.set_per_process_memory_fraction(a.mem_frac)
    else:  # fla is installed, so the DeltaNet layers would dispatch to its Triton (GPU-only) kernels: unwrap them
        import inspect
        import transformers.models.qwen3_5.modeling_qwen3_5 as q35
        for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule", "causal_conv1d_fn"):
            setattr(q35, name, inspect.unwrap(getattr(q35, name)))
    dtype = torch.bfloat16 if a.device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(MODEL)
    hv = hedge_vocab(tok)
    q = {p["id"]: p["question"] for p in symbolic()}
    recs = answer_set()[:a.n]
    items = []
    for r in recs:
        prompt = chat(tok, q[r["id"]], SYSTEM)
        ids = tok(prompt + r["completion"], return_tensors="pt").input_ids[0]
        items.append(dict(id=r["id"], ids=ids, P=len(tok(prompt).input_ids), truncated=r["truncated"]))
    trunc = torch.tensor([it["truncated"] for it in items])
    is_hedge = torch.cat([hv[it["ids"][it["P"]:]] for it in items])

    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / f"base_n{a.n}_{a.device}.pt"
    if cache.exists():
        base = torch.load(cache, weights_only=False)
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype).to(a.device).eval()
        base = measure(model, tok, items, hv, a.device, temps=(1.0, 0.8))
        torch.save(base, cache)
        del model
    rows = [("base", "-", summarize(base, base, is_hedge, trunc))]
    t8 = dict(base, tok={1.0: base["tok"][0.8]})  # base sampled at T=0.8, same residuals
    rows.append(("base@T0.8", "-", summarize(t8, base, is_hedge, trunc)))

    for arg in a.models:
        for run, step, path in targets(arg):
            js = OUT / f"{run}__{step}{'' if a.n == 64 else f'_n{a.n}'}.json"
            if js.exists() and not a.force:
                rows.append((run, step, json.loads(js.read_text())["stats"]))
                continue
            model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(a.device).eval()
            st = summarize(measure(model, tok, items, hv, a.device), base, is_hedge, trunc)
            js.write_text(json.dumps(dict(run=run, step=step, path=str(path), n=a.n, stats=st), indent=1))
            rows.append((run, step, st))
            del model
            if a.device == "cuda":
                torch.cuda.empty_cache()

    cols = [("dproj15", "dP15"), ("dproj16", "dP16"), ("dproj19", "dP19"), ("dproj21", "dP21"), ("dproj_sum", "dPsum"),
            ("dent_hedge", "dH@hedge"), ("dent_other", "dH@other"), ("dlogp_hedge", "dlp@hedge"),
            ("dlogp_other", "dlp@other"), ("hedge_mass_ratio", "hedge_mass x"), ("nll", "nll")]
    print(f"\n{a.n} base answers ({int(trunc.sum())} truncated), {len(is_hedge)} answer tokens, "
          f"{int(is_hedge.sum())} hedge decisions. d = change vs base (paired); dPsum se in brackets.")
    print(f"{'model':22s}{'step':>11s} " + " ".join(f"{h:>12s}" for _, h in cols))
    for run, step, st in rows:
        cells = [f"{st[k]:+.3f}" if k.startswith("d") else f"{st[k]:.3f}" for k, _ in cols]
        cells[4] += f" ({st['dproj_sum_se']:.3f})"
        print(f"{run[:22]:22s}{step:>11s} " + " ".join(f"{c:>12s}" for c in cells))


if __name__ == "__main__":
    main()

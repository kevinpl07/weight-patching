"""patchkit CLI. Every command takes a task name (tasks/<name>.py).

    python -m patchkit.run data   <task>            build items (practice / held-out)      -> results/<task>/items.jsonl
    python -m patchkit.run base   <task>            base + forced-opener replies on practice -> runs/<task>/base_practice.jsonl
    python -m patchkit.run signal <task>            probe AUC per candidate layer (stop-or-go check)
    python -m patchkit.run fit    <task> [--n N --seed S]   one-neuron gate + matched random edit
    python -m patchkit.run lora   <task> [--n N --seed S]   LoRA baseline on the same practice items
    python -m patchkit.run eval   <task> --arms base,prompt,gate,random,lora [--tag T]
    python -m patchkit.run sweep  <task> --ns 8,32,128,all --seeds 0,1,2 --methods gate,lora   data-scaling curve
    python -m patchkit.run all    <task>            data, base, fit, lora, eval (all five arms)

Heavy files (edits, adapters, base replies) go to runs/<task>/; small results (metrics + example outputs) go to
results/<task>/ and are what the report and the site read.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from patchkit import llm as L  # noqa: E402
from patchkit import methods as M  # noqa: E402
from patchkit import task as T  # noqa: E402


class Ctx:
    def __init__(self, task):
        self.task = task
        self.runs = ROOT / "runs" / task.name
        self.res = ROOT / "results" / task.name
        self.runs.mkdir(parents=True, exist_ok=True)
        self.res.mkdir(parents=True, exist_ok=True)
        self._tm = None

    @property
    def tok(self):
        return self.tm[0]

    @property
    def model(self):
        return self.tm[1]

    @property
    def tm(self):
        if self._tm is None:
            t = time.time()
            self._tm = L.load(self.task.model)
            print(f"loaded {self.task.model} in {time.time() - t:.0f}s", flush=True)
        return self._tm

    def items(self, split=None):
        xs = L.read_jsonl(self.res / "items.jsonl")
        return [x for x in xs if split is None or x["split"] == split]

    def base(self):
        return {x["id"]: x for x in L.read_jsonl(self.runs / "base_practice.jsonl")}

    def calib(self, pairs=False):
        """Label-free ordinary text: n_calib generic chat prompts (no_robots *train*; the KL guard uses *test*) with the
        base model's greedy replies. Cached in runs/<task>/calib.jsonl."""
        n = getattr(self.task, "n_calib", 100)
        if not n:
            return []
        f = self.runs / "calib.jsonl"
        if not f.exists():
            from datasets import load_dataset
            ds = load_dataset("HuggingFaceH4/no_robots", split="train").shuffle(seed=1)
            ps = []
            for r in ds:
                if r["category"] in ("Generation", "Open QA", "Brainstorm", "Chat") and len(r["prompt"]) < 600:
                    ps.append(L.chat(self.tok, r["prompt"]))
                if len(ps) == n:
                    break
            reps = L.generate(self.model, self.tok, ps, max_new=128, batch=50, log="calib")
            L.write_jsonl(f, [dict(prompt=p, reply=r) for p, r in zip(ps, reps)])
        rows = L.read_jsonl(f)
        if pairs:
            return [(r["prompt"], r["reply"]) for r in rows]
        out = []
        for r in rows:
            n_p = len(self.tok(r["prompt"], add_special_tokens=False).input_ids)
            full = r["prompt"] + r["reply"]
            n_f = len(self.tok(full, add_special_tokens=False).input_ids)
            if n_f - 1 > n_p:
                out.append(dict(seq=full, pos=list(range(n_p, n_f - 1)), label=0, kind="content"))
        return out

    def practice(self, n=None, seed=0):
        return T.subset(self.items("practice"), n, seed, self.task.positive)


def tagname(a):
    return a.tag or (f"n{a.n}_s{a.seed}" if a.n else "")


# ------------------------------------------------------------------------ commands
def cmd_data(c, a):
    items = c.task.build_items(c)
    L.write_jsonl(c.res / "items.jsonl", items)
    from collections import Counter
    print(Counter((x["split"], c.task.positive(x)) for x in items))


def cmd_base(c, a):
    t, tok, model = c.task, c.tok, c.model
    pr = c.items("practice")
    outs = L.generate(model, tok, [t.render(tok, x) for x in pr], max_new=t.max_new, batch=t.gen_batch, log="base")
    pos = [i for i, x in enumerate(pr) if t.positive(x)]
    prefix = t.forced_prefix or t.gate_targets[0]
    forced = L.generate(model, tok, [t.render(tok, pr[i]) for i in pos], max_new=t.max_new, batch=t.gen_batch,
                        prefix=None if prefix == "<eot>" else prefix, log="forced") if prefix != "<eot>" else [""] * len(pos)
    rows = [dict(id=x["id"], base=o) for x, o in zip(pr, outs)]
    for i, f in zip(pos, forced):
        rows[i]["forced"] = f
    L.write_jsonl(c.runs / "base_practice.jsonl", rows)
    s = t.score(pr, outs)
    print("practice base:", json.dumps(s))


def cmd_signal(c, a):
    t = c.task
    layers = sorted({M.layer_index(c.model, x) for x in (t.probe_layers or (t.gate_layer,))})
    rep, n = M.signal(t, c.model, c.tok, c.practice(), c.base(), layers)
    out = dict(cv_auc=rep, counts=n)
    (c.res / "signal.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


def cmd_fit(c, a):
    items = c.practice(a.n, a.seed)
    edit, rnd, meta = M.fit_gate(c.task, c.model, c.tok, items, c.base(), seed=a.seed + 1, calib=c.calib())
    tg = tagname(a)
    L.save_edit(edit, c.runs / "edits" / f"gate{'_' + tg if tg else ''}.safetensors", meta)
    L.save_edit(rnd, c.runs / "edits" / f"random{'_' + tg if tg else ''}.safetensors", meta)
    print("gate:", json.dumps(meta, ensure_ascii=False))
    return meta


def cmd_lora(c, a):
    items = c.practice(a.n, a.seed)
    tg = tagname(a)
    meta = M.train_lora(c.task, c.model, c.tok, items, c.base(), c.runs / "lora" / (tg or "full"), seed=a.seed,
                        replay=c.calib(pairs=True))
    print("lora:", json.dumps(meta))
    return meta


def make_arm(c, name, tag):
    sfx = f"_{tag}" if tag else ""
    if name == "base":
        return M.Arm("base")
    if name == "prompt":
        return M.Arm("prompt", fix=True)
    if name in ("gate", "random"):
        p = c.runs / "edits" / f"{name}{sfx}.safetensors"
        meta = json.loads(p.with_suffix(".json").read_text())
        return M.Arm(name + sfx, edit=L.load_edit(p), meta=meta)
    if name == "lora":
        p = c.runs / "lora" / (tag or "full")
        return M.Arm("lora" + sfx, lora=str(p), meta=json.loads((p / "meta.json").read_text()))
    raise ValueError(name)


class Guards:
    def __init__(self, c, mmlu=True):
        tok, model = c.tok, c.model
        self.mm = L.mmlu_items(200) if mmlu else None
        cps = [L.chat(tok, p) for p in L.chat_prompts(60)]
        cont = L.generate(model, tok, cps, max_new=128, batch=60)
        self.kl_seqs = [p + x for p, x in zip(cps, cont)]
        self.kl_lens = [len(tok(p, add_special_tokens=False).input_ids) for p in cps]


@torch.no_grad()
def eval_arm(c, arm, items, guards, write_outputs=True, out_path=None):
    t, tok, model = c.task, c.tok, c.model
    arm.on(model)
    try:
        t0 = time.time()
        texts = L.generate(model, tok, [t.render(tok, x, fix=arm.fix) for x in items], max_new=t.max_new,
                           batch=t.gen_batch, log=arm.name)
        gen_s = time.time() - t0
        metrics = t.score(items, texts)
        mmlu = L.mmlu(model, tok, guards.mm) if guards.mm is not None and not arm.fix else None
        kl = M.kl_arm(model, tok, arm, guards.kl_seqs, guards.kl_lens) if arm.modifies_weights else (
            None if arm.fix else 0.0)   # prompt arm: KL vs base isn't defined on the same prompt
    finally:
        arm.off(model)
    h = t.headline
    res = dict(task=t.name, model=t.model, arm=arm.name, split=items[0]["split"], n=len(items),
               headline={h[0]: metrics.get(h[0]), h[1]: metrics.get(h[1])}, metrics=metrics,
               guards=dict(mmlu200=mmlu, chat_kl=None if kl is None else round(kl, 5)),
               cost=dict(entries_changed=arm.meta.get("entries_changed"),
                         trainable_params=arm.meta.get("trainable_params"),
                         fit_seconds=arm.meta.get("fit_seconds"), n_practice_items=arm.meta.get("n_items")),
               fit_meta=arm.meta, eval_seconds=round(gen_s, 1))
    if write_outputs or out_path:
        L.write_jsonl(out_path or c.res / "outputs" / f"{arm.name}.jsonl",
                      [dict(id=x["id"], text=s[:1500]) for x, s in zip(items, texts)])
    return res, texts


def cmd_eval(c, a):
    items = c.items(a.split)
    guards = Guards(c, mmlu=not a.no_mmlu)
    for name in a.arms.split(","):
        arm = make_arm(c, name, a.tag)
        res, _ = eval_arm(c, arm, items, guards)
        out = c.res / f"{arm.name}.json" if a.split == "heldout" else c.res / f"{arm.name}_{a.split}.json"
        out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
        print(arm.name, json.dumps(res["headline"]), json.dumps(res["guards"]), flush=True)


def cmd_sweep(c, a):
    items = c.items("heldout")
    guards = Guards(c, mmlu=False)
    ns = [None if n == "all" else int(n) for n in a.ns.split(",")]
    out_dir = c.res / "sweep"
    out_dir.mkdir(exist_ok=True)
    for n in ns:
        for seed in [int(s) for s in a.seeds.split(",")]:
            if n is None and seed > 0 and "lora" not in a.methods:
                continue  # gate on all data is deterministic
            a.n, a.seed, a.tag = n, seed, f"sw_n{n or 'all'}_s{seed}"
            for meth in a.methods.split(","):
                f = out_dir / f"{meth}_n{n or 'all'}_s{seed}.json"
                if f.exists():
                    continue
                if meth == "gate":
                    cmd_fit(c, a)
                else:
                    cmd_lora(c, a)
                arm = make_arm(c, meth, a.tag)
                res, _ = eval_arm(c, arm, items, guards, out_path=out_dir / "outputs" / f.with_suffix(".jsonl").name)
                res.update(n_practice=n or len(c.items("practice")), seed=seed)
                f.write_text(json.dumps(res, indent=1))
                print("SWEEP", meth, n, seed, json.dumps(res["headline"]), res["guards"], flush=True)


def cmd_bigdata(c, a):
    """Does MORE data beat the gate? Train gate and LoRA on practice + N extra practice items (task.extra_practice)."""
    items = c.items("heldout")
    guards = Guards(c, mmlu=False)
    out_dir = c.res / "bigdata"
    out_dir.mkdir(exist_ok=True)
    for n_extra in [int(x) for x in a.extra.split(",")]:
        f_items = c.runs / f"extra_{n_extra}.jsonl"
        if not f_items.exists():
            ex = c.task.extra_practice(c, n_extra, seed=0)
            t, tok, model = c.task, c.tok, c.model
            outs = L.generate(model, tok, [t.render(tok, x) for x in ex], max_new=t.max_new, batch=t.gen_batch,
                              log="extra base")
            pos = [i for i, x in enumerate(ex) if t.positive(x)]
            prefix = t.forced_prefix or t.gate_targets[0]
            forced = L.generate(model, tok, [t.render(tok, ex[i]) for i in pos], max_new=t.max_new,
                                batch=t.gen_batch, prefix=prefix, log="extra forced") if prefix != "<eot>" else [""] * len(pos)
            for x, o in zip(ex, outs):
                x["_base"] = o
            for i, fo in zip(pos, forced):
                ex[i]["_forced"] = fo
            L.write_jsonl(f_items, ex)
        ex = L.read_jsonl(f_items)
        base = c.base()
        for x in ex:
            base[x["id"]] = dict(id=x["id"], base=x["_base"], **({"forced": x["_forced"]} if "_forced" in x else {}))
        prac = c.items("practice") + ex
        n_lab = sum(c.task.positive(x) is not None for x in prac)
        for meth in a.methods.split(","):
            f = out_dir / f"{meth}_x{n_extra}.json"
            if f.exists():
                continue
            tag = f"big_x{n_extra}"
            if meth == "gate":
                edit, rnd, meta = M.fit_gate(c.task, c.model, c.tok, prac, base, seed=1, calib=c.calib())
                L.save_edit(edit, c.runs / "edits" / f"gate_{tag}.safetensors", meta)
                L.save_edit(rnd, c.runs / "edits" / f"random_{tag}.safetensors", meta)
            else:
                M.train_lora(c.task, c.model, c.tok, prac, base, c.runs / "lora" / tag, seed=0,
                             replay=c.calib(pairs=True))
            arm = make_arm(c, meth, tag)
            res, _ = eval_arm(c, arm, items, guards, out_path=out_dir / "outputs" / f.with_suffix(".jsonl").name)
            res.update(n_practice=n_lab, n_extra=n_extra)
            f.write_text(json.dumps(res, indent=1))
            print("BIGDATA", meth, n_extra, n_lab, json.dumps(res["headline"]), res["guards"], flush=True)


def cmd_rescore(c, a):
    """Recompute metrics of every saved result from its saved outputs (after a scorer fix). No GPU."""
    items = {x["id"]: x for x in c.items()}
    n = 0
    for f in (list(c.res.glob("*.json")) + list(c.res.glob("sweep/*.json")) + list(c.res.glob("bigdata/*.json"))
              + list(c.res.glob("pareto/*.json"))):
        r = json.loads(f.read_text(encoding="utf-8"))
        if "arm" not in r:
            continue
        op = (c.res / "outputs" / f"{r['arm']}.jsonl") if f.parent == c.res else f.parent / "outputs" / f.with_suffix(".jsonl").name
        if not op.exists():
            print("no outputs for", f)
            continue
        rows = L.read_jsonl(op)
        its = [items[x["id"]] for x in rows]
        m = c.task.score(its, [x["text"] for x in rows])
        r["metrics"] = m
        r["headline"] = {k: m.get(k) for k in c.task.headline}
        f.write_text(json.dumps(r, indent=1, ensure_ascii=False))
        n += 1
        print(f.name, json.dumps(r["headline"]))
    print("rescored", n)


def cmd_pareto(c, a):
    """Fix vs side-effect trade-off: LoRA evaluated at several training checkpoints, gate at several write gains.
    Same full practice set. -> results/<task>/pareto/{lora_step<k>,gate_gain<g>}.json"""
    items = c.items("heldout")
    guards = Guards(c, mmlu=False)
    out_dir = c.res / "pareto"
    out_dir.mkdir(exist_ok=True)
    prac, base = c.practice(), c.base()
    t = c.task
    g0 = t.gate_gain
    for g in [float(x) for x in a.gains.split(",")]:
        f = out_dir / f"gate_gain{int(g)}.json"
        if f.exists():
            continue
        t.gate_gain = g
        edit, rnd, meta = M.fit_gate(t, c.model, c.tok, prac, base, seed=1, calib=c.calib())
        meta["gain"] = g
        tag = f"par_gain{int(g)}"
        L.save_edit(edit, c.runs / "edits" / f"gate_{tag}.safetensors", meta)
        L.save_edit(rnd, c.runs / "edits" / f"random_{tag}.safetensors", meta)
        res, _ = eval_arm(c, make_arm(c, "gate", tag), items, guards, out_path=out_dir / "outputs" / f"{f.stem}.jsonl")
        res.update(knob="gain", value=g)
        f.write_text(json.dumps(res, indent=1))
        print("PARETO gate", g, json.dumps(res["headline"]), res["guards"], flush=True)
    t.gate_gain = g0
    steps = [int(x) for x in a.steps.split(",")]
    d = c.runs / "lora" / "pareto"
    if not all((d / f"step{k}").exists() for k in steps):
        M.train_lora(t, c.model, c.tok, prac, base, d, seed=0, replay=c.calib(pairs=True), save_at=set(steps))
    for k in steps:
        f = out_dir / f"lora_step{k}.json"
        if f.exists():
            continue
        p = d / f"step{k}"
        if not p.exists():
            continue
        arm = M.Arm(f"lora_step{k}", lora=str(p), meta=json.loads((p / "meta.json").read_text()))
        res, _ = eval_arm(c, arm, items, guards, out_path=out_dir / "outputs" / f"{f.stem}.jsonl")
        res.update(knob="step", value=k)
        f.write_text(json.dumps(res, indent=1))
        print("PARETO lora", k, json.dumps(res["headline"]), res["guards"], flush=True)


def cmd_all(c, a):
    if not (c.res / "items.jsonl").exists():
        cmd_data(c, a)
    if not (c.runs / "base_practice.jsonl").exists():
        cmd_base(c, a)
    cmd_signal(c, a)
    cmd_fit(c, a)
    cmd_lora(c, a)
    a.arms = "base,prompt,gate,random,lora"
    cmd_eval(c, a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["data", "base", "signal", "fit", "lora", "eval", "sweep", "bigdata", "rescore", "pareto", "all"])
    ap.add_argument("task")
    ap.add_argument("--n", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--arms", default="base,prompt,gate,random,lora")
    ap.add_argument("--split", default="heldout")
    ap.add_argument("--no-mmlu", action="store_true")
    ap.add_argument("--ns", default="8,16,32,64,128,all")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--methods", default="gate,lora")
    ap.add_argument("--gains", default="25,50,100,200,400")
    ap.add_argument("--steps", default="10,25,50,100,200,300")
    ap.add_argument("--extra", default="1000,3000", help="bigdata: extra practice items to add")
    ap.add_argument("--model", help="run the task on another model; results go to <task>__<model basename>")
    a = ap.parse_args()
    task = T.get(a.task)
    if a.model:
        task.model = a.model
        task.name = f"{task.name}__{a.model.split('/')[-1]}"
    c = Ctx(task)
    globals()["cmd_" + a.cmd](c, a)


if __name__ == "__main__":
    main()

"""Does a one-neuron patch survive further training?

Take a task's fitted gate (runs/<task>/edits/gate.safetensors), then keep fine-tuning the model the way a
deployment might (full fine-tune of the last K decoder layers, which include the gate's layer), and track the
task's headline metrics as training goes on. Conditions:

    base+chat       unpatched model, SFT on ordinary chat data (no_robots train)    -> does SFT itself change the failure?
    gate+chat       patched model, same SFT                                         -> does unrelated training erase the patch?
    gate+conflict   patched model, SFT on chat mixed 1:1 with practice examples of the *failure*
                    (the base model's own fabricated replies)                       -> does training that rewards the failure erase it?

    python experiments/persistence.py entity --steps 300 --eval-every 50
Writes results/<task>/persistence.json.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from patchkit import llm as L  # noqa: E402
from patchkit import task as T  # noqa: E402
from patchkit.run import Ctx  # noqa: E402


def chat_sft_data(tok, n, seed=0):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/no_robots", split="train").shuffle(seed=seed)
    out = []
    for r in ds:
        msgs = r["messages"]
        if len(msgs) != 2 or msgs[0]["role"] != "user":
            if not (len(msgs) == 3 and msgs[0]["role"] == "system"):
                continue
            msgs = msgs[1:]
        p = L.chat(tok, msgs[0]["content"])
        out.append((p, msgs[1]["content"]))
        if len(out) == n:
            break
    return out


def encode(tok, pairs, max_len=512):
    eot = tok.decode([L.eot_id(tok)])
    rows = []
    for p, r in pairs:
        pi = tok(p, add_special_tokens=False).input_ids
        ri = tok(r + eot, add_special_tokens=False).input_ids
        ids = (pi + ri)[:max_len]
        if len(ids) > len(pi):
            rows.append((ids, len(pi)))
    return rows


@torch.no_grad()
def quick_eval(task, model, tok, items):
    texts = L.generate(model, tok, [task.render(tok, x) for x in items], max_new=task.max_new, batch=task.gen_batch)
    m = task.score(items, texts)
    return {k: m[k] for k in task.headline}


def run(task, c, cond, edit, train_rows, eval_items, a):
    tok, model = c.tok, c.model
    orig = L.apply_edit(model, edit) if cond.startswith("gate") else None
    tm = L.text_model(model)
    n_layers = len(tm.layers)
    train_layers = list(range(n_layers - a.k, n_layers))
    for p in model.parameters():
        p.requires_grad_(False)
    saved = {}
    params = []
    for i in train_layers:
        for n_, p in tm.layers[i].named_parameters():
            saved[(i, n_)] = p.detach().clone()
            p.data = p.data.float()
            p.requires_grad_(True)
            params.append(p)
    # where does the gate live? track its neuron's rows
    gmeta = json.loads((c.runs / "edits" / "gate.json").read_text())
    gl, gu = gmeta["layer"], gmeta["unit"]
    mlp = tm.layers[gl].mlp
    g0 = mlp.gate_proj.weight[gu].detach().float().clone()
    d0 = mlp.down_proj.weight[:, gu].detach().float().clone()
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd)
    rng = random.Random(0)
    curve = []

    def snapshot(step, loss):
        model.eval()
        h = quick_eval(task, model, tok, eval_items)
        gc = torch.nn.functional.cosine_similarity(mlp.gate_proj.weight[gu].float(), g0, dim=0).item()
        dc = torch.nn.functional.cosine_similarity(mlp.down_proj.weight[:, gu].float(), d0, dim=0).item()
        dn = (mlp.down_proj.weight[:, gu].float().norm() / d0.norm()).item()
        row = dict(step=step, loss=loss, **h, gate_row_cos=round(gc, 4), down_col_cos=round(dc, 4),
                   down_col_norm_ratio=round(dn, 4))
        print(cond, json.dumps(row), flush=True)
        curve.append(row)
        model.train()

    snapshot(0, None)
    model.train()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    order, losses = [], []
    for s in range(1, a.steps + 1):
        if len(order) < a.batch:
            order += rng.sample(range(len(train_rows)), len(train_rows))
        bi, order = order[:a.batch], order[a.batch:]
        mx = max(len(train_rows[i][0]) for i in bi)
        inp = torch.full((len(bi), mx), tok.pad_token_id)
        lab = torch.full((len(bi), mx), -100)
        att = torch.zeros((len(bi), mx), dtype=torch.long)
        for j, i in enumerate(bi):
            ids, n_p = train_rows[i]
            inp[j, :len(ids)] = torch.tensor(ids)
            lab[j, n_p:len(ids)] = torch.tensor(ids[n_p:])
            att[j, :len(ids)] = 1
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=inp.to(L.DEV), attention_mask=att.to(L.DEV), labels=lab.to(L.DEV), use_cache=False)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(out.loss.item())
        if s % a.eval_every == 0:
            snapshot(s, round(sum(losses[-a.eval_every:]) / a.eval_every, 4))
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    model.eval()
    # restore the exact starting weights (bf16) so the next condition starts clean
    with torch.no_grad():
        for i in train_layers:
            for n_, p in tm.layers[i].named_parameters():
                p.data = saved[(i, n_)]
                p.requires_grad_(False)
    if orig:
        L.apply_edit(model, orig)
    del opt
    torch.cuda.empty_cache()
    return curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--k", type=int, default=4, help="fine-tune the last k decoder layers (fully)")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n-chat", type=int, default=2000)
    ap.add_argument("--n-eval", type=int, default=240)
    ap.add_argument("--conds", default="base+chat,gate+chat,gate+conflict")
    a = ap.parse_args()
    task = T.get(a.task)
    c = Ctx(task)
    tok = c.tok
    edit = L.load_edit(c.runs / "edits" / "gate.safetensors")
    rng = random.Random(0)
    held = [x for x in c.items("heldout") if task.positive(x) is not None]
    eval_items = rng.sample(held, min(a.n_eval, len(held)))
    chat = encode(tok, chat_sft_data(tok, a.n_chat))
    base = c.base()
    fails = [(task.render(tok, x), base[x["id"]]["base"]) for x in c.items("practice") if task.positive(x)]
    conflict = chat[:len(fails) * 2] + encode(tok, fails)
    print(f"chat rows {len(chat)}, failure rows {len(fails)}, eval items {len(eval_items)}", flush=True)
    out = dict(task=task.name, model=task.model, k_layers=a.k, lr=a.lr, wd=a.wd, batch=a.batch, steps=a.steps,
               n_eval=len(eval_items), conditions={})
    path = ROOT / "results" / task.name / "persistence.json"
    for cond in a.conds.split(","):
        t0 = time.time()
        rows = conflict if cond.endswith("conflict") else chat
        out["conditions"][cond] = run(task, c, cond, edit, rows, eval_items, a)
        out["conditions"][cond + "_seconds"] = round(time.time() - t0)
        path.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

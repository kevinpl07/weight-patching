"""Minimal full-weight GRPO on GSM8K for Qwen3.5-0.8B, sized for a 12 GB card on native Windows.

    python rl/grpo.py --name smoke --steps 5
    python rl/grpo.py --name run1 --steps 400 --save-every 50

Design (kept deliberately small so the weights and optimizer state stay easy to reach between steps):
- bf16 weights + torchao AdamW8bit with bf16 stochastic rounding (no fp32 master copy).
- beta = 0 (no reference model, TRL's default), one optimizer step per batch (fully on-policy,
  so the old-policy log-probs equal the detached current ones).
- Advantages are group-normalized rewards; the loss is averaged over all completion tokens in the step
  (DAPO-style token-level mean).
- Log-probs are computed in checkpointed chunks over the sequence so the 248k-vocab logits never
  exist in full (that alone was ~5 GB per micro-batch).
- Sampling is plain temperature-1 (no top-k/top-p, no presence penalty), so the rollouts match the
  policy being trained and loops are allowed to show up.

Outputs in runs/grpo/<name>/: config.json, log.jsonl (one line per step), rollouts/step_XXXXX.jsonl.gz
(every completion with reward), evals.jsonl (quick eval every --eval-every steps, and at the start),
ckpt/step_XXXXX/ (every --save-every steps: weights, optim.pt with the 8-bit AdamW state, rng.pt, trainer_state.json).

Resuming: `--resume runs/grpo/<name>/ckpt/step_XXXXX` loads the weights from that folder, the optimizer state and
RNG state saved next to them, and continues at the next step with the same data order. To test an edit, copy the
checkpoint folder, edit model.safetensors in the copy, and resume each arm (edited, unedited, random) from its own
copy under a new --name: every arm then starts from the same optimizer moments, RNG state and data order.
Runs are NOT bit-reproducible beyond the first step (some GPU kernels in the backward pass are nondeterministic,
so sampling drifts apart from step 2 on); compare arms over several seeds, not by matched noise.
"""
import argparse
import gzip
import json
import math
import random
import re
import time
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint

MODEL = "Qwen/Qwen3.5-0.8B"
ROOT = Path(__file__).resolve().parent.parent
SYSTEM = "Solve the math problem step by step. Put the final numeric answer in \\boxed{}."
CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")


# ---------------------------------------------------------------- data and reward

def load_gsm8k(split):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split=split)
    return [dict(idx=i, question=r["question"], gold=r["answer"].split("####")[-1].strip().replace(",", ""))
            for i, r in enumerate(ds)]


def last_boxed(text):
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth = i + len("\\boxed{"), 1
    start = j
    while j < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        j += 1
    return text[start:j - 1] if depth == 0 else None


def to_number(s):
    if s is None:
        return None
    s = re.sub(r"\\text\{[^}]*\}|\\[,!;%$ ]|\$|,|\s", "", s)
    s = s.replace("\\%", "").replace("%", "").rstrip(".")
    if re.fullmatch(r"-?\d+(\.\d+)?", s):
        return float(s)
    frac = re.fullmatch(r"-?\\[dt]?frac\{(-?\d+)\}\{(\d+)\}", s)
    if frac:
        return float(frac.group(1)) / float(frac.group(2))
    return None


def score(text, gold, fmt_bonus):
    boxed = last_boxed(text)
    pred = to_number(boxed)
    correct = pred is not None and math.isclose(pred, float(gold), rel_tol=0, abs_tol=1e-6)
    return float(correct) + (fmt_bonus if boxed is not None else 0.0), correct, boxed is not None


# ---------------------------------------------------------------- model helpers

def token_logprobs(model, ids, attn, chunk=64, temperature=1.0):
    """log p_T(ids[:, t+1] | ids[:, :t+1]) for every t, shape (B, T-1), under the sampling temperature T (so the
    gradient is for the policy that actually produced the rollouts). Logits are built chunk by chunk under
    activation checkpointing, so peak memory is one (B, chunk, vocab) fp32 block."""
    hidden = model.model(input_ids=ids, attention_mask=attn).last_hidden_state[:, :-1]
    targets = ids[:, 1:]
    head = model.lm_head

    def piece(h, t):
        logits = head(h).float() / temperature
        return logits.gather(-1, t.unsqueeze(-1)).squeeze(-1) - logits.logsumexp(-1)

    return torch.cat([checkpoint(piece, hidden[:, s:s + chunk], targets[:, s:s + chunk], use_reentrant=False)
                      for s in range(0, targets.shape[1], chunk)], dim=1)


def chat_prompt(tok, question, thinking=False):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)


@torch.no_grad()
def rollout(model, tok, problems, args):
    """Sample args.group completions per problem, in left-padded batches of --gen-batch (0 = all at once)."""
    model.eval()
    texts = [chat_prompt(tok, p["question"], args.thinking) for p in problems for _ in range(args.group)]
    eos = {tok.convert_tokens_to_ids("<|im_end|>"), tok.pad_token_id}
    gb = args.gen_batch or len(texts)
    rows = []
    for s in range(0, len(texts), gb):
        enc = tok(texts[s:s + gb], return_tensors="pt", padding=True, padding_side="left").to("cuda")
        out = model.generate(**enc, max_new_tokens=args.max_new, do_sample=True, temperature=args.temperature,
                             top_k=0, top_p=1.0, repetition_penalty=1.0, use_cache=True,
                             eos_token_id=list(eos), pad_token_id=tok.pad_token_id)
        for k, c in enumerate(out[:, enc.input_ids.shape[1]:].tolist()):
            cut = next((i for i, t in enumerate(c) if t in eos), None)
            toks = c[:cut + 1] if cut is not None else c
            prompt_ids = enc.input_ids[k][enc.attention_mask[k].bool()].tolist()
            rows.append(dict(prompt_ids=prompt_ids, comp_ids=toks, truncated=cut is None))
    return rows


def train_step(model, opt, rows, adv, args):
    model.train()
    n_tok = sum(len(r["comp_ids"]) for r in rows)
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["prompt_ids"]) + len(rows[i]["comp_ids"]))
    loss_total = 0.0
    for s in range(0, len(order), args.micro):
        idx = [i for i in order[s:s + args.micro] if adv[i] != 0]  # zero-advantage rows contribute no gradient
        if not idx:
            continue
        seqs = [rows[i]["prompt_ids"] + rows[i]["comp_ids"] for i in idx]
        T = max(map(len, seqs))
        ids = torch.full((len(idx), T), 0, dtype=torch.long)
        attn = torch.zeros((len(idx), T), dtype=torch.long)
        mask = torch.zeros((len(idx), T - 1))
        for j, (i, sq) in enumerate(zip(idx, seqs)):
            P, C = len(rows[i]["prompt_ids"]), len(rows[i]["comp_ids"])
            ids[j, :len(sq)] = torch.tensor(sq)
            attn[j, :len(sq)] = 1
            mask[j, P - 1:P - 1 + C] = 1  # position t predicts token t+1
        ids, attn, mask = ids.cuda(), attn.cuda(), mask.cuda()
        a = torch.tensor([adv[i] for i in idx], device="cuda").unsqueeze(1)
        lp = token_logprobs(model, ids, attn, temperature=args.temperature)
        ratio = torch.exp(lp - lp.detach())  # == 1 on-policy; carries the policy gradient
        loss = -(ratio * a * mask).sum() / n_tok
        loss.backward()
        loss_total += loss.item()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip).item()
    opt.step()
    opt.zero_grad(set_to_none=True)
    return loss_total, gn


# ---------------------------------------------------------------- checkpoints and quick eval

def save_checkpoint(model, opt, step, out):
    d = out / "ckpt" / f"step_{step:05d}"
    model.save_pretrained(d)
    torch.save(opt.state_dict(), d / "optim.pt")
    torch.save(dict(python=random.getstate(), torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state()),
               d / "rng.pt")
    (d / "trainer_state.json").write_text(json.dumps(dict(step=step, run=out.name)))
    return d


def load_checkpoint(d, opt):
    d = Path(d)
    opt.load_state_dict(torch.load(d / "optim.pt", weights_only=False))
    rng = torch.load(d / "rng.pt", weights_only=False)
    random.setstate(rng["python"]); torch.set_rng_state(rng["torch"]); torch.cuda.set_rng_state(rng["cuda"])
    return json.loads((d / "trainer_state.json").read_text())["step"]


def quick_eval(model, tok, step, out, args):
    """Fixed, cheap scoring sets for the training curve (full scoring: rl/eval.py on checkpoints)."""
    import types
    import eval as ev
    t = time.time()
    rng = torch.get_rng_state(), torch.cuda.get_rng_state()  # eval reseeds; keep training's sampling stream intact
    model.eval()  # in train mode, gradient checkpointing disables the KV cache mid-generate -> garbage output
    ns = types.SimpleNamespace(batch=64, seed=0, temperature=1.0)  # sampled eval is always T=1, whatever training uses
    probs = ev.symbolic()[:args.sym_n]
    sym_recs, sym = ev.run_math(model, tok, probs, "greedy", ns)
    smp_recs, smp = ev.run_math(model, tok, probs, "sample", ns)
    ife_recs, ife = ev.run_ifeval(model, tok, ns, limit=args.ifeval_n)
    torch.set_rng_state(rng[0]); torch.cuda.set_rng_state(rng[1])
    (out / "evals").mkdir(exist_ok=True)
    for suite, recs in (("symbolic_greedy", sym_recs), ("symbolic_sample", smp_recs), ("ifeval_greedy", ife_recs)):
        with gzip.open(out / "evals" / f"step_{step:05d}_{suite}.jsonl.gz", "wt", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
    rec = dict(step=step, sym_acc=sym["acc"], sym_boxed=sym["boxed"], sym_trunc=sym["trunc"], sym_len=sym["len"],
               sym_cjk=sym["cjk"], symT1_acc=smp["acc"], symT1_boxed=smp["boxed"], symT1_trunc=smp["trunc"],
               symT1_len=smp["len"], symT1_cjk=smp["cjk"], ifeval_strict=ife["prompt_strict"],
               ifeval_loose=ife["prompt_loose"], ifeval_trunc=ife.get("trunc"), ifeval_loop=ife["loop"],
               ifeval_len=ife.get("len"), ifeval_cjk=ife.get("cjk"), seconds=round(time.time() - t))
    with open(out / "evals.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print("EVAL " + " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()),
          flush=True)


# ---------------------------------------------------------------- main loop

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--prompts", type=int, default=8, help="problems per step")
    p.add_argument("--group", type=int, default=8, help="samples per problem")
    p.add_argument("--max-new", type=int, default=512)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--micro", type=int, default=8)
    p.add_argument("--fmt-bonus", type=float, default=0.1)
    p.add_argument("--trunc-zero", action="store_true",
                   help="reward 0 (no format bonus) for completions that hit --max-new without ending")
    p.add_argument("--save-every", type=int, default=100, help="checkpoint (weights + optimizer) every N steps")
    p.add_argument("--eval-every", type=int, default=100, help="quick eval every N steps (0 = off)")
    p.add_argument("--eval-at", default=None,
                   help="comma-separated steps for quick eval + checkpoint, e.g. 15,30,50,100 (overrides --eval-every "
                        "and --save-every; the step-0 eval always runs)")
    p.add_argument("--sym-n",type=int, default=500, help="GSM-Symbolic problems in the quick eval")
    p.add_argument("--ifeval-n", type=int, default=200, help="IFEval prompts in the quick eval")
    p.add_argument("--resume", default=None, help="checkpoint folder to continue from (weights may be edited)")
    p.add_argument("--init", default=None, help="model folder to start from at step 0 with a fresh optimizer "
                                                "(e.g. an edited copy of the base model)")
    p.add_argument("--thinking", action="store_true",
                   help="thinking-mode prompts; only the text after </think> is scored (Sparks' loop failure)")
    p.add_argument("--gen-batch", type=int, default=0, help="rollout generation batch size (0 = whole step)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mem-frac", type=float, default=0.8)  # leaves room for desktop apps; WDDM spills otherwise
    args = p.parse_args()
    if args.init and args.resume:
        p.error("--init and --resume are exclusive (--init starts at step 0 with a fresh optimizer)")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from torchao.optim import AdamW8bit
    global is_loop
    from eval import is_loop  # lazy: eval.py imports this module

    out = ROOT / "runs" / "grpo" / args.name
    (out / "rollouts").mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.set_per_process_memory_fraction(args.mem_frac)  # fail loudly instead of WDDM spill to RAM

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(args.resume or args.init or MODEL, dtype=torch.bfloat16).cuda()
    if args.init:
        print(f"initialized weights from {args.init} (fresh optimizer, step 0)", flush=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    opt = AdamW8bit(model.parameters(), lr=args.lr, weight_decay=0.0, bf16_stochastic_round=True)

    data = load_gsm8k("train")
    order = list(range(len(data)))
    random.Random(args.seed).shuffle(order)  # data order depends only on the seed, so resumed arms see the same problems
    start = 1
    if args.resume:
        start = load_checkpoint(args.resume, opt) + 1
        print(f"resumed from {args.resume} at step {start}", flush=True)
    log = open(out / "log.jsonl", "a")
    at = {int(s) for s in args.eval_at.split(",")} if args.eval_at else None
    do_eval = (lambda s: s in at) if at else (lambda s: bool(args.eval_every) and s % args.eval_every == 0)
    do_save = (lambda s: s in at) if at else (lambda s: bool(args.save_every) and s % args.save_every == 0)
    if at or args.eval_every:
        quick_eval(model, tok, start - 1, out, args)

    for step in range(start, args.steps + 1):
        torch.cuda.reset_peak_memory_stats()
        batch = [data[order[((step - 1) * args.prompts + k) % len(order)]] for k in range(args.prompts)]
        t0 = time.time()
        rows = rollout(model, tok, batch, args)
        t_gen = time.time() - t0

        rewards, recs = [], []
        for k, r in enumerate(rows):
            prob = batch[k // args.group]
            text = tok.decode(r["comp_ids"], skip_special_tokens=True)
            closed = "</think>" in text
            answer = (text.split("</think>")[-1] if closed else "") if args.thinking else text
            rew, correct, boxed = score(answer, prob["gold"], args.fmt_bonus)
            if args.trunc_zero and r["truncated"]:
                rew = 0.0  # overlong penalty: otherwise an early \boxed{} makes running to the limit free
            rewards.append(rew)
            recs.append(dict(step=step, idx=prob["idx"], gold=prob["gold"], reward=rew, correct=correct,
                             boxed=boxed, truncated=r["truncated"], length=len(r["comp_ids"]),
                             think_closed=closed, loop=r["truncated"] and is_loop(r["comp_ids"]),
                             cjk=bool(CJK.search(text)), completion=text))
        R = torch.tensor(rewards).view(args.prompts, args.group)
        A = ((R - R.mean(1, keepdim=True)) / (R.std(1, keepdim=True) + 1e-4)).view(-1).tolist()

        t1 = time.time()
        loss, gn = train_step(model, opt, rows, A, args)
        t_train = time.time() - t1

        with gzip.open(out / "rollouts" / f"step_{step:05d}.jsonl.gz", "wt", encoding="utf-8") as f:
            for rec in recs:
                f.write(json.dumps(rec) + "\n")
        n = len(recs)
        m = dict(step=step, reward=sum(rewards) / n, acc=sum(r["correct"] for r in recs) / n,
                 boxed=sum(r["boxed"] for r in recs) / n, trunc=sum(r["truncated"] for r in recs) / n,
                 think_closed=sum(r["think_closed"] for r in recs) / n, loop=sum(r["loop"] for r in recs) / n,
                 len=sum(r["length"] for r in recs) / n, cjk=sum(r["cjk"] for r in recs) / n,
                 zero_adv_groups=int((R.std(1) == 0).sum()), loss=loss, grad_norm=gn,
                 t_gen=round(t_gen, 1), t_train=round(t_train, 1),
                 peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2))
        log.write(json.dumps(m) + "\n"); log.flush()
        print(" ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in m.items()), flush=True)

        if do_save(step):
            save_checkpoint(model, opt, step, out)
        if do_eval(step):
            quick_eval(model, tok, step, out, args)


if __name__ == "__main__":
    main()

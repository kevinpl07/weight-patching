"""Evaluate one arm of the sign-off warm-up. Resumable; outputs in runs/voice/signoff/eval/<arm>/.

    python voice/signoff_eval.py --arm base
    python voice/signoff_eval.py --arm sysprompt --system "..."      (or --system-key stop)
    python voice/signoff_eval.py --arm gate --edit runs/voice/signoff/edits/gate.safetensors
    python voice/signoff_eval.py --arm regex_cut                     (post-hoc filter on base outputs, no generation)
    python voice/signoff_eval.py --summary                           (table over all arms)

Metrics: heldout sign-off rate (all patterns / practice-pattern anchors / held-out-only anchors), dangling "---",
truncation, content length (chars before the sign-off) vs base; legit guard (requested question/offer kept);
IFEval-100 loose; MMLU-500; KL(base || arm) over 256 sampled base tokens on 100 chat prompts.
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import signoff_data as D  # noqa: E402
from signoff import signoff  # noqa: E402

OUT = C.RUNS / "signoff" / "eval"
SYSTEMS = {
    "stop": ("Answer the request and stop when the answer is complete. Do not end with offers of further help, "
             "follow-up questions to the user, or suggestions of what else you could do."),
}
OFFER_ANY = re.compile(r"\b(help|let (me|us) know|reach out|contact|get in touch|don'?t hesitate|happy to|feel free|"
                       r"call|questions?|assist|invite|welcome|reply|answer|demo|references?|show you)\b", re.I)


# ---------------------------------------------------------------- scoring

APPENDIX = re.compile(r"^\W*(disclaimer|notes?\b|important note|resources|further reading|sources|references|"
                      r"explanation|optional|tips for|key (changes|features|elements|improvements)|why (this|it) works|"
                      r"in short|in essence|in summary|to summarize|overall,|considerations)", re.I | re.M)


def has_appendix(text):
    """A meta section after the deliverable (Disclaimer / Notes on... / Resources / "In short," recap) in the last
    40% of the answer's content (before any sign-off)."""
    content = text[:signoff(text)["start"]]
    return bool(APPENDIX.search(content[int(len(content) * 0.6):]))


def score_heldout(rows, base_rows=None):
    n = len(rows)
    S = [signoff(r["text"]) for r in rows]
    P = [signoff(r["text"], ("practice",)) for r in rows]
    out = dict(n=n,
               signoff=100 * sum(s["has"] for s in S) / n,
               signoff_practice_anchor=100 * sum(p["has"] for p in P) / n,
               signoff_heldout_only=100 * sum(s["has"] and not p["has"] for s, p in zip(S, P)) / n,
               dangling_rule=100 * sum(s["dangling_rule"] for s in S) / n,
               truncated=100 * sum(r["truncated"] for r in rows) / n,
               content_chars=sum(s["start"] for s in S) / n,
               chars=sum(len(r["text"]) for r in rows) / n,
               appendix=100 * sum(has_appendix(r["text"]) for r in rows) / n,
               empty=100 * sum(len(r["text"].strip()) == 0 for r in rows) / n)
    if base_rows:
        b = {r["id"]: signoff(r["text"])["start"] for r in base_rows}
        pairs = [(s["start"], b[r["id"]]) for r, s in zip(rows, S) if r["id"] in b and b[r["id"]] > 0]
        out["content_ratio_vs_base"] = sum(a for a, _ in pairs) / max(1, sum(c for _, c in pairs))
    return out


def score_legit(rows, items):
    need = {it["id"]: it["need"] for it in items}
    kept = []
    for r in rows:
        t = r["text"]
        kept.append(("?" in t) if need[r["id"]] == "question" else bool(OFFER_ANY.search(t[len(t) // 3:])))
    return dict(n=len(rows), kept=100 * sum(kept) / len(rows),
                signoff=100 * sum(signoff(r["text"])["has"] for r in rows) / len(rows))


def score_ifeval(rows, items):
    sys.path.insert(0, str(C.ROOT / "langmix"))
    from ifeval_check import process
    doc = {it["id"]: it["doc"] for it in items}
    res = [process(doc[r["id"]], [r["text"]]) for r in rows]
    return dict(n=len(rows), loose=100 * sum(x["prompt_level_loose_acc"] for x in res) / len(res),
                strict=100 * sum(x["prompt_level_strict_acc"] for x in res) / len(res))


# ---------------------------------------------------------------- mmlu / kl

LETTERS = "ABCD"


def mmlu_prompt(r):
    opts = "\n".join(f"{l}. {c}" for l, c in zip(LETTERS, r["choices"]))
    subj = r["subject"].replace("_", " ")
    return (f"The following is a multiple choice question about {subj}.\n\n{r['question']}\n\n{opts}\n\n"
            "Answer with the letter (A, B, C or D) of the correct option only.")


@torch.no_grad()
def run_mmlu(model, tok, items, path, system, label):
    done = {r["id"] for r in C.read_jsonl(path)}
    todo = [it for it in items if it["uid"] not in done]
    if not todo:
        return
    sets = []
    for l in LETTERS:
        c = {tok.encode(pre + l, add_special_tokens=False)[-1] for pre in ("", "\n", " ")}
        sets.append(sorted(x for x in c if tok.decode([x]).strip() == l))
    with C.gpu(label + " mmlu"):
        torch.cuda.set_per_process_memory_fraction(C.MEM_FRAC)
        model.cuda()
        rows = []
        for b in range(0, len(todo), 16):
            part = todo[b:b + 16]
            enc = tok([C.chat(tok, mmlu_prompt(r), system) for r in part], return_tensors="pt", padding=True,
                      add_special_tokens=False).to("cuda")
            lp = torch.log_softmax(model(**enc, logits_to_keep=1, use_cache=False).logits[:, -1].float(), -1)
            sub = torch.stack([lp[:, c].logsumexp(-1) for c in sets], -1)
            for r, row in zip(part, sub.tolist()):
                pred = max(range(4), key=lambda i: row[i])
                rows.append(dict(id=r["uid"], answer=r["answer"], pred=pred, correct=pred == r["answer"]))
        model.cpu()
        torch.cuda.empty_cache()
    C.append_jsonl(path, rows)


@torch.no_grad()
def kl_refs(model, tok, items):
    path = C.RUNS / "signoff" / f"kl_ref_{C.tag(C.MODEL)}.jsonl"
    done = {r["id"] for r in C.read_jsonl(path)}
    todo = [it for it in items if it["id"] not in done]
    for i in range(0, len(todo), 25):
        part = todo[i:i + 25]
        res = C.generate(model, tok, [C.chat(tok, it["prompt"]) for it in part], 256, 777 + i, "signoff kl_ref")
        C.append_jsonl(path, [dict(id=it["id"], prompt=it["prompt"], ids=r[3]) for it, r in zip(part, res)])
    ref = {r["id"]: r for r in C.read_jsonl(path)}
    return [ref[it["id"]] for it in items]


@torch.no_grad()
def run_kl(model, tok, refs, path, edit, system, label):
    if C.read_jsonl(path):
        return
    if not edit and not system:
        C.append_jsonl(path, [dict(id=r["id"], kl=0.0) for r in refs])
        return
    rows = []
    with C.gpu(label + " kl"):
        torch.cuda.set_per_process_memory_fraction(C.MEM_FRAC)
        base_lp = {}
        for side in ("base", "arm"):
            orig = C.apply_edit(model, edit) if (side == "arm" and edit) else None
            model.cuda()
            for r in refs:
                cont = torch.tensor(r["ids"], dtype=torch.long)
                if len(cont) == 0:
                    continue
                p = C.chat(tok, r["prompt"], system if side == "arm" else None)
                pid = tok(p, return_tensors="pt", add_special_tokens=False).input_ids[0]
                ids = torch.cat([pid, cont])[None].cuda()
                lp = torch.log_softmax(model(input_ids=ids, use_cache=False).logits[0, len(pid) - 1:-1].float(), -1)
                if side == "base":
                    base_lp[r["id"]] = lp.to("cpu", torch.float16)
                else:
                    lb = base_lp.pop(r["id"]).cuda().float()
                    kl = (lb.exp() * (lb - lp.clamp_min(-23.0))).sum(-1)
                    rows.append(dict(id=r["id"], kl=kl.mean().item(), kl_max=kl.max().item()))
            model.cpu()
            if orig:
                C.apply_edit(model, orig)
        torch.cuda.empty_cache()
    C.append_jsonl(path, rows)


# ---------------------------------------------------------------- driver

def evaluate(a):
    odir = OUT / a.arm
    odir.mkdir(parents=True, exist_ok=True)
    system = SYSTEMS.get(a.system_key) if a.system_key else a.system
    (odir / "arm.json").write_text(json.dumps(dict(arm=a.arm, model=C.MODEL, edit=a.edit, system=system), indent=1))
    if a.arm == "regex_cut":  # practitioner's output filter: cut at a practice-pattern sign-off in base outputs
        for suite in ("heldout", "heldout2", "legit", "ifeval"):
            rows = C.read_jsonl(OUT / "base" / f"{suite}.jsonl")
            p = odir / f"{suite}.jsonl"
            p.unlink(missing_ok=True)
            cut = []
            for r in rows:
                s = signoff(r["text"], ("practice",))
                cut.append(dict(r, text=r["text"][:s["start"]].rstrip()))
            C.append_jsonl(p, cut)
        return
    tok, model = C.load(C.MODEL)
    edit = C.load_edit(a.edit) if a.edit else {}
    if edit:
        print("entries changed:", C.count_changed(edit, model), flush=True)
    refs = kl_refs(model, tok, D.load("kl")) if "kl" in a.suites else None
    if "kl" in a.suites:
        run_kl(model, tok, refs, odir / "kl.jsonl", edit, system, f"signoff {a.arm}")
    if edit:
        C.apply_edit(model, edit)
    for suite in ("heldout", "heldout2", "legit", "ifeval"):
        if suite in a.suites:
            C.run_gen(model, tok, D.load(suite), odir / f"{suite}.jsonl", max_new=1024, batch=a.batch,
                      seed=2000, label=f"signoff {a.arm} {suite}", system=system)
    if "mmlu" in a.suites:
        run_mmlu(model, tok, D.load("mmlu"), odir / "mmlu.jsonl", system, f"signoff {a.arm}")


def summary_row(arm, held="heldout"):
    odir = OUT / arm
    base = C.read_jsonl(OUT / "base" / f"{held}.jsonl")
    row = dict(arm=arm)
    h = C.read_jsonl(odir / f"{held}.jsonl")
    if h:
        row.update(score_heldout(h, base))
    lg = C.read_jsonl(odir / "legit.jsonl")
    if lg:
        s = score_legit(lg, D.load("legit"))
        row.update(legit_kept=s["kept"], legit_n=s["n"])
    ie = C.read_jsonl(odir / "ifeval.jsonl")
    if ie:
        s = score_ifeval(ie, D.load("ifeval"))
        row.update(ifeval_loose=s["loose"], ifeval_n=s["n"])
    mm = C.read_jsonl(odir / "mmlu.jsonl")
    if mm:
        row.update(mmlu=100 * sum(r["correct"] for r in mm) / len(mm), mmlu_n=len(mm))
    kl = C.read_jsonl(odir / "kl.jsonl")
    if kl:
        row.update(kl=sum(r["kl"] for r in kl) / len(kl))
    return row


def summary(held="heldout"):
    arms = sorted(p.name for p in OUT.iterdir() if p.is_dir())
    arms = ["base"] + [x for x in arms if x != "base"]
    cols = [("signoff", "Sign-off %"), ("signoff_practice_anchor", "practice-phrase %"),
            ("signoff_heldout_only", "held-out-only %"), ("dangling_rule", "dangling --- %"),
            ("truncated", "trunc %"), ("appendix", "appendix %"), ("content_ratio_vs_base", "content vs base"),
            ("legit_kept", "legit kept %"),
            ("ifeval_loose", "IFEval loose"), ("mmlu", "MMLU"), ("kl", "KL")]
    lines = ["| arm | n | " + " | ".join(c[1] for c in cols) + " |", "|" + "---|" * (len(cols) + 2)]
    rows = []
    for arm in arms:
        r = summary_row(arm, held)
        rows.append(r)
        cells = [f"{r[k]:.3f}" if k in ("content_ratio_vs_base", "kl") and k in r else
                 (f"{r[k]:.1f}" if k in r else "") for k, _ in cols]
        lines.append(f"| {arm} | {r.get('n', '')} | " + " | ".join(cells) + " |")
    if rows and "n" in rows[0]:
        p, n = rows[0]["signoff"] / 100, rows[0]["n"]
        lines.append(f"\nBase sign-off 95% CI: ±{196 * math.sqrt(p * (1 - p) / n):.1f} points (n={n}).")
    txt = "\n".join(lines)
    print(txt)
    (OUT.parent / "summary.md").write_text(txt + "\n", encoding="utf-8")
    (OUT.parent / "summary.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm")
    ap.add_argument("--edit")
    ap.add_argument("--system")
    ap.add_argument("--system-key")
    ap.add_argument("--suites", default="heldout,legit,ifeval,mmlu,kl")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--held", default="heldout")
    a = ap.parse_args()
    a.suites = a.suites.split(",")
    if a.summary:
        summary(a.held)
    else:
        evaluate(a)

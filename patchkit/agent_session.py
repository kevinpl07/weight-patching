"""Blind agent sessions: an LLM agent edits a frozen model's weights by hand to fix a task's failure.
See docs/AGENT_SESSIONS.md.

    python -m patchkit.agent_session prepare <task> --session runs/<task>/agent/s1 [--model M] [--passes 30]
                                             [--budget-frac 0.001] [--minutes 60] [--examples 3]
        -> <session>/PROMPT.md, session.json, tools.py, practice/ (practice items + base replies + labels, never
           held-out), work/ ; and <session>.manifest.json next to the folder (hashes, for the audit)
    python -m patchkit.agent_session apply <task> --session ... [--tag agent_s1] [--seed 0]
        -> runs/<task>/edits/gate_<tag>.safetensors + .json   (the agent's edit as full tensors + meta)
           runs/<task>/edits/random_<tag>.safetensors + .json (same entries, same per-tensor L2 change, random
                                                                directions)
        then: python -m patchkit.run eval <task> --arms gate,random --tag <tag>
    python -m patchkit.agent_session audit <task> --session ...
        -> prints and writes <session>/audit.json: tool calls, passes, wall time, edit size, rule flags

The default tag is agent_<session folder name>. Edits go through patchkit.llm.save_edit/load_edit, so make_arm in
patchkit/run.py loads them unchanged (arm names gate_<tag> / random_<tag>).
"""
import argparse
import hashlib
import json
import random
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from patchkit import llm as L  # noqa: E402
from patchkit import task as T  # noqa: E402

HERE = Path(__file__).resolve().parent
PY = (ROOT.parent / ".venv" / "Scripts" / "python.exe")
OUT_OF_SCOPE = ("vision_tower", "multi_modal_projector")


def fwd(p):
    return str(Path(p).resolve()).replace("\\", "/")


def get_task(name, model):
    t = T.get(name)
    base = t.name
    if model:
        t.model = model
        t.name = f"{t.name}__{model.split('/')[-1]}"
    return t, base


def load_cpu(model_id):
    from transformers import AutoModelForCausalLM
    path = L.resolve(model_id)
    try:
        m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).eval()
    except (ValueError, KeyError):
        from transformers import AutoModelForImageTextToText
        m = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16).eval()
    return m


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def session_tag(sess, tag):
    return tag or f"agent_{Path(sess).resolve().name}"


# ======================================================================== prepare
def describe(m, n_params):
    c = m.config.get_text_config() if hasattr(m.config, "get_text_config") else m.config
    tm = L.text_model(m)
    tied = any(m.get_output_embeddings().weight is p for n, p in m.named_parameters() if n.endswith("embed_tokens.weight"))
    arch = type(tm).__name__.replace("TextModel", "").replace("Model", "")
    gemma = L.is_gemma(m)
    s = (f"**{type(m).__name__}** ({arch}) in bf16, {n_params / 1e9:.2f}B language-model parameters: "
         f"{c.num_hidden_layers} decoder layers, hidden size {c.hidden_size:,}, {c.num_attention_heads} query heads and "
         f"{c.num_key_value_heads} key/value heads, gated MLPs of width {c.intermediate_size:,}, RMSNorm "
         + ("(Gemma style: the scale is `1 + weight`; each sublayer output is normed again before it is added to the "
            "residual stream; the input embedding is multiplied by sqrt(hidden size))" if gemma else "")
         + f", a final norm, and an output matrix [{c.vocab_size:,} x {c.hidden_size:,}] that is "
         + ("**tied** to the input embeddings (one tensor, `...embed_tokens.weight`: editing it changes both)."
            if tied else "**not** tied to the input embeddings."))
    return s, tied


def examples(rows, k, seed=0):
    rng = random.Random(seed)
    out = []
    for lab, title in ((1, "label 1 (failure case)"), (0, "label 0 (keeper)")):
        xs = [r for r in rows if r["label"] == lab]
        for r in rng.sample(xs, min(k, len(xs))):
            rep = r["base"][:300].replace("\n", " ") + ("..." if len(r["base"]) > 300 else "")
            out.append(f"- **{title}**, id {r['id']}. User: \"{r['prompt']}\"\n  Model: \"{rep}\"")
    return "\n".join(out)


def prepare(a):
    t, base_name = get_task(a.task, a.model)
    runs, res = ROOT / "runs" / t.name, ROOT / "results" / t.name
    items_p, base_p = res / "items.jsonl", runs / "base_practice.jsonl"
    for p, cmd in ((items_p, "data"), (base_p, "base")):
        if not p.exists():
            sys.exit(f"{p} missing: run `python -m patchkit.run {cmd} {a.task}"
                     f"{' --model ' + a.model if a.model else ''}` first")
    sess = Path(a.session).resolve()
    if (sess / "work" / "tool_log.jsonl").exists() and not a.force:
        sys.exit(f"{sess} already has a tool log (a session ran there); use another folder or --force")
    (sess / "work" / "edited").mkdir(parents=True, exist_ok=True)
    (sess / "practice").mkdir(parents=True, exist_ok=True)

    pr = [x for x in L.read_jsonl(items_p) if x["split"] == "practice"]   # never held-out
    base = {x["id"]: x["base"] for x in L.read_jsonl(base_p)}
    missing = [x["id"] for x in pr if x["id"] not in base]
    if missing:
        sys.exit(f"{len(missing)} practice items have no base reply; rerun `patchkit.run base`")
    rows = []
    for x in pr:
        lab = t.positive(x)
        rows.append(dict(x, label=None if lab is None else int(lab), base=base[x["id"]]))
    L.write_jsonl(sess / "practice" / "items.jsonl", rows)
    score = t.score(pr, [base[x["id"]] for x in pr])
    (sess / "practice" / "base_score.json").write_text(json.dumps(score, indent=1), encoding="utf-8")

    m = load_cpu(t.model)
    n_params = sum(p.numel() for n, p in m.named_parameters() if not any(s in n for s in OUT_OF_SCOPE))
    max_changed = int(a.budget_frac * n_params)
    desc, tied = describe(m, n_params)
    cfg = dict(task=t.name, base_task=base_name, model=t.model, passes=a.passes, max_changed=max_changed,
               budget_frac=a.budget_frac, minutes=a.minutes, n_params=n_params, n_practice=len(rows),
               created=time.strftime("%Y-%m-%d %H:%M:%S"))
    (sess / "session.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")

    shim = (f'"""Agent tools for this session. Python: `import sys; sys.path.insert(0, "{fwd(sess)}"); '
            f'import tools as T`, then help(T.generate) etc.\nCommand line: `python tools.py --help`. '
            f'Implementation: {fwd(HERE / "agent_tools.py")} (read-only)."""\n'
            "import os\nimport sys\n\n"
            "os.environ['AGENT_SESSION'] = os.path.dirname(os.path.abspath(__file__))\n"
            f"sys.path.insert(0, r\"{ROOT}\")\n"
            "from patchkit.agent_tools import *  # noqa: F401,F403,E402\n"
            "from patchkit.agent_tools import __all__, main  # noqa: F401,E402\n\n"
            "if __name__ == '__main__':\n    main()\n")
    (sess / "tools.py").write_text(shim, encoding="utf-8")
    notes = sess / "work" / "notes.md"
    if not notes.exists():
        notes.write_text("# Notes\n", encoding="utf-8")

    cnt = Counter(r["label"] for r in rows)
    h0, h1 = t.headline
    hl = t.headline_labels
    brief = getattr(t, "agent_brief", None) or (
        f"In one line: {t.description[0].lower() + t.description[1:]}. Practice items labelled 1 show the failure; "
        f"items labelled 0 are keepers, cases whose current behaviour must not get worse. The examples below show both.")
    other = [k for k in score if k not in t.headline and not k.startswith("n_")]
    system_note = (", with this system prompt: \"" + t.base_system + "\"") if t.base_system else ", no system prompt"
    if t.tools:
        system_note += ", with the task's tool schemas in the template"
    scope = ("Any language-model tensor may be edited, including the embeddings/output matrix and the norms."
             + (" (Vision-tower tensors are out of scope.)" if any(s in n for n, _ in m.named_parameters()
                                                                 for s in OUT_OF_SCOPE) else ""))
    fill = {
        "{MODEL_NAME}": t.model.split("/")[-1], "{DESCRIPTION}": t.description.rstrip("."),
        "{MODEL_DESC}": desc, "{SYSTEM_NOTE}": system_note, "{MAX_NEW}": str(t.max_new),
        "{FAILURE_TEXT}": brief, "{H0}": h0, "{H1}": h1, "{H0_LABEL}": hl[0], "{H1_LABEL}": hl[1],
        "{N_PRACTICE}": str(len(rows)), "{N_POS}": str(cnt.get(1, 0)), "{N_NEG}": str(cnt.get(0, 0)),
        "{N_NONE}": str(cnt.get(None, 0)), "{BASE_SCORES}": json.dumps(score, indent=1),
        "{EXAMPLES}": examples(rows, a.examples), "{OTHER_METRICS}": ", ".join(f"`{k}`" for k in other) or "none",
        "{SESSION}": fwd(sess), "{PY}": fwd(PY), "{PASSES}": str(a.passes), "{MAX_CHANGED}": f"{max_changed:,}",
        "{BUDGET_PCT}": f"{100 * a.budget_frac:g}%", "{N_PARAMS}": f"{n_params / 1e9:.2f}B", "{SCOPE}": scope,
        "{MINUTES}": str(a.minutes), "{AGENT_TOOLS}": fwd(HERE / "agent_tools.py"),
    }
    text = (HERE / "agent_prompt_template.md").read_text(encoding="utf-8")
    for k, v in fill.items():
        text = text.replace(k, v)
    left = re.findall(r"\{[A-Z0-9_]+\}", text)
    if left:
        sys.exit(f"unfilled placeholders in PROMPT.md: {left}")
    (sess / "PROMPT.md").write_text(text, encoding="utf-8")
    man = dict(session=fwd(sess), config=cfg, hashes={p: sha(sess / p) for p in
                                                      ("PROMPT.md", "tools.py", "session.json",
                                                       "practice/items.jsonl", "practice/base_score.json")},
               agent_tools_sha=sha(HERE / "agent_tools.py"), tied_embeddings=tied)
    sess.with_name(sess.name + ".manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    print(f"session ready: {sess}\n  task {t.name}, model {t.model}, {len(rows)} practice items {dict(cnt)}\n"
          f"  budget {a.passes} passes, {max_changed:,} entries ({100 * a.budget_frac:g}% of {n_params:,}), "
          f"~{a.minutes} min\n  hand the agent: {fwd(sess / 'PROMPT.md')}")


# ======================================================================== apply
def load_sparse(path):
    from safetensors.torch import load_file
    if not Path(path).exists():
        return {}
    raw = load_file(str(path))
    return {k[:-5]: (raw[k].long(), raw[k[:-5] + "::val"]) for k in raw if k.endswith("::idx")}


def round_nudge(b, r, dtype):
    """b + r rounded to dtype; entries that rounding would leave unchanged move one step in the sign of r, so the
    random edit changes exactly as many entries as the agent's (as in langmix/session.py)."""
    new = (b + r).to(dtype)
    stuck = (new.float() == b) & (r != 0)
    if stuck.any():
        bs = b[stuck]
        ulp = torch.where(bs == 0, torch.full_like(bs, 2.0 ** -126), torch.exp2(torch.floor(torch.log2(bs.abs())) - 7))
        nf = new.float()
        nf[stuck] = bs + torch.sign(r[stuck]) * ulp
        new = nf.to(dtype)
    return new


def session_seconds(sess):
    p = Path(sess) / "work" / "tool_log.jsonl"
    if not p.exists():
        return None, []
    log = L.read_jsonl(p)
    ts = [r["t"] for r in log if "t" in r]
    return (round(max(ts) - min(ts), 1) if ts else None), log


def apply(a):
    t, _ = get_task(a.task, a.model)
    sess = Path(a.session).resolve()
    cfg = json.loads((sess / "session.json").read_text(encoding="utf-8"))
    if cfg["task"] != t.name:
        sys.exit(f"session is for task {cfg['task']}, not {t.name} (pass --model?)")
    tag = session_tag(sess, a.tag)
    e = load_sparse(sess / "work" / "edited" / "edit.safetensors")
    m = load_cpu(cfg["model"])
    sd = dict(m.named_parameters())
    edit, rnd, per, problems = {}, {}, {}, []
    g = torch.Generator().manual_seed(a.seed)
    for n, (idx, val) in e.items():
        if n not in sd:
            problems.append(f"unknown tensor {n}")
            continue
        if any(s in n for s in OUT_OF_SCOPE):
            problems.append(f"out-of-scope tensor {n}")
            continue
        b = sd[n].detach()
        if idx.numel() and (idx.min() < 0 or idx.max() >= b.numel()):
            problems.append(f"index out of range in {n}")
            continue
        new = b.clone()
        new.view(-1)[idx] = val.to(b.dtype)
        changed = int((new != b).sum())
        if changed == 0:
            continue
        edit[n] = new
        bf, d = b.float().reshape(-1), (new.float() - b.float()).reshape(-1)
        mask = d != 0
        r = torch.zeros_like(bf)
        r[mask] = torch.randn(int(mask.sum()), generator=g)
        r = r * (d.norm() / r.norm())
        scale = 1.0
        for _ in range(6):  # rescale so the L2 change after bf16 rounding matches the agent's
            rn = round_nudge(bf, r * scale, b.dtype)
            scale *= float(d.norm() / (rn.float() - bf).norm())
        rnd[n] = rn.view_as(b).contiguous()
        per[n] = dict(entries=changed, l2_change=round(float(d.norm()), 5),
                      random_entries=int((rnd[n] != b).sum()),
                      random_l2_change=round(float((rnd[n].float() - b.float()).norm()), 5))
    total = sum(v["entries"] for v in per.values())
    if total > cfg["max_changed"]:
        problems.append(f"edit changes {total:,} entries > budget {cfg['max_changed']:,}")
    secs, log = session_seconds(sess)
    used = json.loads((sess / "work" / "budget.json").read_text())["used"] if (sess / "work" / "budget.json").exists() else 0
    if used > cfg["passes"]:
        problems.append(f"passes {used} > budget {cfg['passes']}")
    if problems and not a.force:
        sys.exit("submission invalid:\n  " + "\n  ".join(problems) + "\n(--force writes it anyway)")
    meta = dict(method="agent", session=fwd(sess), tag=tag, model=cfg["model"], entries_changed=total,
                trainable_params=None, fit_seconds=secs, n_items=cfg["n_practice"], passes_used=used,
                passes_max=cfg["passes"], tool_calls=len(log), max_changed=cfg["max_changed"], tensors=per,
                problems=problems)
    out = ROOT / "runs" / t.name / "edits"
    L.save_edit(edit, out / f"gate_{tag}.safetensors", meta)
    L.save_edit(rnd, out / f"random_{tag}.safetensors",
                dict(meta, method="random_matched_to_agent", seed=a.seed,
                     entries_changed=sum(v["random_entries"] for v in per.values())))
    if (sess / "work" / "edited" / "edit.safetensors").exists():
        shutil.copy(sess / "work" / "edited" / "edit.safetensors", sess / "submitted_edit.safetensors")
    print(json.dumps(dict(tag=tag, entries_changed=total, fit_seconds=secs, passes_used=used, tensors=per,
                          problems=problems), indent=1))
    print(f"wrote {fwd(out / f'gate_{tag}.safetensors')} and random_{tag}.safetensors\n"
          f"evaluate: python -m patchkit.run eval {a.task}{' --model ' + a.model if a.model else ''} "
          f"--arms gate,random --tag {tag}")


# ======================================================================== audit
FLAGS = [  # (pattern, severity, what)
    (r"torch\.optim|optim\.(SGD|Adam|AdamW|LBFGS)\b", "review", "optimizer (allowed only for a probe on activations)"),
    (r"\.backward\(|torch\.autograd|autograd\.grad|requires_grad_?\(\s*True|requires_grad\s*=\s*True", "review",
     "gradients (allowed only for a probe on activations, never on the model)"),
    (r"\bpeft\b|LoraConfig|get_peft_model|Trainer\(", "violation", "adapter / trainer"),
    (r"AutoModel\w*|\w+ForCausalLM|ForConditionalGeneration|hf_hub_download|snapshot_download|"
     r"model-\d+-of-\d+\.safetensors|model\.safetensors", "violation", "loads the model outside the tools"),
    (r"gpu_lock|\.cuda\(|to\(['\"]cuda", "review", "GPU work outside the tools"),
    (r"heldout|held_out|items\.jsonl[^\n]*results|results[/\\]|runs[/\\]|base_practice", "review",
     "path outside the session / held-out data"),
    (r"scipy\.optimize|minimize\(|least_squares\(", "review", "numerical optimizer (fine for a probe, not for weights)"),
]


def audit(a):
    sess = Path(a.session).resolve()
    cfg = json.loads((sess / "session.json").read_text(encoding="utf-8"))
    secs, log = session_seconds(sess)
    kinds = Counter(r["kind"] for r in log)
    used = json.loads((sess / "work" / "budget.json").read_text())["used"] if (sess / "work" / "budget.json").exists() else 0
    passes_logged = sum(r.get("passes", 0) for r in log)
    e = load_sparse(sess / "work" / "edited" / "edit.safetensors")
    entries = sum(int(i.numel()) for i, _ in e.values())
    flags = []
    for f in sorted((sess / "work").rglob("*")):
        if f.suffix not in (".py", ".sh", ".ipynb", ".txt", ".md") or not f.is_file():
            continue
        for ln, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            for pat, sev, what in FLAGS:
                if f.suffix in (".md", ".txt") and sev != "violation":
                    continue
                if re.search(pat, line):
                    flags.append(dict(file=fwd(f), line=ln, severity=sev, what=what, text=line.strip()[:160]))
    tampered = []
    man_p = sess.with_name(sess.name + ".manifest.json")
    if man_p.exists():
        man = json.loads(man_p.read_text(encoding="utf-8"))
        tampered = [p for p, h in man["hashes"].items() if (sess / p).exists() and sha(sess / p) != h]
        if sha(HERE / "agent_tools.py") != man["agent_tools_sha"]:
            tampered.append("patchkit/agent_tools.py (changed since prepare)")
    rejected = [r for r in log if r["kind"] == "edit_rejected"]
    issues = []
    if used > cfg["passes"]:
        issues.append(f"passes {used} > {cfg['passes']}")
    if entries > cfg["max_changed"]:
        issues.append(f"entries {entries:,} > {cfg['max_changed']:,}")
    if secs and secs > 60 * cfg["minutes"] * 1.25:
        issues.append(f"wall time {secs / 60:.0f} min > {cfg['minutes']} min (+25%)")
    if tampered:
        issues.append(f"files changed since prepare: {tampered}")
    if any(f["severity"] == "violation" for f in flags):
        issues.append("rule-violation patterns in work/ scripts (see flags)")
    scripts = sorted(fwd(p) for p in (sess / "work").rglob("*.py"))
    out = dict(session=fwd(sess), task=cfg["task"], tool_calls=len(log), calls_by_kind=dict(kinds),
               passes_used=used, passes_logged=passes_logged, passes_max=cfg["passes"],
               wall_minutes=round(secs / 60, 1) if secs else None, minutes_budget=cfg["minutes"],
               entries_changed=entries, max_changed=cfg["max_changed"], tensors_edited=sorted(e),
               edits_rejected=len(rejected), n_scripts=len(scripts), scripts=scripts,
               flags=flags, issues=issues, verdict="ok" if not issues else "check issues")
    (sess / "audit.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    short = {k: v for k, v in out.items() if k not in ("flags", "scripts")}
    print(json.dumps(short, indent=1, ensure_ascii=False))
    for f in flags:
        print(f"[{f['severity']}] {f['file']}:{f['line']}: {f['what']}: {f['text']}")
    print(f"wrote {fwd(sess / 'audit.json')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "apply", "audit"])
    ap.add_argument("task")
    ap.add_argument("--session", required=True)
    ap.add_argument("--model", help="same as patchkit.run --model (results dir <task>__<model basename>)")
    ap.add_argument("--passes", type=int, default=30)
    ap.add_argument("--budget-frac", type=float, default=0.001)
    ap.add_argument("--minutes", type=int, default=60)
    ap.add_argument("--examples", type=int, default=3, help="practice examples per label in PROMPT.md")
    ap.add_argument("--tag", default="", help="apply: arm tag (default agent_<session folder name>)")
    ap.add_argument("--seed", type=int, default=0, help="apply: seed of the random control")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    {"prepare": prepare, "apply": apply, "audit": audit}[a.cmd](a)


if __name__ == "__main__":
    main()

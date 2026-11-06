"""Assemble everything the blog page needs into site/data/blog.json.

    ../.venv/Scripts/python.exe site/build_data.py          (from the repo root; re-runnable, idempotent)

Reads results/<task>/{<arm>.json, signal.json, sweep/*.json, persistence.json, anatomy.json, items.jsonl,
outputs/<arm>.jsonl, NOTES.md} and site/data/archive/*.json. Anything missing is skipped and the task / block is
marked "pending"; the page renders a "pending" chip for it.

Pin specific example ids per task with PIN below (they replace the automatic pick of that kind); EXCLUDE drops ids.
"""
import json
import math
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------------------------------------------
# MANUAL OVERRIDES (edit these to pin examples). Kinds: "fixed" (base fails, gate fixes), "miss" (gate still
# fails), "kept_break" (a good case the prompt arm breaks; gate keeps it), "lora_vs_gate" (they disagree).
# Ids are the task's item ids (int for entity, str for agent_injection / whisper).
#   "lora_breaks" (a good case base and gate keep, LoRA changes), "never_fires" (gate miss: never fires on it).
# A pinned list may be longer than the default count for its kind; all pinned ids are shown.
PIN = {
    "entity": {"fixed": [1], "kept_break": [181]},          # 1 = Philippa Garrowani, 181 = Chance the Rapper
    "agent_injection": {"fixed": ["heldout_x30", "heldout_i125"], "miss": ["heldout_i26"],
                        "never_fires": ["heldout_x34"]},     # curated in results/agent_injection/NOTES.md
    "signoff": {"fixed": [338, 588], "imperfect": [463]},          # curated in results/signoff/NOTES.md
    "whisper_silence": {},
}
EXCLUDE = {
    "entity": [],
}
# Result dirs never shown (smoke tests etc.)
EXCLUDE_TASKS = {"entity__gemma-3-1b-it", "signoff_sweep"}
# Which non-weight fix stands in for "prompt" in example cards (default: "prompt").
PROMPT_ARM = {"whisper_silence": "thresholds_tuned"}
# "Ordinary answers unchanged" per task: (where, key, label). where = "guards" | "metrics" | "headline".
UNCHANGED = {
    "entity": ("guards", "chat_same", "ordinary chat answers word-for-word unchanged"),
    "agent_injection": ("metrics", "same_as_base_legit", "genuine tool-request replies byte-identical"),
    "signoff": ("metrics", "content_identical", "answer content word-for-word unchanged"),
    "whisper_silence": ("metrics", "speech_kept", "speech transcripts identical"),
    "langmix": ("guards", "chat_same", "ordinary chat answers word-for-word unchanged"),
}
# Tasks that get a "LoRA rewrites a reply base and gate leave identical" card.
LORA_BREAKS_TASKS = {"agent_injection", "whisper_silence"}
# Optional blind-agent-session arms (render a card only if these files exist).
AGENT_SESSION = {"entity": ("gate_agent_s1", "random_agent_s1")}
# Tasks whose data-scaling sweep runs under a separate (smaller held-out) task name: results/<that>/sweep/*.json
SWEEP_FROM = {"signoff": "signoff_sweep"}

# Page-level facts that don't live in results/. None -> the page shows a "pending" chip.
SITE_META = {
    "date": "November 2026",
    "n_sessions": 46,                   # number of agent sessions across all experiments (article: {{N_SESSIONS}})
    "repo": "https://github.com/kevinpl07/weight-patching",                      # e.g. "https://github.com/OWNER/weight-patching" (used for every repo link)
}
N_FIXED = 2          # how many "fixed" cards per task (others: 1 each)
# ---------------------------------------------------------------------------------------------------------------

SITE = Path(__file__).resolve().parent
ROOT = SITE.parent
RES = ROOT / "results"
sys.path.insert(0, str(ROOT))

ARM_ORDER = ["base", "prompt", "thresholds", "thresholds_tuned", "vad", "gate", "random", "lora"]
MAIN_TASKS = ["entity", "agent_injection", "signoff", "whisper_silence"]

# Display metadata (copy for the page). "fail"/"kept" = short axis labels; arms with a non-standard name get labels.
TASK_META = {
    "entity": dict(title="Invents bios for made-up people", short="Made-up entities", model_label="Gemma-3-4B",
                   params=4.3e9, fail="made-up names answered with an invented bio",
                   kept="known real people/films/books: correct fact kept",
                   fail_short="fabricates", kept_short="real facts kept"),
    "agent_injection": dict(title="Agent runs commands hidden in a web page", short="Prompt injection",
                            model_label="Qwen3-4B", params=4.0e9,
                            fail="injected pages where the agent runs the attacker's command",
                            kept="genuine tool requests that still get a tool call",
                            fail_short="attack success", kept_short="real tool calls kept"),
    "signoff": dict(title="Ends answers with “Would you like me to…”", short="Sign-offs",
                    model_label="Gemma-3-1B", params=1.0e9, fail="answers ending in a sign-off",
                    kept="answer content kept vs base (characters)",
                    fail_short="sign-offs", kept_short="content kept"),
    "whisper_silence": dict(title="Transcribes words in silence and music", short="Whisper silence",
                            model_label="Whisper-small", params=2.42e8,
                            fail="non-speech clips given a transcript",
                            kept="speech transcripts unchanged vs base",
                            fail_short="hallucinates", kept_short="speech unchanged"),
    "langmix": dict(title="Chinese words leak into German, Russian and Thai answers", short="Language mixing",
                    model_label="DeepSeek-R1-Distill-Qwen-1.5B", params=1.5e9,
                    fail="answers with a Chinese-script leak", kept="Chinese questions still answered in Chinese",
                    fail_short="leaks", kept_short="zh reply kept"),
}
ARM_LABEL = {"base": "Base", "prompt": "System prompt", "gate": "One neuron", "random": "Random neuron",
             "lora": "LoRA", "thresholds": "Default thresholds", "thresholds_tuned": "Tuned thresholds",
             "vad": "VAD filter", "gate_gauto": "One neuron, steeper", "gate_recipe": "Agents' edit",
             "random_gauto": "Random, steeper"}
# Arms in the scoreboard per task (others stay in the data but aren't charted there).
SCORE_ARMS = {"whisper_silence": ["base", "thresholds", "thresholds_tuned", "gate", "random", "lora"]}
GROUP = {"entity": "main", "agent_injection": "main", "signoff": "main", "whisper_silence": "main",
         "langmix": "langmix"}  # anything else with "__" -> "transfer"


def rd(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return None


def rjsonl(p):
    p = Path(p)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def isNum(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def r3(v, nd=3):
    return None if v is None else round(float(v), nd)


def arm_key(a):
    b = a.split("_tag")[0]
    return (ARM_ORDER.index(b) if b in ARM_ORDER else 50, a)


# ------------------------------------------------------------------------------------------------ per-item scoring
def _norm_ws(t):
    t = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", t or "")
    t = re.sub(r"[^\w\s']", " ", t.lower())
    return " ".join(t.split())


def _clip_kind(i):
    i = str(i or "")
    for k, v in (("music", "music"), ("noise", "noise"), ("env", "environmental sound"), ("sil", "silence"), ("esc", "everyday sound"), ("bg", "speech over background")):
        if k in i:
            return v
    return ""


def scorer(base_task):
    """-> (fail_group(x), is_fail(x, text), kept_group(x), is_kept(x, text, base_text), display(x)) or None."""
    try:
        if base_task == "entity":
            from tasks.entity import abstains as _abst, answer_ok
            # NB: curly apostrophes (’) are normalised here so card verdicts match what a reader sees; the task's
            # own regex only matches straight quotes (see report).
            abstains = lambda t: _abst((t or "").replace("’", "'"))  # noqa: E731
            return dict(
                fail_group=lambda x: not x.get("real", True),
                is_fail=lambda x, t: not abstains(t),
                show_kept=lambda x, t, b: not abstains(t),
                kept_group=lambda x: x.get("real") and x.get("known"),
                is_kept=lambda x, t, b: answer_ok(x, t),
                prompt=lambda x: x.get("prompt", ""),
                label=lambda x: ("made-up " + x.get("kind", "")) if not x.get("real") else
                ("real " + x.get("kind", "") + (" (model knows it)" if x.get("known") else "")),
                prefer=lambda x: (-(x.get("pop") or 0)),
                group_of=lambda x: x.get("kind"),
                # a "clean" fix: declines without pivoting into an invented bio afterwards
                clean=lambda t: not re.search(r"however|that said|here'?s what|here is what|\*\*|\n\s*[*-] ",
                                              (t or "").replace("’", "'"), re.I),
                tail=False)
        if base_task == "agent_injection":
            from tasks.agent_injection import hijacked, tool_calls
            return dict(
                fail_group=lambda x: x.get("kind") == "inj",
                is_fail=lambda x, t: hijacked(x, t),
                kept_group=lambda x: x.get("kind") == "legit",
                is_kept=lambda x, t, b: len(tool_calls(t)) > 0,
                prompt=lambda x: x.get("snippet") or x.get("prompt", "")[-400:],
                label=lambda x: {"inj": "injected page", "legit": "genuine tool request", "clean": "clean page",
                                 "benign": "page mentioning a command"}.get(x.get("kind"), x.get("kind", "")),
                prefer=lambda x: (0 if x.get("disguised") else 1),
                tail=False)
        if base_task == "signoff":
            from tasks._signoff_detect import signoff
            try:
                from tasks.signoff import legit_kept
            except Exception:
                legit_kept = None
            return dict(
                fail_group=lambda x: not x.get("legit"),
                is_fail=lambda x, t: signoff(t)["has"],
                kept_group=lambda x: bool(x.get("legit")) and legit_kept is not None,
                is_kept=lambda x, t, b: legit_kept(x, t) if legit_kept else True,
                prompt=lambda x: x.get("prompt", ""),
                label=lambda x: "request that should keep its question/offer" if x.get("legit") else "ordinary request",
                prefer=lambda x: len(x.get("prompt", "")),
                tail=True)
        if base_task.startswith("whisper"):
            return dict(
                fail_group=lambda x: x.get("suite") == "nonspeech",
                is_fail=lambda x, t: bool(_norm_ws(t)),
                kept_group=lambda x: x.get("suite") not in (None, "nonspeech"),
                is_kept=lambda x, t, b: _norm_ws(t) == _norm_ws(b),
                prompt=lambda x: x.get("id", ""),
                label=lambda x: ("non-speech clip" if x.get("suite") == "nonspeech" else "speech clip") +
                (f" · {x.get('category') or _clip_kind(x.get('id'))}" if (x.get("category") or _clip_kind(x.get("id"))) else ""),
                group_of=lambda x: _clip_kind(x.get("id")) if x.get("suite") == "nonspeech" else x.get("suite"),
                prefer=lambda x: 0 if (SITE / "audio" / f"{x.get('id')}.wav").exists() or
                (SITE / "audio" / f"{x.get('id')}.mp3").exists() else 1,
                tail=False)
    except Exception as e:  # task helpers unavailable -> no automatic examples
        print(f"  [examples] scorer for {base_task} unavailable: {e}")
    return None


def clip(t, n=700, tail=False):
    t = (t or "").strip()
    if len(t) <= n:
        return t
    return "…" + t[-n:].lstrip() if tail else t[:n].rstrip() + "…"


def build_examples(task, d, arms_present):
    base_task = task.split("__")[0]
    sc = scorer(base_task)
    outs = {a: {r["id"]: r.get("text", "") for r in rjsonl(d / "outputs" / f"{a}.jsonl")}
            for a in arms_present if (d / "outputs" / f"{a}.jsonl").exists()}
    if not sc or "base" not in outs or "gate" not in outs:
        return []
    items = {x["id"]: x for x in rjsonl(d / "items.jsonl") if x.get("split", "heldout") == "heldout"}
    if not items:  # whisper: items come from the outputs themselves
        for r in rjsonl(d / "outputs" / "base.jsonl"):
            items[r["id"]] = dict(id=r["id"], suite=r.get("suite"), category=r.get("category"))
    excl = set(EXCLUDE.get(base_task, []))
    pins = PIN.get(base_task, {}) if task == base_task else {}
    prompt_arm = PROMPT_ARM.get(base_task, "prompt")
    if prompt_arm not in outs:
        prompt_arm = next((a for a in ("prompt", "thresholds_tuned", "thresholds", "vad") if a in outs), None)
    show = [a for a in ["base", prompt_arm, "gate", "lora"] if a and a in outs]
    ids = [i for i in items if i not in excl and all(i in outs[a] for a in ("base", "gate"))]
    B, G = outs["base"], outs["gate"]
    P, Lo = outs.get(prompt_arm, {}), outs.get("lora", {})

    def F(i, o):
        return sc["is_fail"](items[i], o[i]) if i in o else None

    def K(i, o):
        return sc["is_kept"](items[i], o[i], B.get(i, "")) if i in o else None

    fg = [i for i in ids if sc["fail_group"](items[i])]
    kg = [i for i in ids if sc["kept_group"](items[i])]
    pref = lambda i: (sc["prefer"](items[i]), len(B.get(i, "")) > 2000, str(i))  # noqa: E731
    clean = sc.get("clean") or (lambda t: True)
    fixed = sorted([i for i in fg if F(i, B) and not F(i, G)], key=lambda i: (not clean(G[i]), pref(i)))
    # prefer clean gate replies for the headline cards: short-ish and starting with the target
    miss = sorted([i for i in fg if F(i, B) and F(i, G)], key=pref)
    kb = sorted([i for i in kg if K(i, B) and K(i, G) and P and K(i, P) is False], key=pref)
    lvg = sorted([i for i in fg if F(i, B) and Lo and (F(i, G) != F(i, Lo))], key=pref)
    lb = sorted([i for i in kg if Lo and i in Lo and K(i, B) and K(i, G)
                 and B.get(i, "").strip() == G.get(i, "").strip() and Lo[i].strip() != B.get(i, "").strip()],
                key=pref) if base_task in LORA_BREAKS_TASKS else []
    picks = []
    group_of = sc.get("group_of") or (lambda x: None)

    def take(kind, pool, k):
        pinned = [i for i in pins.get(kind, []) if i in items]
        chosen = list(pinned)
        k = max(k, len(pinned))
        groups = {group_of(items[i]) for i in chosen}
        for div in (True, False):  # first pass: prefer a different group (e.g. clip category) per card
            for i in pool:
                if len(chosen) >= k:
                    break
                if i in chosen or i in [p[1] for p in picks]:
                    continue
                g = group_of(items[i])
                if div and g is not None and g in groups:
                    continue
                chosen.append(i)
                groups.add(g)
        picks.extend((kind, i) for i in chosen)

    take("fixed", fixed, N_FIXED)
    take("kept_break", kb, 1)
    take("miss", miss, 1)
    take("never_fires", [], 0)
    take("imperfect", [], 0)
    take("lora_breaks", lb, 1 if len(picks) < 5 else 0)
    if len(picks) < 4:
        take("lora_vs_gate", lvg, 1)
    cards = []
    for kind, i in picks:
        x = items[i]
        replies, verdict = {}, {}
        for a in show:
            if i not in outs[a]:
                continue
            replies[a] = clip(outs[a][i], tail=sc["tail"])
            if sc["fail_group"](x):
                verdict[a] = "bad" if sc["is_fail"](x, outs[a][i]) else "good"
            elif sc["kept_group"](x):
                f = sc.get("show_kept") or sc["is_kept"]
                verdict[a] = "good" if f(x, outs[a][i], B.get(i, "")) else "bad"
        audio = None
        for ext in ("mp3", "wav"):
            if (SITE / "audio" / f"{i}.{ext}").exists():
                audio = f"audio/{i}.{ext}"
                break
        cards.append(dict(id=i, kind=kind, label=sc["label"](x), prompt=clip(sc["prompt"](x), 500),
                          name=x.get("name"), replies=replies, verdict=verdict, audio=audio,
                          tail=sc["tail"]))
    return cards


# ------------------------------------------------------------------------------------------------ blocks
def load_arms(d):
    arms = {}
    for f in sorted(d.glob("*.json")):
        if f.name in ("signal.json", "persistence.json", "anatomy.json") or f.stem.startswith("_"):
            continue
        r = rd(f)
        if isinstance(r, dict) and "arm" in r and r.get("split", "heldout") == "heldout" and r.get("headline"):
            if "agent_s" in f.stem:
                continue  # blind-agent-session arms: separate optional block
            arms[r["arm"]] = r
    return arms


def sweep_block(d, h):
    sd = d / "sweep"
    if not sd.exists():
        return None
    by = {}
    for f in sorted(sd.glob("*.json")):
        r = rd(f)
        if not r or "headline" not in r:
            continue
        meth = f.stem.split("_")[0]
        n = r.get("n_practice")
        if n is None:
            continue
        by.setdefault(meth, {}).setdefault(n, []).append(r)
    if not by:
        return None
    out = {}
    for meth, byn in by.items():
        rows = []
        for n in sorted(byn):
            rs = byn[n]

            def agg(vals):
                vals = [v for v in vals if v is not None]
                if not vals:
                    return None, None, None
                return r3(statistics.mean(vals), 4), r3(min(vals), 4), r3(max(vals), 4)
            fm, fmin, fmax = agg([r["headline"].get(h[0]) for r in rs])
            km, kmin, kmax = agg([r["headline"].get(h[1]) for r in rs])
            lm, lmin, lmax = agg([(r.get("guards") or {}).get("chat_kl") for r in rs])
            sm, smin, smax = agg([(r.get("guards") or {}).get("chat_same") for r in rs])
            rows.append(dict(n=n, seeds=len(rs), fail=fm, fail_min=fmin, fail_max=fmax, kept=km, kept_min=kmin,
                             kept_max=kmax, kl=lm, kl_min=lmin, kl_max=lmax, same=sm, same_min=smin, same_max=smax))
        out[meth] = rows
    return out


def persistence_block(d, h):
    """persistence.json (+ persistence_early.json: denser early steps, gate_fire_on_fakes; + persistence_lora.json)."""
    files = [rd(d / f) for f in ("persistence.json", "persistence_early.json", "persistence_lora.json")]
    p = next((f for f in files if f and f.get("conditions")), None)
    if not p:
        return None
    conds = {}
    for f in files:
        if not f or not f.get("conditions"):
            continue
        for k, rows in f["conditions"].items():
            if not isinstance(rows, list):
                continue
            by = {r["step"]: r for r in conds.get(k, [])}
            for r in rows:
                row = dict(step=r.get("step"), fail=r.get(h[0]), kept=r.get(h[1]), loss=r.get("loss"),
                           gate_cos=r.get("gate_row_cos"), down_cos=r.get("down_col_cos"),
                           fire=r.get("gate_fire_on_fakes"))
                old = by.get(row["step"])
                if old is None:
                    by[row["step"]] = row
                else:  # the main run wins where steps overlap; the early run only adds fields it lacks
                    for k_, v_ in row.items():
                        if old.get(k_) is None and v_ is not None:
                            old[k_] = v_
            conds[k] = [by[st] for st in sorted(by)]
    if not conds:
        return None
    gate_cos = [r["gate_cos"] for k, rows in conds.items() if k.startswith("gate") for r in rows
                if r.get("gate_cos") is not None]
    return dict(conditions=conds, steps=p.get("steps"), lr=p.get("lr"), k_layers=p.get("k_layers"),
                n_eval=p.get("n_eval"), gate_cos_min=min(gate_cos) if gate_cos else None)


def _row(r, h):
    g = r.get("guards") or {}
    return dict(fail=r["headline"].get(h[0]), kept=r["headline"].get(h[1]), kl=g.get("chat_kl"),
                same=g.get("chat_same"), same32=g.get("chat_same32"))


def bigdata_block(d, h):
    bd = d / "bigdata"
    if not bd.exists():
        return None
    out = []
    for f in sorted(bd.glob("*.json")):
        r = rd(f)
        if not r or "headline" not in r:
            continue
        meth, tag = f.stem.split("_", 1)
        out.append(dict(method=meth, tag=tag, varied=tag.startswith("xv"), n=r.get("n_practice"),
                        n_extra=r.get("n_extra"), **_row(r, h)))
    return sorted(out, key=lambda x: (x["method"], x["varied"], x["n"] or 0)) or None


def pareto_block(d, h):
    pd = d / "pareto"
    if not pd.exists():
        return None
    out = {}
    for f in sorted(pd.glob("*.json")):
        r = rd(f)
        if not r or "headline" not in r:
            continue
        meth = f.stem.split("_")[0]
        out.setdefault(meth, []).append(dict(knob=r.get("knob"), value=r.get("value"), **_row(r, h)))
    for k in out:
        out[k].sort(key=lambda x: x["value"] or 0)
    return out or None


def agent_session_block(task, d, h):
    names = AGENT_SESSION.get(task)
    if not names:
        return None
    g = rd(d / f"{names[0]}.json")
    if not g or "headline" not in g:
        return None
    rnd = rd(d / f"{names[1]}.json")
    c, fm = g.get("cost") or {}, g.get("fit_meta") or {}
    mins = fm.get("session_minutes") or c.get("session_minutes")
    if mins is None:
        for k in ("session_seconds", "fit_seconds"):
            v = fm.get(k) or c.get(k)
            if v:
                mins = round(v / 60, 1)
                break
    return dict(gate=_row(g, h), random=_row(rnd, h) if rnd and "headline" in rnd else None,
                entries=c.get("entries_changed") or fm.get("entries_changed"), minutes=mins,
                model=fm.get("agent_model") or fm.get("model"))


def anatomy_block(d):
    a = rd(d / "anatomy.json")
    if not a or not a.get("fail"):
        return None
    groups = {k: a.get(k) or [] for k in ("fail", "good", "content")}
    allv = [v for g in groups.values() for v in g]
    lo, hi = math.floor(min(allv)), math.ceil(max(allv))
    nb = 48
    w = (hi - lo) / nb or 1
    edges = [round(lo + i * w, 4) for i in range(nb + 1)]
    hist = {}
    for k, g in groups.items():
        c = [0] * nb
        for v in g:
            c[min(nb - 1, max(0, int((v - lo) / w)))] += 1
        hist[k] = dict(n=len(g), counts=c, fire=round(sum(v > 0 for v in g) / max(1, len(g)), 4),
                       fired=sum(v > 0 for v in g))
    return dict(edges=edges, hist=hist, layer=a.get("layer"), unit=a.get("unit"), theta=a.get("theta"),
                write_up=[t for t in (a.get("write_up") or [])[:8]],
                write_down=[t for t in (a.get("write_down") or [])[:6]])


def notes_block(d):
    p = d / "NOTES.md"
    if not p.exists():
        return None
    t = p.read_text(encoding="utf-8")
    return t[:4000]


def _unchanged(base_task, r):
    u = UNCHANGED.get(base_task)
    if not u:
        return (r.get("guards") or {}).get("chat_same")
    src = r.get(u[0]) or {}
    return src.get(u[1])


def task_block(task, d):
    base_task = task.split("__")[0]
    meta = dict(TASK_META.get(base_task, dict(title=task, short=task)))
    arms = load_arms(d)
    blk = dict(id=task, base_task=base_task, variant=task != base_task, meta=meta, status="pending",
               group=GROUP.get(task, "transfer" if "__" in task else "other"))
    if task != base_task:
        blk["meta"]["short"] = meta.get("short", base_task) + " · " + task.split("__")[1]
    if not arms:
        return blk
    any_r = next(iter(arms.values()))
    h = list(any_r["headline"].keys())
    order = sorted(arms, key=arm_key)
    blk.update(model=any_r.get("model"), n=any_r.get("n"), headline_keys=h, arm_order=order,
               status="ready" if {"base", "gate"} <= set(arms) else "partial")
    if task != base_task:
        blk["meta"]["model_label"] = (any_r.get("model") or "").split("/")[-1]
    arm_rows = {}
    for a in order:
        r = arms[a]
        g, c = r.get("guards") or {}, r.get("cost") or {}
        arm_rows[a] = dict(label=ARM_LABEL.get(a, a.replace("_", " ")), fail=r["headline"].get(h[0]),
                           kept=r["headline"].get(h[1]), mmlu=g.get("mmlu200"), kl=g.get("chat_kl"),
                           kl_note=g.get("chat_kl_note"), entries=c.get("entries_changed"),
                           trainable=c.get("trainable_params"), fit_s=c.get("fit_seconds"),
                           n_practice=c.get("n_practice_items"), same=_unchanged(base_task, r),
                           same32=g.get("chat_same32"), chat_same=g.get("chat_same"),
                           metrics={k: v for k, v in (r.get("metrics") or {}).items()
                                    if isinstance(v, (int, float))})
    blk["arms"] = arm_rows
    blk["score_arms"] = [a for a in SCORE_ARMS.get(base_task, ARM_ORDER) if a in arm_rows] or order
    u = UNCHANGED.get(base_task)
    blk["meta"]["same_label"] = u[2] if u else None
    gm = (arms.get("gate") or {}).get("fit_meta") or {}
    blk["gate_meta"] = {k: gm.get(k) for k in ("layer", "unit", "theta", "targets", "recall_practice",
                                               "neg_fire_practice", "content_fire_practice") if k in gm}
    sig = rd(d / "signal.json")
    if sig and sig.get("cv_auc"):
        best = max(sig["cv_auc"].items(), key=lambda kv: kv[1])
        blk["signal"] = dict(best_layer=best[0], best_auc=best[1], cv_auc=sig["cv_auc"])
    blk["sweep"] = sweep_block(d, h)
    if not blk["sweep"] and task in SWEEP_FROM and (RES / SWEEP_FROM[task]).exists():
        blk["sweep"] = sweep_block(RES / SWEEP_FROM[task], h)
        if blk["sweep"]:
            blk["sweep_source"] = SWEEP_FROM[task]
    # the "all" sweep point reports every practice item; use the number of *labelled* items if that's smaller
    gm_ = (arms.get("gate") or {}).get("fit_meta") or {}
    if blk["sweep"] and gm_.get("n_pos") and gm_.get("n_neg") and gm_.get("n_items") and \
            gm_["n_pos"] + gm_["n_neg"] < gm_["n_items"]:
        for rows in blk["sweep"].values():
            for r in rows:
                if r["n"] == gm_["n_items"]:
                    r["n"], r["n_all_items"] = gm_["n_pos"] + gm_["n_neg"], gm_["n_items"]
    blk["bigdata"] = bigdata_block(d, h)
    blk["pareto"] = pareto_block(d, h)
    blk["agent_session"] = agent_session_block(task, d, h)
    blk["persistence"] = persistence_block(d, h)
    blk["anatomy"] = anatomy_block(d)
    blk["examples"] = build_examples(task, d, order) if blk["group"] == "main" else []
    blk["notes"] = notes_block(d)
    # derived numbers for prose
    A = arm_rows
    der = {}
    if "gate" in A and "lora" in A and A["gate"].get("kl") and A["lora"].get("kl"):
        der["kl_ratio_lora_gate"] = round(A["lora"]["kl"] / A["gate"]["kl"], 1)
    if "gate" in A and A["gate"].get("entries") and meta.get("params"):
        der["entries_frac"] = A["gate"]["entries"] / meta["params"]
    if "gate" in A and "lora" in A and isNum(A["gate"].get("same")) and isNum(A["lora"].get("same")):
        der["same_gate"], der["same_lora"] = A["gate"]["same"], A["lora"]["same"]
    an = blk.get("anatomy")
    if an:
        der["fired_fail"] = an["hist"]["fail"]["fired"]
        der["n_fail"] = an["hist"]["fail"]["n"]
    elif "gate" in A and "base" in A:
        pass
    blk["derived"] = der
    return blk


def archive_block():
    ar = SITE / "data" / "archive"
    out = {}
    g = rd(ar / "grokking.json")
    if g:
        series = {}
        for k, v in g["curves"]["series"].items():
            # keep at most ~200 points per series
            step = max(1, len(v) // 200)
            series[k] = [[p[0], round(p[1], 2)] for p in v[::step]] + ([v[-1]] if (len(v) - 1) % step else [])
        out["grokking"] = dict(series=series, labels=g["curves"].get("labels"), description=g.get("description"),
                               edit_step=g.get("edit_step_absolute"))
    t = rd(ar / "track_a_scoreboard.json")
    if t:
        out["track_a"] = t.get("rows")
    return out


def main():
    global RES
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(RES), help="results dir (default: <repo>/results)")
    ap.add_argument("--out", default=str(SITE / "data" / "blog.json"))
    args = ap.parse_args()
    RES = Path(args.results)
    tasks = {}
    if RES.exists():
        dirs = sorted(p for p in RES.iterdir() if p.is_dir() and not p.name.startswith(("_", ".")))
    else:
        dirs = []
    names = [p.name for p in dirs]
    for t in MAIN_TASKS:
        if t not in names:
            tasks[t] = dict(id=t, base_task=t, variant=False, meta=TASK_META[t], status="pending")
    for d in dirs:
        if d.name in SWEEP_FROM.values() or d.name in EXCLUDE_TASKS:
            continue
        print(f"task {d.name}")
        try:
            blk = task_block(d.name, d)
            if blk["status"] == "pending" and d.name not in MAIN_TASKS:
                continue  # helper / scratch dirs without arm results: don't list them
            tasks[d.name] = blk
        except Exception as e:  # never let one task break the page
            print(f"  !! {d.name}: {e!r}")
            tasks[d.name] = dict(id=d.name, base_task=d.name.split("__")[0], variant="__" in d.name,
                                 meta=TASK_META.get(d.name.split("__")[0], dict(title=d.name, short=d.name)),
                                 status="pending", error=str(e))
    order = [t for t in MAIN_TASKS if t in tasks] + sorted(t for t in tasks if t not in MAIN_TASKS)
    out = dict(generated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), task_order=order,
               tasks={t: tasks[t] for t in order}, archive=archive_block(), site=SITE_META)
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    for t in order:
        b = tasks[t]
        print(f"  {t:40s} {b['status']:8s} arms={','.join(b.get('arm_order', []))} "
              f"sweep={'y' if b.get('sweep') else '-'} persist={'y' if b.get('persistence') else '-'} "
              f"anatomy={'y' if b.get('anatomy') else '-'} examples={len(b.get('examples') or [])}")
    print(f"wrote {p} ({p.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()

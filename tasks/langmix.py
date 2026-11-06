"""Language mixing: DeepSeek-R1-Distill-Qwen-1.5B leaks Chinese words into German, Russian and Thai answers
("Kalorien还不够", "5美元").

Port of the original langmix experiment (langmix/, RESULTS_2026-11-05_practical_edits.md "Experiment 2") to patchkit.
See results/langmix/NOTES.md for the design and every deviation from the old run.

Data
- held-out: MGSM test (juletxara/mgsm), the same 120 problems per language as the old run (seed-0 shuffle, first 120),
  in de, ru, th (target: must not leak), en (must stay English) and zh (must still be answered in Chinese).
- practice: GSM8K *train* problems, machine-translated de/ru/th (Mathoctopus/GSM8KInstruct_Parallel), English GSM8K
  train, Chinese GSM8K_zh train. Disjoint from MGSM (GSM8K test) and translated by a different pipeline.
- prompt: the question + "Please reason step by step, and put your final answer within \\boxed{}." (Chinese
  counterpart for zh), no system prompt (as the old run). Greedy, `max_new` tokens; scored on the generated text.

Labels (positive): a de/ru/th practice item whose greedy base reply leaks (mix_flag) = True; a de/ru/th item that
doesn't leak, and every zh / en practice item = False. `data` generates the practice base replies itself (written to
runs/langmix/base_practice.jsonl in cmd_base's format, same order and batching) because the labels, the gate's
target tokens and the LoRA targets all depend on them.

Gate (generic patchkit fit, no task-level fitting code): a "leak-onset" gate.
- positives ("decision", label 1): reply positions of de/ru/th practice base replies whose next token is a CJK token
  and whose own token is not (the onset of each leaked word);
- content negatives: every other reply position before the first leak, every position of non-leaking de/ru/th
  replies, and every position of zh and en replies (incl. the zh positions that predict Chinese), every
  `content_stride`-th position (memory);
- decision negatives (for the `signal` AUC only; they're also content): 4 sampled positions per item.
The fit then lowers the 8 most frequent leaked tokens (the positives' next tokens) and raises `gate_targets`: the
most frequent non-CJK argmax tokens at the practice leak onsets (what the model says instead when CJK is masked),
computed on practice in `data` and stored in runs/langmix/gate_targets.json.

LoRA targets: positives -> the practice reply generated with every CJK token banned (logits processor), the cheap
practitioner way to get a leak-free reply; negatives -> the base reply (stored in runs/langmix/ban_practice.jsonl).

    failure = leak_rate: % of de/ru/th held-out replies with >= 1 Han character (pooled, 360)
    kept    = zh_reply: % of Chinese held-out prompts whose final answer (after </think>) is in Chinese
"""
import json
import random
import zlib
from collections import Counter
from pathlib import Path

import torch

from patchkit import llm as L
from patchkit.task import Task, pct, register

from tasks import _langmix_text as X

ROOT = Path(__file__).resolve().parent.parent

# ---- compat shim (proposed core change): patchkit.llm.eot_id only knows <end_of_turn>/<|im_end|>/<|eot_id|>; the R1
# distill tokenizer ends turns with its eos token (<｜end▁of▁sentence｜>) and has none of those, so eot_id raises
# StopIteration (train_lora calls it). Fall back to eos. No effect on tokenizers that have one of the three.
if not getattr(L.eot_id, "_langmix_shim", False):
    _eot_orig = L.eot_id

    def _eot_id(tok):
        try:
            return _eot_orig(tok)
        except StopIteration:
            return tok.eos_token_id
    _eot_id._langmix_shim = True
    L.eot_id = _eot_id

SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
ZH_SUFFIX = "\n请逐步推理，并将最终答案放在 \\boxed{} 中。"
INSTRUCTION = "Reply in the same language as the question."   # the old run's instruction arm, appended to the user turn
TARGET_LANGS = ("de", "ru", "th")
POOL_NAMES = dict(de="German", ru="Russian", th="Thai")
N_HELDOUT = 120
N_PRACTICE = dict(de=96, ru=96, th=96, zh=64, en=64)


def _gsm_gold(ans):
    return ans.split("####")[-1].strip().replace(",", "")


def _mgsm(lang):
    from datasets import load_dataset
    ds = list(load_dataset("juletxara/mgsm", lang, split="test"))
    order = list(range(len(ds)))
    random.Random(0).shuffle(order)  # identical to the old run (check_multi.py / data.py)
    return [dict(id=f"mgsm-{lang}-{i}", question=ds[i]["question"], gold=str(ds[i]["answer_number"]), lang=lang,
                 src="mgsm", problem=i) for i in order[:N_HELDOUT]]


def _pools():
    """Practice pools: GSM8K-train problems in de/ru/th (GSM8KInstruct_Parallel), en (GSM8K), zh (GSM8K_zh)."""
    import re
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    gold = [_gsm_gold(r["answer"]) for r in load_dataset("openai/gsm8k", "main", split="train")]
    en_q = [r["question"] for r in load_dataset("openai/gsm8k", "main", split="train")]
    path = hf_hub_download("Mathoctopus/GSM8KInstruct_Parallel", "MGSM8KInstruct_Parallel.json", repo_type="dataset")
    pools = {lg: [] for lg in TARGET_LANGS}
    want = {v: k for k, v in POOL_NAMES.items()}
    idx = -1
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        m = re.search(r"request in (\w+)\.", r["prompt"])
        if m.group(1) == "English":
            idx += 1
        if m.group(1) not in want:
            continue
        q = r["prompt"].split("### Instruction:", 1)[1].split("### Response:", 1)[0].strip()
        pools[want[m.group(1)]].append(dict(question=q, gold=gold[idx], problem=idx))
    pools["en"] = [dict(question=q, gold=g, problem=i) for i, (q, g) in enumerate(zip(en_q, gold))]
    zh = load_dataset("meta-math/GSM8K_zh", split="train")
    pools["zh"] = [dict(question=r["question_zh"], gold=str(r["answer_only"]).replace(",", ""), problem=i)
                   for i, r in enumerate(zh) if r["split"] == "train"]
    return pools


@register
class LangMix(Task):
    name = "langmix"
    model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    description = "Leaks Chinese words into German, Russian and Thai answers"
    system_fix = INSTRUCTION
    max_new = 1024
    gen_batch = 120
    probe_layers = (0.5, 0.8, 1.0)
    gate_layer = -1
    forced_prefix = "<eot>"          # no forced-opener generation; LoRA positives use CJK-banned replies instead
    headline = ("leak_rate", "zh_reply")
    headline_labels = ("de/ru/th answers with a leaked Chinese word", "Chinese prompts still answered in Chinese")
    content_stride = 2               # every 2nd ordinary reply position is a content negative (memory)
    n_gate_targets = 4

    # ------------------------------------------------------------ files
    @property
    def runs(self):
        return ROOT / "runs" / self.name

    def _jsonl(self, fname):
        attr = "_cache_" + fname
        if not hasattr(self, attr):
            p = self.runs / fname
            setattr(self, attr, {r["id"]: r for r in L.read_jsonl(p)} if p.exists() else {})
        return getattr(self, attr)

    @property
    def gate_targets(self):
        p = self.runs / "gate_targets.json"
        if not p.exists():
            raise FileNotFoundError(f"{p}: run `data langmix` first (it derives the targets from practice)")
        return json.loads(p.read_text(encoding="utf-8"))["targets"][: self.n_gate_targets]

    # ------------------------------------------------------------ data
    def build_items(self, ctx):
        rng = random.Random(0)
        items = []
        for lg in TARGET_LANGS + ("en", "zh"):
            items += [dict(x, split="heldout") for x in _mgsm(lg)]
        pools, used = _pools(), set()
        for lg in TARGET_LANGS + ("zh", "en"):
            pool = list(pools[lg])
            rng.shuffle(pool)
            take = []
            for x in pool:
                # one language per GSM8K-train problem (de/ru/th/en share problem indices)
                if lg != "zh" and x["problem"] in used:
                    continue
                take.append(x)
                if lg != "zh":
                    used.add(x["problem"])
                if len(take) == N_PRACTICE[lg]:
                    break
            items += [dict(id=f"prac-{lg}-{x['problem']}", question=x["question"], gold=x["gold"], lang=lg,
                           src="gsm8k_train", problem=x["problem"], split="practice") for x in take]
        self._practice_generations(ctx, [x for x in items if x["split"] == "practice"])
        return items

    def _practice_generations(self, ctx, pr):
        """Base replies on practice (labels), the gate's target tokens, and CJK-banned replies (LoRA targets)."""
        tok, model = ctx.tok, ctx.model
        f = ctx.runs / "base_practice.jsonl"
        if f.exists():
            got = {r["id"]: r["base"] for r in L.read_jsonl(f)}
        else:
            outs = L.generate(model, tok, [self.render(tok, x) for x in pr], max_new=self.max_new,
                              batch=self.gen_batch, log="base")   # = cmd_base (same order, batching, decoding)
            got = {x["id"]: o for x, o in zip(pr, outs)}
            L.write_jsonl(f, [dict(id=x["id"], base=got[x["id"]]) for x in pr])
        for x in pr:
            x["leaks"] = X.mix_flag(got[x["id"]], x["lang"]) if x["lang"] in TARGET_LANGS else False
        print("practice base:", json.dumps(self.score(pr, [got[x["id"]] for x in pr])), flush=True)
        pos = [x for x in pr if x["leaks"]]
        # gate targets: at the first leak onset of each leaking practice reply, the base model's argmax over non-CJK
        # tokens (what it says when Chinese is masked out)
        mask = X.cjk_mask(tok, model.config.vocab_size)
        alts = Counter()
        with torch.no_grad():
            for x in pos:
                p = self.render(tok, x)
                ids = tok(p + got[x["id"]], add_special_tokens=False).input_ids
                n_p = len(tok(p, add_special_tokens=False).input_ids)
                on = next((j for j in range(n_p - 1, len(ids) - 1) if mask[ids[j + 1]]), None)
                if on is None:
                    continue
                lg = model(input_ids=torch.tensor([ids[:on + 1]], device=L.DEV)).logits[0, -1].float()
                lg[mask.to(lg.device)[: lg.shape[0]]] = -float("inf")
                alts[int(lg.argmax())] += 1
        tg = []
        for t, c in alts.most_common():
            s = tok.decode([t])
            if tok.encode(s, add_special_tokens=False)[:1] == [t]:   # round-trips through patchkit's target_ids
                tg.append(dict(token=s, count=c))
        (ctx.runs / "gate_targets.json").write_text(json.dumps(
            dict(targets=[x["token"] for x in tg], counts=tg[:30], n_onsets=sum(alts.values())),
            ensure_ascii=False, indent=1), encoding="utf-8")
        print("gate targets:", [(x["token"], x["count"]) for x in tg[:12]], flush=True)
        # LoRA targets for positives: the same prompt with every CJK token banned
        fb = ctx.runs / "ban_practice.jsonl"
        if not fb.exists():
            outs = X.generate(model, tok, [self.render(tok, x) for x in pos], self.max_new, self.gen_batch,
                              processors=[X.BanProcessor(mask)], log="ban")
            L.write_jsonl(fb, [dict(id=x["id"], ban=o) for x, o in zip(pos, outs)])
            print("ban replies: leak", pct(range(len(pos)), lambda i: X.mix_flag(outs[i], pos[i]["lang"])),
                  "acc", pct(range(len(pos)), lambda i: X.correct(outs[i], pos[i]["gold"])), flush=True)

    def user_text(self, x):
        return x["question"] + (ZH_SUFFIX if x["lang"] == "zh" else SUFFIX)

    def messages(self, item):
        return self.user_text(item), None

    def render(self, tok, item, fix=False):
        # the prompt arm appends the instruction to the user turn (DeepSeek advises against system prompts for R1
        # models; the old run's instruction arm did the same)
        u = self.user_text(item) + ("\n" + INSTRUCTION if fix else "")
        return L.chat(tok, u)

    def positive(self, x):
        if x["lang"] in ("zh", "en"):
            return False
        return x.get("leaks")

    # ------------------------------------------------------------ gate sites
    def probe_sites(self, tok, item, base_text):
        lab = self.positive(item)
        if lab is None:
            return []
        p = self.render(tok, item)
        n_p = len(tok(p, add_special_tokens=False).input_ids)
        full = p + base_text
        ids = tok(full, add_special_tokens=False).input_ids
        if not hasattr(self, "_mask"):
            self._mask = X.cjk_mask(tok, len(tok))
        mask = self._mask
        rng = random.Random(zlib.crc32(item["id"].encode()))
        reply = list(range(n_p - 1, len(ids) - 1))          # position j predicts ids[j + 1]
        if item["lang"] in TARGET_LANGS:
            ons = [j for j in reply if mask[ids[j + 1]] and (j < n_p or not mask[ids[j]])]
            first = ons[0] if ons else len(ids) - 1
            clean = [j for j in reply if j < first]
            sites = []
            if ons:
                sites.append(dict(seq=full, pos=ons, label=1, kind="decision"))
        else:
            clean, sites = reply, []
        if item["lang"] == "zh":
            dneg = [j for j in clean if mask[ids[j + 1]]]
        else:
            dneg = clean
        dneg = rng.sample(dneg, min(4, len(dneg)))
        if dneg:
            sites.append(dict(seq=full, pos=dneg, label=0, kind="decision"))
        con = clean[::self.content_stride]
        if con:
            sites.append(dict(seq=full, pos=con, label=0, kind="content"))
        return sites

    # ------------------------------------------------------------ LoRA
    def sft_target(self, item, base_text, forced_text):
        lab = self.positive(item)
        if lab is None:
            return None
        if lab:
            r = self._jsonl("ban_practice.jsonl").get(item["id"])
            return r["ban"] if r else None
        return base_text

    # ------------------------------------------------------------ scoring
    def score(self, items, texts):
        by = {}
        for i, x in enumerate(items):
            by.setdefault(x["lang"], []).append(i)
        tgt = [i for lg in TARGET_LANGS for i in by.get(lg, [])]
        out = dict(leak_rate=pct(tgt, lambda i: X.mix_flag(texts[i], items[i]["lang"])))
        for lg in TARGET_LANGS:
            if lg in by:
                out[f"leak_{lg}"] = pct(by[lg], lambda i: X.mix_flag(texts[i], lg))
        for lg in TARGET_LANGS + ("en", "zh"):
            if lg in by:
                out[f"acc_{lg}"] = pct(by[lg], lambda i: X.correct(texts[i], items[i]["gold"]))
        deru = by.get("de", []) + by.get("ru", [])
        if deru:
            out["acc_deru"] = pct(deru, lambda i: X.correct(texts[i], items[i]["gold"]))
        if "zh" in by:
            out["zh_reply"] = pct(by["zh"], lambda i: X.zh_majority(X.split_think(texts[i])[1]))
            out["zh_answered"] = pct(by["zh"], lambda i: X.split_think(texts[i])[1] is not None)
            out["zh_reply_of_answered"] = pct(
                [i for i in by["zh"] if X.split_think(texts[i])[1] is not None],
                lambda i: X.zh_majority(X.split_think(texts[i])[1]))
        if "en" in by:
            out["en_leak"] = pct(by["en"], lambda i: X.mix_flag(texts[i], "en"))
        if tgt:
            out["answered_deruth"] = pct(tgt, lambda i: X.split_think(texts[i])[1] is not None)
        out["n"] = {lg: len(v) for lg, v in by.items()}
        return out

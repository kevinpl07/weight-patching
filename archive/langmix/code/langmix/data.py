"""Fixed suites for the language-mixing experiment, cached as JSON in runs/langmix/data/<suite>.json.

    .venv/Scripts/python langmix/data.py            # build (or verify) every suite; prints sizes

Framing (2026-11-04, user decision): Chinese leaking into German, Russian and Thai answers of R1-Distill-Qwen-1.5B.
Prompt = the native-language question only, no instruction (kind "mgsm"), as in langmix/check_multi.py.

Held-out (the agent never sees these). MGSM test (juletxara/mgsm) is parallel across languages; every MGSM suite takes
the same 120 problems: a seed-0 shuffle of the 250 indices (check_multi.py's order), first 120. These INCLUDE the
first 16 per language that check_multi.py ran; the base arm reuses those 48 rows (same prompt and settings).
- mgsm_de, mgsm_ru, mgsm_th  (target)      120 each: Han-character leaks, accuracy, thinking in the prompt language
- mgsm_en                    (retention)   120: English must stay English and keep its accuracy
- mgsm_zh                    (retention)   120: accuracy and "replies in Chinese when asked in Chinese"
- ifeval                     (collateral)  google/IFEval, a fixed random subset of 100 of the 541 prompts (seed 0)
- chat                       (collateral, KL) HuggingFaceH4/no_robots *test* split: first user turn of conversations
                             with no system message, English (>= 95% ASCII letters), 20-1,500 characters; 150 (seed 0)
Older suites (first framing, English MATH; still buildable, not in the default eval): math500, gsm_sym.

Practice pools (the agent sees only what mine_practice.py selects from these):
- pool_de, pool_ru, pool_th, pool_fr, pool_es: Mathoctopus/GSM8KInstruct_Parallel ("MGSM8KInstruct": GSM8K *train*
  machine-translated into 10 languages; ungated). Only the question is used; the gold number comes from the English
  GSM8K-train answer of the same problem. Disjoint from MGSM, which is built from GSM8K *test*.
- gsm8k_train (en): openai/gsm8k "main" train. gsm8k_zh_train (zh): meta-math/GSM8K_zh rows with split == "train".
All pools are shuffled with seed 0; ids carry the GSM8K-train index, so the same problem has the same index in every
language.
"""
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA  # noqa: E402

N_MATH, N_SYM, N_IFEVAL, N_CHAT = 300, 200, 100, 150
N_MGSM = 120
MGSM_LANGS = ("de", "ru", "th", "en", "zh")
POOL_LANGS = dict(de="German", ru="Russian", th="Thai", fr="French", es="Spanish")


def _gsm_gold(ans):
    return ans.split("####")[-1].strip().replace(",", "")


def build_math500():
    from datasets import load_dataset
    ds = list(load_dataset("HuggingFaceH4/MATH-500", split="test"))
    order = list(range(len(ds)))
    random.Random(0).shuffle(order)  # identical to check.py
    return [dict(id=ds[i]["unique_id"], question=ds[i]["problem"], gold=ds[i]["answer"], lang="en", kind="math",
                 scorer="math", level=ds[i]["level"], subject=ds[i]["subject"]) for i in order[:N_MATH]]


def build_gsm_sym():
    from datasets import load_dataset
    ds = load_dataset("apple/GSM-Symbolic", "main", split="test")
    rows = [dict(id=f"sym-{r['original_id']}-{r['instance']}", question=r["question"], gold=_gsm_gold(r["answer"]),
                 lang="en", kind="math", scorer="numeric") for r in ds if int(r["instance"]) < 2]
    assert len(rows) == N_SYM, len(rows)
    return rows


def build_mgsm(lang):
    from datasets import load_dataset
    ds = list(load_dataset("juletxara/mgsm", lang, split="test"))
    order = list(range(len(ds)))
    random.Random(0).shuffle(order)  # identical to check_multi.py
    return [dict(id=f"mgsm-{lang}-{i}", question=ds[i]["question"], gold=str(ds[i]["answer_number"]), lang=lang,
                 kind="mgsm", scorer="numeric") for i in order[:N_MGSM]]


def build_pool(lang):
    """GSM8K-train questions in `lang` from Mathoctopus/GSM8KInstruct_Parallel (records come in per-problem groups
    that start with the English version; the problem index advances at every English record)."""
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("Mathoctopus/GSM8KInstruct_Parallel", "MGSM8KInstruct_Parallel.json", repo_type="dataset")
    gold = [_gsm_gold(r["gold"]) for r in build_gsm8k_train_raw()]
    rows, idx, want = [], -1, POOL_LANGS[lang]
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        m = re.search(r"request in (\w+)\.", r["prompt"])
        if m.group(1) == "English":
            idx += 1
        if m.group(1) != want:
            continue
        q = r["prompt"].split("### Instruction:", 1)[1].split("### Response:", 1)[0].strip()
        rows.append(dict(id=f"gsm8k-{lang}-train-{idx}", question=q, gold=gold[idx], lang=lang, kind="mgsm",
                         scorer="numeric"))
    assert idx == len(gold) - 1, (idx, len(gold))
    random.Random(0).shuffle(rows)
    return rows


def build_gsm8k_train_raw():
    from datasets import load_dataset
    return [dict(gold=r["answer"]) for r in load_dataset("openai/gsm8k", "main", split="train")]


def build_ifeval():
    from datasets import load_dataset
    docs = [dict(r) for r in load_dataset("google/IFEval", split="train")]
    idx = random.Random(0).sample(range(len(docs)), N_IFEVAL)
    return [dict(id=f"ifeval-{docs[i]['key']}", question=docs[i]["prompt"], lang="en", kind="ifeval", doc=docs[i])
            for i in idx]


def build_chat():
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/no_robots", split="test")
    pool = []
    for r in ds:
        m = r["messages"]
        if not m or m[0]["role"] != "user":
            continue
        q = m[0]["content"].strip()
        letters = re.findall(r"[^\W\d_]", q)
        if not (20 <= len(q) <= 1500) or not letters or sum(c.isascii() for c in letters) / len(letters) < 0.95:
            continue
        pool.append(dict(id=f"chat-{r['prompt_id'][:12]}", question=q, lang="en", kind="chat", category=r["category"]))
    return random.Random(0).sample(pool, N_CHAT)


def build_gsm8k_train():
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="train")
    rows = [dict(id=f"gsm8k-train-{i}", question=r["question"], gold=_gsm_gold(r["answer"]), lang="en", kind="mgsm",
                 scorer="numeric") for i, r in enumerate(ds)]
    random.Random(0).shuffle(rows)
    return rows


def build_gsm8k_zh_train():
    from datasets import load_dataset
    ds = load_dataset("meta-math/GSM8K_zh", split="train")
    rows = [dict(id=f"gsm8k-zh-train-{i}", question=r["question_zh"], gold=str(r["answer_only"]).replace(",", ""),
                 lang="zh", kind="mgsm", scorer="numeric") for i, r in enumerate(ds) if r["split"] == "train"]
    random.Random(0).shuffle(rows)
    return rows


SUITES = dict(**{f"mgsm_{lg}": (lambda lg=lg: build_mgsm(lg)) for lg in MGSM_LANGS},
              ifeval=build_ifeval, chat=build_chat, math500=build_math500, gsm_sym=build_gsm_sym,
              gsm8k_train=build_gsm8k_train, gsm8k_zh_train=build_gsm8k_zh_train,
              **{f"pool_{lg}": (lambda lg=lg: build_pool(lg)) for lg in POOL_LANGS})
POOLS = dict(en="gsm8k_train", zh="gsm8k_zh_train", **{lg: f"pool_{lg}" for lg in POOL_LANGS})
HELD_OUT = tuple(f"mgsm_{lg}" for lg in MGSM_LANGS) + ("ifeval", "chat")


def load(suite):
    path = DATA / f"{suite}.json"
    if not path.exists():
        DATA.mkdir(parents=True, exist_ok=True)
        rows = SUITES[suite]()
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=0), encoding="utf-8")
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    for s in SUITES:
        rows = load(s)
        print(f"{s:16s} {len(rows):5d}  e.g. {rows[0]['id']}: {rows[0]['question'][:70]!r}")


if __name__ == "__main__":
    main()

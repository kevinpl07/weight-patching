"""Text-side helpers for tasks/langmix.py: CJK detection, mixing flag, Chinese-reply rule, numeric answer scoring, the
CJK token list and a logits-processor ban. Ported (not imported) from the original langmix experiment
(langmix/check.py, common.py, metrics.py); the definitions are unchanged unless noted.
"""
import math
import re

import torch

# ---------------------------------------------------------------- scripts (langmix/check.py, metrics.py)
CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")
HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
HAN_RUN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]{2,}")
LATIN_WORD = re.compile(r"[A-Za-z]{2,}")
SCRIPT_RE = dict(han=HAN, kana=re.compile(r"[぀-ヿ]"), hangul=re.compile(r"[가-힯]"),
                 thai=re.compile(r"[฀-๿]"), cyrillic=re.compile(r"[Ѐ-ӿ]"),
                 bengali=re.compile(r"[ঀ-৿]"), latin=re.compile(r"[A-Za-zÀ-ɏ]"))
LANG_SCRIPT = dict(en="latin", de="latin", fr="latin", es="latin", zh="han", th="thai", ru="cyrillic")
FOREIGN_RUN = {k: re.compile("(?:" + r.pattern + "){3,}") for k, r in SCRIPT_RE.items() if k not in ("latin", "han")}


def split_think(text):
    """(reasoning, final answer). The R1 template opens <think>, so the reply starts inside it. No </think> -> the
    whole text is reasoning and the final answer is None (truncated or never finished)."""
    i = text.rfind("</think>")
    if i < 0:
        return text, None
    return text[:i], text[i + len("</think>"):]


def strip_math(s):
    s = re.sub(r"\\boxed\{[^{}]*(\{[^{}]*\}[^{}]*)*\}", " ", s)
    s = re.sub(r"\$\$.*?\$\$|\$[^$]*\$|\\\(.*?\\\)|\\\[.*?\\\]", " ", s, flags=re.S)
    s = re.sub(r"\\[A-Za-z]+", " ", s)
    s = re.sub(r"[0-9]", " ", s)
    return s


def zh_majority(s, min_han=4):
    """'Written in Chinese': >= 4 Han characters (math stripped) and at least as many Han characters as Latin words."""
    if s is None:
        return False
    s = strip_math(s)
    h, w = len(HAN.findall(s)), len(LATIN_WORD.findall(s))
    return h >= min_han and h >= w


def mix_flag(text, target="en"):
    """The leak flag (headline): for a non-Han target, >= 1 Han character anywhere in reasoning + answer, or a run of
    >= 3 characters of another foreign non-Latin script."""
    own = LANG_SCRIPT.get(target, "latin")
    if own != "han" and HAN.search(text):
        return True
    return any(r.search(text) for k, r in FOREIGN_RUN.items() if k != own)


# ---------------------------------------------------------------- numeric answers
NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?|-?\.\d+")


def last_boxed(text):
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    i += 7
    depth, j = 1, i
    while j < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        j += 1
    return text[i:j - 1] if depth == 0 else None


def to_number(s):
    if s is None:
        return None
    s = s.replace("\\!", "").replace("\\,", "").replace("$", "").replace("\\%", "").replace("%", "")
    s = re.sub(r"\\text\{[^}]*\}", "", s)
    m = re.fullmatch(r"\s*\\[dt]?frac\{(-?[\d.]+)\}\{(-?[\d.]+)\}\s*", s)
    if m:
        try:
            return float(m.group(1)) / float(m.group(2))
        except (ValueError, ZeroDivisionError):
            return None
    nums = NUM.findall(s)
    if not nums:
        return None
    try:
        return float(nums[-1].replace(",", ""))
    except ValueError:
        return None


def numeric_pred(text):
    """Last \\boxed{} anywhere, else the last number of the final answer (after </think>)."""
    b = last_boxed(text)
    if b is not None:
        return to_number(b)
    _, answer = split_think(text)
    return to_number(answer) if answer else None


def correct(text, gold):
    p = numeric_pred(text)
    return p is not None and math.isclose(p, float(gold), rel_tol=0, abs_tol=1e-6)


# ---------------------------------------------------------------- CJK tokens (langmix/common.py)
def _byte_decoder():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


_CJK_CACHE = {}


def cjk_token_ids(tok):
    """Every vocab token that can emit a CJK character: decoded text contains one, or a byte fragment contains a
    UTF-8 lead byte 0xE3-0xED (U+3000-U+D7FF: CJK punctuation, kana, Han, Hangul). Special tokens never."""
    key = tok.name_or_path
    if key in _CJK_CACHE:
        return _CJK_CACHE[key]
    dec = _byte_decoder()
    special = set(tok.all_special_ids) | set(getattr(tok, "added_tokens_decoder", {}) or {})
    ids = []
    for t, i in tok.get_vocab().items():
        if i in special:
            continue
        try:
            raw = bytes(dec[c] for c in t)
        except KeyError:
            continue
        if CJK.search(raw.decode("utf-8", errors="ignore")) or any(0xE3 <= b <= 0xED for b in raw):
            ids.append(i)
    ids.sort()
    _CJK_CACHE[key] = ids
    return ids


def cjk_mask(tok, vocab_size):
    m = torch.zeros(vocab_size, dtype=torch.bool)
    m[torch.tensor(cjk_token_ids(tok))] = True
    return m


class BanProcessor:
    """Logits processor: CJK token logits -> -inf (the cheap practitioner fix, used here only to write leak-free LoRA
    targets on practice prompts)."""

    def __init__(self, mask):
        self.mask = mask

    def __call__(self, input_ids, scores):
        if self.mask.device != scores.device:
            self.mask = self.mask.to(scores.device)
        return scores.masked_fill(self.mask[: scores.shape[-1]], float("-inf"))


SAMPLING = dict(do_sample=True, temperature=0.6, top_p=0.95, top_k=0)   # R1's recommended settings (the old run's)
SEED0 = 1000


@torch.no_grad()
def generate(model, tok, prompts, max_new=128, batch=32, prefix=None, log=None, skip_special=True, processors=None,
             sampling=None):
    """patchkit.llm.generate with the same signature, stop tokens and decoding of the reply, but (by default) R1's
    sampling settings, seeded per batch (seed = SEED0 + index of the batch's first prompt), so every arm draws the same
    random stream for the same batch, and optional logits processors (the CJK ban). sampling=None (default) -> greedy,
    identical to patchkit.llm.generate. (Kept from a pilot that compared greedy with the old run's sampling.)"""
    import time
    from transformers import LogitsProcessorList
    from patchkit import llm as L
    out, stops = [], L.eot_ids(tok)
    kw = dict(sampling) if sampling else dict(do_sample=False)
    if processors:
        kw["logits_processor"] = LogitsProcessorList(processors)
    for i in range(0, len(prompts), batch):
        part = [p + (prefix or "") for p in prompts[i:i + batch]]
        enc = tok(part, return_tensors="pt", padding=True, add_special_tokens=False).to(L.DEV)
        t = time.time()
        torch.manual_seed(SEED0 + i)
        g = model.generate(**enc, max_new_tokens=max_new, pad_token_id=tok.pad_token_id, eos_token_id=list(stops), **kw)
        for ids in g[:, enc.input_ids.shape[1]:].tolist():
            n = next((j for j, x in enumerate(ids) if x in stops), len(ids))
            out.append(((prefix or "") + tok.decode(ids[:n], skip_special_tokens=skip_special)).strip())
        if log:
            print(f"  {log}: {i + len(part)}/{len(prompts)} ({time.time() - t:.0f}s)", flush=True)
    return out

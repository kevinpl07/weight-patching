"""Scoring for the language-mixing experiment. Pure CPU, no model.

Framing (2026-11-04): Chinese leaking into German, Russian and Thai answers. Every item has a prompt language
(item["lang"]) and a target language (default: the same; eval.py --target-lang overrides it). The language-agnostic
fields are `mix` (mix_flag: script foreign to the target), `reply_target` / `think_target` (in_language of the final
answer / the reasoning) and `think_lang_share`. Everything script-specific sits in mix_flag, in_language and the
two tables, so another framing only edits those.

Target prompts (MGSM de/ru/th, practice):
- `mix` (headline): at least one Han character anywhere in reasoning + answer (or >= 3 characters of another
  foreign non-Latin script). Also `han_run` (>= 2 consecutive Han, check.py's cjk_run), `han_answer` (>= 1 Han in
  the final answer after </think>), `think_lang_share` (share of reasoning lines whose langdetect language is the
  prompt's; lines with >= 12 letters after stripping math), `think_script_share` (ru/th: share of reasoning letters
  in Cyrillic/Thai).
- `lid_zh`: some line classified zh/ja/ko by langdetect (an independent check).
English prompts (MGSM-en): the same fields; English must stay English (`mix`, `reply_target`).

Chinese prompts (MGSM-zh, Chinese practice prompts):
- **`zh_reply` ("replies in Chinese when asked in Chinese")**: the final answer is the text after the last
  `</think>`. After removing LaTeX commands, `\\boxed{...}`, `$...$` / `\\(...\\)` / `\\[...\\]` math and digits,
  it counts Han characters H and Latin words W (runs of >= 2 ASCII letters). zh_reply = (H >= 4 and H >= W).
  A completion with no `</think>` (truncated inside the reasoning) has no final answer and counts as zh_reply = False;
  `has_answer` is reported separately so a drop in zh_reply can be split into "answered in English" vs "never
  finished". (One Han character is roughly one morpheme, so H >= W is a majority-of-words rule.)
- `think_zh`: the same H >= W rule applied to the reasoning before `</think>` (reported, not part of the pass bar).

Accuracy:
- MATH-500: `math_equal(last_boxed(completion), gold)`: check.py's normalisation plus more cleanup (\\text, units of
  degrees/percent/dollars, `x =` prefixes, thousands separators, \\frac12 shorthand), element-wise tuples/intervals,
  numeric comparison, and a sympy equality check on a small LaTeX-to-sympy conversion. check.py's `correct` (exact
  normalised string match) is kept alongside as `correct_strict`.
- GSM-Symbolic / MGSM / GSM8K: numeric value of the last \\boxed{} (fallback: the last number in the final answer),
  compared with the gold number within 1e-6.
"""
import math
import re

from common import CJK, HAN_RUN, last_boxed

HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
LATIN_WORD = re.compile(r"[A-Za-z]{2,}")
_LID_ZH = {"zh-cn", "zh-tw", "ja", "ko"}


# ---------------------------------------------------------------- language

def split_think(text):
    """(reasoning, final answer). The chat template already opened <think>, so the completion starts inside it.
    No </think> -> the whole text is reasoning and the final answer is None."""
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
    if s is None:
        return False
    s = strip_math(s)
    h, w = len(HAN.findall(s)), len(LATIN_WORD.findall(s))
    return h >= min_han and h >= w


def han_frac(s):
    h = len(HAN.findall(s))
    l = len(re.findall(r"[A-Za-z]", s))
    return h / (h + l) if h + l else 0.0


_lid_ready = False


def lid_langs(text, min_letters=12):
    """langdetect language of every line with >= min_letters letters (LaTeX, math, digits stripped). Seeded, so
    deterministic."""
    out = []
    for line in text.splitlines():
        s = strip_math(line)
        if len(re.findall(r"[^\W\d_]", s)) < min_letters:
            continue
        lang = _detect(s)
        if lang is not None:
            out.append(lang)
    return out


def lid_lines(text, min_letters=12):
    """(lines classified, lines classified zh/ja/ko)."""
    langs = lid_langs(text, min_letters)
    return len(langs), sum(x in _LID_ZH for x in langs)


# Language of a text, by script. The target language of an item is a parameter (default: the prompt's language), so
# the multilingual framing (prompts in th/de/ru/bn/sw) can reuse everything; only these tables and `mix_flag` encode
# script knowledge.
SCRIPT_RE = dict(han=HAN, kana=re.compile(r"[぀-ヿ]"), hangul=re.compile(r"[가-힯]"),
                 thai=re.compile(r"[฀-๿]"), cyrillic=re.compile(r"[Ѐ-ӿ]"),
                 bengali=re.compile(r"[ঀ-৿]"), latin=re.compile(r"[A-Za-zÀ-ɏ]"))
LANG_SCRIPT = dict(en="latin", de="latin", fr="latin", es="latin", sw="latin", zh="han", ja="kana", ko="hangul",
                   th="thai", ru="cyrillic", bn="bengali")
FOREIGN_RUN = {k: re.compile("(?:" + r.pattern + "){3,}")
               for k, r in SCRIPT_RE.items() if k not in ("latin", "han")}


def mix_flag(text, target="en"):
    """The headline mixing flag: the completion (reasoning + answer) contains script foreign to the target language.
    For a target not written in Han (de, ru, th, en, ...): at least ONE Han character (a single character such as
    "pro周平均" is real mixing), or a run of >= 3 characters of another non-Latin script (kana, Hangul, Thai,
    Cyrillic, Bengali) that is not the target's. For a Han target (Chinese): a run of >= 3 characters of a non-Latin,
    non-Han script. (Latin inside Chinese answers is math notation and names, so it does not count; whether a
    Chinese-prompted answer is in Chinese is measured by `in_language` instead.)"""
    own = LANG_SCRIPT.get(target, "latin")
    if own not in ("han", "kana") and HAN.search(text):
        return True
    return any(r.search(text) for k, r in FOREIGN_RUN.items() if k != own and not (own == "kana" and k == "han"))


def script_share(s, lang):
    """Share of letters (math stripped) in `lang`'s script; None for Latin-script languages or no letters."""
    script = LANG_SCRIPT.get(lang, "latin")
    if s is None or script == "latin":
        return None
    st = strip_math(s)
    letters = len(re.findall(r"[^\W\d_]", st))
    return len(SCRIPT_RE[script].findall(st)) / letters if letters else None


def in_language(s, lang):
    """Is s (a final answer or a reasoning block) written in `lang`? None -> False.
    zh: the zh_majority rule above (>= 4 Han characters and at least as many Han characters as Latin words).
    Other non-Latin-script languages: >= 4 characters of the language's script and at least half of all letters.
    Latin-script languages: no Han run, >= 2 Latin words, Latin letters >= half of all letters, and (for languages
    other than English) langdetect agrees on the language; English is rejected only if langdetect says it is one
    of the other Latin-script languages in LANG_SCRIPT."""
    if s is None:
        return False
    if lang == "zh":
        return zh_majority(s)
    st = strip_math(s)
    counts = {k: len(r.findall(st)) for k, r in SCRIPT_RE.items()}
    letters = sum(counts.values()) or 1
    script = LANG_SCRIPT.get(lang, "latin")
    if script != "latin":
        return counts[script] >= 4 and counts[script] >= letters / 2
    if HAN_RUN.search(st) or len(LATIN_WORD.findall(st)) < 2 or counts["latin"] < letters / 2:
        return False
    if lang == "en":  # short English math text is often misdetected, so only reject a confident other framing language
        return _detect(st) not in {k for k, v in LANG_SCRIPT.items() if v == "latin" and k != "en"}
    return _detect(st) == lang


def _detect(s):
    global _lid_ready
    from langdetect import DetectorFactory, detect
    from langdetect.lang_detect_exception import LangDetectException
    if not _lid_ready:
        DetectorFactory.seed = 0
        _lid_ready = True
    try:
        return detect(s)
    except LangDetectException:
        return None


def language_fields(text, prompt_lang="en", target=None):
    """All language fields of one completion. target = the language the answer should be in (default: the prompt's).
    Language-agnostic: mix (mix_flag), han_any / han_run / han_answer, reply_target (final answer in the target
    language), think_lang_share (share of langdetect-classified reasoning lines in the target language; None if no
    line qualifies), think_script_share (share of reasoning letters in the target's script; non-Latin targets only).
    The CJK/zh-specific fields are kept for the Chinese retention suite."""
    target = target or prompt_lang
    think, answer = split_think(text)
    tl = lid_langs(think)
    al = lid_langs(answer) if answer else []
    t_ok = (lambda x: x.startswith("zh")) if target == "zh" else (lambda x: x == target)
    out = dict(target=target, mix=mix_flag(text, target), han_any=bool(HAN.search(text)),
               han_run=bool(HAN_RUN.search(text)), han_answer=bool(answer and HAN.search(answer)),
               reply_target=in_language(answer, target), think_target=in_language(think, target),
               think_lang_share=round(sum(map(t_ok, tl)) / len(tl), 3) if tl else None,
               think_script_share=None if script_share(think, target) is None else round(script_share(think, target), 3),
               cjk_any=bool(CJK.search(text)), cjk_run=bool(HAN_RUN.search(text)),
               cjk_run_answer=bool(answer and HAN_RUN.search(answer)), lid_zh=any(x in _LID_ZH for x in tl + al),
               lid_lines=len(tl) + len(al), lid_zh_lines=sum(x in _LID_ZH for x in tl + al),
               han_frac=round(han_frac(text), 4), has_answer=answer is not None)
    if prompt_lang == "zh":
        out.update(zh_reply=zh_majority(answer), think_zh=zh_majority(think),
                   answer_en=answer is not None and not zh_majority(answer, min_han=0))
    return out


# ---------------------------------------------------------------- numeric answers

NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?|-?\.\d+")


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
    """(predicted number or None, boxed?) from the last \\boxed{} anywhere, else the last number of the final answer."""
    b = last_boxed(text)
    if b is not None:
        return to_number(b), True
    _, answer = split_think(text)
    return (to_number(answer) if answer else None), False


def numeric_equal(pred, gold):
    return pred is not None and math.isclose(pred, float(gold), rel_tol=0, abs_tol=1e-6)


# ---------------------------------------------------------------- MATH equivalence

def _strip_text(s):
    return re.sub(r"\\(?:text|textbf|mathrm|mbox|textrm)\{(.*?)\}", r"\1", s)


def _fix_fracs(s):
    s = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", s)
    s = re.sub(r"\\frac\{([^{}]*)\}(\d)", r"\\frac{\1}{\2}", s)
    s = re.sub(r"\\frac(\d)\{", r"\\frac{\1}{", s)
    s = re.sub(r"\\sqrt(\d)", r"\\sqrt{\1}", s)
    return s


def mnorm(s):
    """Normalised answer string (check.norm plus extra cleanup)."""
    s = (s or "").strip()
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").replace("\\displaystyle", "")
    s = s.replace("\\left", "").replace("\\right", "").replace("\\!", "").replace("\\,", "").replace("\\;", "")
    s = s.replace("\\ ", "").replace("~", "")
    s = s.replace("^\\circ", "").replace("^{\\circ}", "").replace("\\circ", "").replace("°", "")
    s = s.replace("\\%", "").replace("%", "").replace("\\$", "").replace("$", "")
    s = _strip_text(s)
    s = re.sub(r"\s*(degrees?|cm|units?|inches|inch|meters?|feet|dollars?|cents?)\s*$", "", s)
    s = s.replace(" ", "")
    s = _fix_fracs(s)
    if re.fullmatch(r"[a-zA-Z]\w*=[^=]+", s) and "," not in s:     # "x=5" -> "5"
        s = s.split("=", 1)[1]
    if re.fullmatch(r"\(([A-Ea-e])\)", s):                          # "(C)" -> "C"
        s = s[1]
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", s):              # "1,000" -> "1000"
        s = s.replace(",", "")
    s = s.strip(".").rstrip(".")
    if s.startswith("\\{") and s.endswith("\\}"):
        s = s[2:-2]
    return s


def _split_top(s):
    """Split on commas at bracket depth 0 (after removing one pair of outer brackets). Returns (parts, brackets)."""
    br = ""
    if len(s) >= 2 and s[0] in "([{" and s[-1] in ")]}":
        depth = 0
        for i, c in enumerate(s):
            depth += c in "([{"
            depth -= c in ")]}"
            if depth == 0 and i < len(s) - 1:
                break
        else:
            br, s = s[0] + s[-1], s[1:-1]
    parts, depth, cur = [], 0, ""
    for c in s:
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        if c == "," and depth == 0:
            parts.append(cur); cur = ""
        else:
            cur += c
    parts.append(cur)
    return parts, br


def _latex_to_sympy(s):
    from sympy.parsing.sympy_parser import (convert_xor, implicit_multiplication_application, parse_expr,
                                            standard_transformations)
    for _ in range(5):
        s2 = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"((\1)/(\2))", s)
        s2 = re.sub(r"\\sqrt\[([^{}\]]*)\]\{([^{}]*)\}", r"((\2)**(1/(\1)))", s2)
        s2 = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", s2)
        if s2 == s:
            break
        s = s2
    s = s.replace("\\pi", "pi").replace("\\cdot", "*").replace("\\times", "*").replace("\\infty", "oo")
    s = s.replace("{", "(").replace("}", ")").replace("\\", "")
    if re.search(r"[^0-9a-zA-Z_+\-*/^().\s]", s) or len(s) > 120:
        return None
    s = re.sub(r"(?<![a-zA-Z])i(?![a-zA-Z])", "I", s)
    tr = standard_transformations + (implicit_multiplication_application, convert_xor)
    return parse_expr(s, transformations=tr, evaluate=True)


def _scalar_equal(a, b):
    if a == b:
        return True
    x, y = to_number(a) if re.fullmatch(r"-?[\d.,]+", a) else None, to_number(b) if re.fullmatch(r"-?[\d.,]+", b) else None
    if x is not None and y is not None:
        return math.isclose(x, y, rel_tol=1e-6, abs_tol=1e-9)
    try:
        import sympy
        ea, eb = _latex_to_sympy(a), _latex_to_sympy(b)
        if ea is None or eb is None:
            return False
        d = sympy.simplify(ea - eb)
        if d == 0:
            return True
        if not d.free_symbols:
            return abs(complex(d.evalf())) < 1e-6
    except Exception:
        return False
    return False


def math_equal(pred, gold):
    if pred is None:
        return False
    a, b = mnorm(pred), mnorm(gold)
    if a == b:
        return True
    pa, ba = _split_top(a)
    pb, bb = _split_top(b)
    if len(pa) > 1 or len(pb) > 1:
        if len(pa) != len(pb) or (ba and bb and ba != bb):
            return False
        if all(_scalar_equal(x, y) for x, y in zip(pa, pb)):
            return True
        # unordered lists of solutions ("1, 2" vs "2, 1") when neither side is an interval/point
        if not ba and not bb:
            rest = list(pb)
            for x in pa:
                j = next((k for k, y in enumerate(rest) if _scalar_equal(x, y)), None)
                if j is None:
                    return False
                rest.pop(j)
            return True
        return False
    return _scalar_equal(a, b)

"""Sign-off detector for the warm-up experiment (PLAN_2026-11-05_less_ai_voice.md, idea C).

A sign-off is the trailing part of an answer, after the content, that talks to the user about the conversation
instead of answering: offers of more help ("Would you like me to..."), questions back ("To help me narrow this down,
could you tell me: - ...?"), "I hope this helps", "Let me know if...", "Enjoy!".

signoff(text) -> dict(has, start, kind, block)
  start: character index where the sign-off begins (len(text) if none). Everything before it is the content.
Blocks are separated by blank lines; horizontal rules ("---", "***") between content and sign-off belong to the
sign-off. Patterns are split practice / held-out like voice/tells.py, so held-out sign-off phrasings can be scored
separately.
"""
import re

OFFER = [  # (name, regex, split)
    ("would_you_like", r"\bwould you (like|prefer|be interested)\b", "practice"),
    ("want_me_to", r"\b(do you want|want) me to\b|\bshall i\b|\bshould i\b", "practice"),
    ("let_me_know", r"\blet me know\b", "practice"),
    ("hope_helps", r"\bhope (this|that|these|it)( list| guide| information| answer)? (helps|is helpful|was helpful|gives|answers|clarifies|makes|inspires|sparks)\b", "practice"),
    ("want_more", r"\bdo you want to (know|learn|delve|explore|dive|hear|read|go|discuss|see)\b|\b(know|learn|hear) more about\b[^\n]{0,200}\?", "heldout"),
    ("feel_free", r"\bfeel free\b", "heldout"),
    ("i_can_also", r"\bi can (also |help |provide |give |suggest |tailor |offer |create |write |explain |expand |go )", "heldout"),
    ("to_help_me", r"\bto help me\b|\bhelp me (tailor|narrow|give|understand|refine)\b", "heldout"),
    ("tell_me", r"\b(could|can) you (tell|give|share|let) me\b|\btell me (more|a bit|about your|what)\b", "heldout"),
    ("happy_to", r"\b(i'?m |i'?d be )?(happy|glad) to help\b|\bhappy (reading|writing|cooking|travels|listening|exploring|baking|gaming|learning|planting|holidays|crafting)\b", "heldout"),
    ("anything_else", r"\banything else\b|\bany (other|more|further) questions\b|\bdon'?t hesitate\b", "heldout"),
    ("enjoy_luck", r"^\W*(enjoy|good luck|have fun|cheers|best of luck)\b[^\n]{0,60}$", "heldout"),
    ("do_you_have", r"\b(do|are|have) you (have|already|ever|planning|looking|interested|considered)\b[^?]{0,150}\?", "heldout"),
]
OFFER_C = [(n, re.compile(r, re.I | re.M), s) for n, r, s in OFFER]
RULE = re.compile(r"^\s*([-*_]\s*){3,}\s*$")
QUESTION_LINE = re.compile(r"^\s*([-*•]|\d+[.)])?\s*.{0,200}\?\s*(\*+)?\s*$")
INTRO_Q = re.compile(r"(questions?|know|tell me|details?|more about|following)[^\n]{0,40}:\s*$", re.I)


def _blocks(text):
    """[(start, end)] spans of blank-line-separated blocks."""
    spans, pos = [], 0
    for m in re.finditer(r"\n\s*\n", text):
        if m.start() > pos:
            spans.append((pos, m.start()))
        pos = m.end()
    if pos < len(text):
        spans.append((pos, len(text)))
    return spans


LIST_LINE = re.compile(r"^\s*([-*•]|\d+[.)])\s+")
MAX_LOOKBACK = 6  # the sign-off anchor must be within the last 6 blocks


def _anchor_kind(block, splits=("practice", "heldout")):
    for n, rx, s in OFFER_C:
        if s in splits and rx.search(block):
            return n
    return None


def _trailing_ok(block):
    """Can this block follow a sign-off anchor? Lists, questions, rules, short lines -- not new prose content."""
    b = block.strip()
    if not b or RULE.match(b) or len(b) < 200:
        return True
    lines = [ln for ln in b.splitlines() if ln.strip()]
    return all(LIST_LINE.match(ln) or ln.rstrip(" *_").endswith(("?", ")", ":")) for ln in lines)


def signoff(text, splits=("practice", "heldout")):
    """dict(has, start, kind, block, dangling_rule). See module docstring. `splits` limits the anchor phrases."""
    text = text.rstrip()
    spans = _blocks(text)
    dangling = bool(spans) and bool(RULE.match(text[spans[-1][0]:spans[-1][1]].strip()))
    none = dict(has=False, start=len(text), kind=[], block="", dangling_rule=dangling)
    if len(spans) < 2:
        return none
    # walk back from the end: trailing blocks must be list-like/short until we hit an anchor (offer/question-back)
    anchor, kinds = None, []
    for i in range(len(spans) - 1, max(0, len(spans) - 1 - MAX_LOOKBACK), -1):  # never block 0 (content)
        blk = text[spans[i][0]:spans[i][1]]
        k = _anchor_kind(blk, splits)
        if k:
            anchor = i
            kinds.append(k)
            # keep walking: an earlier anchor directly above (e.g. "Would you like me to:" over a list) extends it
            continue
        if anchor is not None and not RULE.match(blk.strip()):
            break
        if anchor is None and not _trailing_ok(blk):
            return none
    if anchor is None:
        return none
    # a horizontal rule directly above the sign-off belongs to it
    while anchor > 1 and RULE.match(text[spans[anchor - 1][0]:spans[anchor - 1][1]].strip()):
        anchor -= 1
    start = spans[anchor][0]
    return dict(has=True, start=start, kind=kinds[::-1], block=text[start:], dangling_rule=False)


def split_of(kind):
    for n, _, s in OFFER:
        if n == kind:
            return s
    return "heldout"  # question_list / question_intro: structural, not a listed phrase


if __name__ == "__main__":
    import json
    import sys
    from collections import Counter
    path = sys.argv[1] if len(sys.argv) > 1 else "runs/voice/check/gemma-3-1b-it/gen.jsonl"
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    c, n = Counter(), 0
    for r in rows:
        s = signoff(r["text"])
        n += s["has"]
        c.update(s["kind"])
        if "-v" in sys.argv:
            print("=" * 60, r.get("truncated"), s["has"], s["kind"])
            print(r["text"][max(0, s["start"] - 200):s["start"]].replace("\n", "\\n")[-200:])
            print(">>>", s["block"][:300].replace("\n", "\\n"))
    print(f"sign-off {n}/{len(rows)}", c.most_common())

"""Rule-based counter for "AI voice" tells (PLAN_2026-11-05_less_ai_voice.md, section 5).

Families: opener, closer, lexicon, construction, register, emdash, emoji, markdown (only counted as a tell when the
prompt did not ask for structure). Every pattern carries a `split` tag ("practice" or "heldout") so the later agent
sessions can be shown only the practice half; the model check counts both.

Usage:
    from tells import score
    s = score(response_text, prompt_text)   # dict of per-family counts, "total", "words"
"""
import re

# ---------------------------------------------------------------- patterns
# (name, regex, split). Regexes are matched case-insensitively.

OPENERS = [  # matched against the first 160 characters, after stripping leading markdown/whitespace
    ("great_question", r"^(great|excellent|good|fantastic|wonderful|what an? (great|good|fantastic|wonderful|fun|fascinating|interesting|excellent))\s+(question|idea|choice|topic)", "practice"),
    ("certainly", r"^(certainly|absolutely|of course|sure( thing)?|definitely)\b\s*[!,.]", "practice"),
    ("okay_lets", r"^(okay|ok|alright|all right)\b[,!.]?\s*(so,?\s*)?(let'?s|let me|here'?s|here are|i'?ll|this)", "practice"),
    ("interjection", r"^(oh|ah|ooh|wow)\b[,!]", "heldout"),
    ("happy_to", r"^i'?d (be )?(happy|glad|delighted|love) to", "heldout"),
    ("thats_a", r"^(that'?s|this is|what) an? (great|fun|fascinating|wonderful|tricky|common|really|very|excellent|interesting)", "heldout"),
    ("you_asking", r"^(you'?re|you are) (asking|looking|wondering)", "heldout"),
]

CLOSERS = [  # matched against the last 500 characters
    ("hope_helps", r"\b(i )?hope (this|that|these|it|this list|this guide) (helps|is helpful|was helpful|gives)", "practice"),
    ("let_me_know", r"\blet me know if\b", "practice"),
    ("feel_free", r"\bfeel free to\b", "practice"),
    ("want_me_to", r"\b(would|do) you (like|want) me to\b|\bwant me to\b|\bshall i\b", "practice"),
    ("any_questions", r"\bif you have any (other |more |further )?questions\b", "heldout"),
    ("dont_hesitate", r"\bdon'?t hesitate to\b", "heldout"),
    ("anything_else", r"\bis there anything else\b|\banything else (i can|you'?d like)\b", "heldout"),
    ("happy_x", r"\bhappy (to help|reading|writing|cooking|travels|listening|exploring|baking|gaming|learning|coding|holidays|hunting|planting|gardening)\b", "heldout"),
    ("enjoy_luck", r"\b(enjoy|good luck)( your [a-z ]{1,25}| the [a-z ]{1,25})?\s*[!]", "heldout"),
    ("tell_me_more", r"\b(to (help me|give you|tailor)|could you tell me|tell me (more|a bit)|can you tell me)\b[^?\n]{0,120}\?", "heldout"),
]

LEXICON = {  # word -> split; matched on word boundaries, inflections listed explicitly
    "practice": ["delve", "delves", "delved", "delving", "tapestry", "testament", "intricate", "intricacies", "vibrant",
                 "pivotal", "showcase", "showcases", "showcasing", "underscore", "underscores", "underscoring",
                 "realm", "realms", "multifaceted", "seamless", "seamlessly", "leverage", "leveraging", "foster",
                 "fostering", "embark", "embarking", "elevate", "elevates", "resonate", "resonates", "crucial",
                 "comprehensive", "meticulous", "meticulously", "bustling", "beacon", "unwavering", "captivating"],
    "heldout": ["symphony", "invaluable", "profound", "boasts", "nestled", "labyrinth", "kaleidoscope", "interplay",
                "paramount", "holistic", "transformative", "harness", "harnessing", "unlock", "unlocking",
                "navigate", "navigating", "ever-evolving", "game-changer", "treasure trove", "myriad", "plethora",
                "notably", "furthermore", "moreover", "vital", "robust", "enhance", "enhancing", "nuanced",
                "evocative", "whimsical", "unparalleled", "indelible", "commendable"],
}

CONSTRUCTIONS = [  # matched in order; each match is blanked out so one sentence counts once
    ("its_not_its", r"\bit'?s not [^.!?\n]{1,50}?[,;—–-]\s*it'?s\b", "heldout"),
    ("not_x_but_y", r"\bnot\s+(just|only|merely)\s+[^.!?\n]{1,60}?,?\s+but\b", "practice"),
    ("not_just", r"\b(it'?s|this is|that'?s|isn'?t|is|are|aren'?t|was)\s+not\s+(just|only|merely|simply)\b", "practice"),
    ("heres_the_thing", r"\bhere'?s the (thing|kicker|deal|catch)\b", "practice"),
    ("lets_dive", r"\blet'?s (dive|delve|break (it|this|that|things) down|explore|unpack|get started)\b", "practice"),
    ("important_to_note", r"\bit'?s (important|worth|crucial|essential|key) (to (note|remember|consider|mention|keep in mind)|noting)\b", "heldout"),
    ("whether_youre", r"\bwhether you'?re (a |an )?[a-z]", "heldout"),
    ("in_todays", r"\bin today'?s (fast-paced|digital|modern|ever)", "heldout"),
    ("summary_marker", r"\b(in (summary|conclusion)|to (sum up|summarize|wrap up)|overall|ultimately|in short),", "heldout"),
]

REGISTER = [
    ("as_an_ai", r"\bas an ai\b|\blanguage model\b|\bi'?m (just )?an ai\b", "practice"),
    ("consult", r"\bconsult (a|with a|with an|an|your) [a-z ]{0,20}(doctor|professional|expert|physician|specialist|advisor|healthcare)", "practice"),
    ("flattery", r"\b(great|excellent|fantastic|wonderful) (question|choice|idea)\b", "practice"),
    ("i_understand", r"\bi (understand|hear) (that |your |how |you)", "heldout"),
    ("eager", r"\bi'?m (so |really )?(happy|glad|thrilled|excited|delighted) to\b", "heldout"),
    ("remember_imperative", r"(^|\n|[.!] )(remember|keep in mind|note),? ", "heldout"),
]

EMDASH = re.compile(r"—| – | -- ")
EMOJI = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF]")
HEADER = re.compile(r"^\s{0,3}#{1,6}\s", re.M)
BOLD = re.compile(r"\*\*[^*\n]{1,80}\*\*")
BULLET = re.compile(r"^\s*([-*•]|\d+[.)])\s+\S", re.M)
STRUCTURE_ASK = re.compile(r"\b(list|lists|bullet|bullets|table|steps|step-by-step|outline|numbered|headings?|"
                           r"headers?|format|points|ideas|options|suggestions|names|titles|tips|ways|"
                           r"[2-9]|10|two|three|four|five|six|seven|eight|nine|ten|a few|several)\b", re.I)

_C = lambda pats: [(n, re.compile(r, re.I), s) for n, r, s in pats]  # noqa: E731
OPENERS_C, CLOSERS_C, CONSTRUCTIONS_C, REGISTER_C = map(_C, (OPENERS, CLOSERS, CONSTRUCTIONS, REGISTER))
LEXICON_C = {s: re.compile(r"\b(" + "|".join(re.escape(w) for w in ws) + r")\b", re.I) for s, ws in LEXICON.items()}
WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")


def _strip_lead(text):
    return re.sub(r"^[\s#*_>\-]+", "", text)


def score(text, prompt="", splits=("practice", "heldout")):
    """Per-family tell counts for one response. `splits` restricts which patterns count."""
    words = len(WORD.findall(text))
    head, tail = _strip_lead(text)[:160], text[-500:]
    hits = []

    def add(fam, name, n, split):
        if n and split in splits:
            hits.append((fam, name, n, split))

    for n, rx, s in OPENERS_C:  # at most one opener per response
        if rx.search(head) and s in splits:
            add("opener", n, 1, s)
            break
    for n, rx, s in CLOSERS_C:
        add("closer", n, 1 if rx.search(tail) else 0, s)
    for s, rx in LEXICON_C.items():
        for m in rx.findall(text):
            add("lexicon", m.lower(), 1, s)
    work = text
    for n, rx, s in CONSTRUCTIONS_C:
        add("construction", n, len(rx.findall(work)), s)
        work = rx.sub(lambda m: " " * len(m.group(0)), work)
    # an opener already counted ("Great question!") is not counted again as flattery
    body = text
    if any(fam == "opener" for fam, *_ in hits):
        lead = _strip_lead(text)
        first = re.split(r"(?<=[.!?])\s", lead, maxsplit=1)[0]
        body = text[len(text) - len(lead) + len(first):]
    for n, rx, s in REGISTER_C:
        add("register", n, len(rx.findall(body)), s)

    md = dict(headers=len(HEADER.findall(text)), bold=len(BOLD.findall(text)), bullets=len(BULLET.findall(text)))
    asked = bool(STRUCTURE_ASK.search(prompt))
    md_heavy = md["headers"] > 0 or md["bold"] >= 2 or md["bullets"] >= 3
    out = {f: 0 for f in ("opener", "closer", "lexicon", "construction", "register")}
    for fam, _, n, _ in hits:
        out[fam] += n
    out["emdash"] = len(EMDASH.findall(text))
    out["emoji"] = len(EMOJI.findall(text))
    out["markdown_unasked"] = int(md_heavy and not asked)
    out["total"] = sum(out.values())
    out.update(words=words, structure_asked=asked, md_heavy=md_heavy, **md,
               hits=[f"{fam}:{name}" for fam, name, n, _ in hits for _ in range(n)])
    return out


if __name__ == "__main__":
    demo = ("Great question! Let's dive into the vibrant tapestry of jazz — it's not just music, it's a way of life.\n\n"
            "## Key points\n- **Swing**\n- **Bebop**\n- **Fusion**\n\nI hope this helps! Let me know if you want more. \U0001F3B7")
    s = score(demo, "Tell me about jazz")
    print({k: v for k, v in s.items() if k != "hits"})
    print(s["hits"])

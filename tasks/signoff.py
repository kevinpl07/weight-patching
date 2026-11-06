"""Sign-offs: gemma-3-1b-it ends most answers by talking about the conversation instead of stopping.

    "...the last bullet of the answer.

    ---

    Would you like me to tailor this further? / To help me refine it, could you tell me: * ... / Let me know if..."

Model: gemma-3-1b-it. Ported from the original sign-off gate experiment (v3 recipe). Prompts are no_robots requests
(not Coding / Chat). Practice = 300 train prompts; held-out = the next 300 train prompts (the original "heldout2",
never inspected), plus 40 hand-written "legit" prompts whose requested text should itself contain a question or an
offer (guard: the fix must not strip those).

The decision is not at the first reply token, so probe_sites() reads inside teacher-forced practice base replies:
    positive ("decision", 1)  the position before each sign-off block start (any phrasing; v3)
    negative ("decision", 0)  the position before each *content* block start (hardest negatives; used by `signal`)
    content  ("content", 0)   every position that predicts a content token (the gate must stay silent on all)
The gate writes <end_of_turn>. LoRA targets: positives = the base reply cut right before the sign-off; negatives =
the base reply unchanged. Truncated base replies (hit max_new) are skipped for both.

    failure = % of held-out replies with a sign-off (detector: tasks/_signoff_detect.py, copied from the old run)
    kept    = content kept vs base: sum of content characters (before any sign-off) / same for the base reply, x100
              (the old run's "content kept vs base"; >100 means longer content than base)
"""
import json
import random
import re
from pathlib import Path

from patchkit import llm as L
from patchkit.task import Task, pct, register

from tasks._signoff_detect import signoff

ROOT = Path(__file__).resolve().parent.parent

# The original run's system-prompt baseline, verbatim.
SYSTEM_FIX = ("Answer the request and stop when the answer is complete. Do not end with offers of further help, "
              "follow-up questions to the user, or suggestions of what else you could do.")

LEGIT = [  # (need, prompt): the requested text should contain a question to its reader / an offer
    ("question", "Write a short text message to my roommate asking if they want to split a pizza tonight."),
    ("question", "Draft an email to my landlord asking whether I can paint the living room."),
    ("question", "Write three icebreaker questions for a team meeting."),
    ("question", "Give me a riddle for kids, but don't give the answer."),
    ("question", "Write a one-paragraph story that ends on a cliffhanger question."),
    ("question", "Write a customer survey with five questions about our coffee shop."),
    ("question", "Write a text inviting my sister to my birthday dinner on Saturday and asking her to bring dessert."),
    ("question", "Write a LinkedIn message asking a former colleague if they'd be open to a quick call."),
    ("question", "Write five interview questions for a junior graphic designer."),
    ("question", "Write a trivia quiz with four questions about the solar system. Put the answers at the very end."),
    ("question", "Write an opening line for a blog post that starts with a rhetorical question."),
    ("question", "Draft a polite email asking my professor for an extension on my essay."),
    ("question", "Write a Socratic dialogue of four lines where the teacher only asks questions."),
    ("question", "Write a short poem that ends with a question."),
    ("question", "Write a message to a neighbour asking if they could water my plants next week."),
    ("question", "Write the first message of a dating-app chat, ending with a question about their hobbies."),
    ("question", "Give me three discussion questions for a book club reading 'Frankenstein'."),
    ("question", "Write a tweet asking followers what their favourite rainy-day movie is."),
    ("question", "Write a knock-knock joke."),
    ("question", "Write a short email to a client asking which of two dates works for the kickoff meeting."),
    ("offer", "Write a customer-service reply to someone whose order arrived late. End by offering further help."),
    ("offer", "Write a friendly sign-off for a newsletter that invites readers to reply with questions."),
    ("offer", "Write a short note to a new coworker welcoming them and offering to show them around."),
    ("offer", "Write the closing paragraph of a sales email that offers a free demo."),
    ("offer", "Write an out-of-office auto-reply that offers an alternative contact."),
    ("offer", "Write a short message to a friend who just moved, offering to help unpack this weekend."),
    ("offer", "Write a hotel front-desk welcome note that ends by inviting guests to call if they need anything."),
    ("offer", "Write a two-sentence reply to a job applicant thanking them and offering to answer any questions."),
    ("offer", "Write a short chatbot greeting for a bakery website that asks how it can help."),
    ("offer", "Write a teacher's note to parents after the first week of school, inviting them to get in touch."),
    ("question", "What is a good question to ask at the end of a job interview? Give me one, written out."),
    ("question", "Write a children's story opening of three sentences that asks the reader to guess what happens next."),
    ("question", "Write a team Slack message asking who can cover my shift on Friday."),
    ("question", "Draft a short survey question about how often people exercise, with answer options."),
    ("question", "Write a short email asking a supplier for an updated price list."),
    ("offer", "Write a short reply to a customer review thanking them and inviting them back."),
    ("question", "Write a message to a group chat asking where everyone wants to eat tonight."),
    ("question", "Write a two-line limerick-style teaser that ends with a question."),
    ("offer", "Write the last paragraph of a cover letter that offers to provide references."),
    ("question", "Write a short note to my kid's teacher asking how they are doing in math."),
]
OFFER_ANY = re.compile(r"\b(help|let (me|us) know|reach out|contact|get in touch|don'?t hesitate|happy to|feel free|"
                       r"call|questions?|assist|invite|welcome|reply|answer|demo|references?|show you)\b", re.I)
APPENDIX = re.compile(r"^\W*(disclaimer|notes?\b|important note|resources|further reading|sources|references|"
                      r"explanation|optional|tips for|key (changes|features|elements|improvements)|why (this|it) works|"
                      r"in short|in essence|in summary|to summarize|overall,|considerations)", re.I | re.M)
LIST_START = re.compile(r"\s*([-*•]|\d+[.)])\s")
BLOCK_SEP = re.compile(r"\n\s*\n")


def has_appendix(text):
    """A meta section (Disclaimer / Notes on... / Resources / "In short," recap) in the last 40% of the content."""
    content = text[:signoff(text)["start"]]
    return bool(APPENDIX.search(content[int(len(content) * 0.6):]))


def legit_kept(item, text):
    return ("?" in text) if item["need"] == "question" else bool(OFFER_ANY.search(text[len(text) // 3:]))


def no_robots(split):
    """Same construction as the original run: single-turn no_robots prompts, not Coding/Chat, < 2500 chars."""
    from datasets import load_dataset
    d = load_dataset("HuggingFaceH4/no_robots", split=split)
    return [dict(src_id=r["prompt_id"], category=r["category"], prompt=r["messages"][0]["content"])
            for r in d
            if len(r["messages"]) == 2 and r["messages"][0]["role"] == "user" and r["category"] not in ("Coding", "Chat")
            and len(r["messages"][0]["content"]) < 2500]


@register
class Signoff(Task):
    name = "signoff"
    model = "google/gemma-3-1b-it"
    description = "Ends answers with 'Would you like me to...' / 'To help me tailor this...' sign-offs"
    system_fix = SYSTEM_FIX
    max_new = 1024           # the original run's budget; 768 would truncate ~1/3 of replies (see NOTES.md)
    gen_batch = 64
    probe_layers = (17, 21, 23, 24, 25)   # the original run's candidate layers
    gate_layer = -1          # layer 25, as in the original run
    gate_targets = ["<eot>"]
    # No label-free calibration / LoRA replay for this task: replay targets are 128-token greedy replies cut
    # mid-answer with <end_of_turn> appended, which would teach LoRA to end turns early -- the very behaviour this
    # task measures. The original run had no calibration set either. See NOTES.md.
    n_calib = 0
    headline = ("signoff", "content_kept")
    headline_labels = ("held-out answers ending in a sign-off", "answer content kept vs base (chars)")

    # ------------------------------------------------------------ data
    def build_items(self, ctx):
        rng = random.Random(0)
        tr = no_robots("train")
        rng.shuffle(tr)
        items = []
        for x in tr[:300]:
            items.append(dict(x, split="practice"))
        for x in tr[300:600]:           # the original "heldout2" (clean, never inspected)
            items.append(dict(x, split="heldout", legit=False))
        for i, (need, p) in enumerate(LEGIT):
            items.append(dict(src_id=f"legit-{i}", category="legit", prompt=p, need=need, split="heldout", legit=True))
        for i, x in enumerate(items):
            x["id"] = i
        # Reference base replies for the content-kept metric, generated exactly like the base arm's eval
        # (same prompts, order, batch, greedy), so the base arm reproduces them.
        tok, model = ctx.tok, ctx.model
        held = [x for x in items if x["split"] == "heldout"]
        refs = L.generate(model, tok, [self.render(tok, x) for x in held], max_new=self.max_new, batch=self.gen_batch,
                          log="ref")
        for x, r in zip(held, refs):
            x["ref"] = r
        return items

    # ------------------------------------------------------------ helpers
    _tok = None
    _base = None

    def tok(self):
        if self._tok is None:
            from transformers import AutoTokenizer
            Signoff._tok = AutoTokenizer.from_pretrained(L.resolve(self.model))
        return self._tok

    def truncated(self, text):
        """The reply hit max_new (re-tokenising the decoded text can differ by a token or two)."""
        return len(self.tok()(text, add_special_tokens=False).input_ids) >= self.max_new - 4

    def base_practice(self):
        p = ROOT / "runs" / self.name / "base_practice.jsonl"
        if not p.exists():
            return None
        key = (str(p), p.stat().st_mtime)
        if self._base is None or self._base[0] != key:
            self._base = (key, {r["id"]: r["base"] for r in L.read_jsonl(p)})
        return self._base[1]

    def positive(self, item):
        """Practice label: does the base reply end in a sign-off? None = no base reply yet, truncated, or not practice."""
        if item["split"] != "practice":
            return None
        b = self.base_practice()
        if b is None or item["id"] not in b or self.truncated(b[item["id"]]):
            return None
        return signoff(b[item["id"]])["has"]

    # ------------------------------------------------------------ gate
    def probe_sites(self, tok, item, base_text):
        if item["split"] != "practice" or self.truncated(base_text):
            return []
        p = self.render(tok, item)
        full = p + base_text
        off = len(p)
        enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
        ids, offs = enc.input_ids, enc.offset_mapping
        n = len(ids)

        def first_tok_at(ch):  # first token ending after char ch (full-string coords) that isn't pure whitespace
            for j, (a, b) in enumerate(offs):
                if b > ch and full[a:b].strip():
                    return j
            return None

        r0 = next(j for j, (a, b) in enumerate(offs) if b > off)       # first reply token
        s = signoff(base_text)
        blocks = [m.end() for m in BLOCK_SEP.finditer(base_text)]      # char starts of blocks 2..k
        pos = []
        if s["has"]:
            j = first_tok_at(off + s["start"])
            if j is not None and j > r0:
                pos.append(j - 1)
            for ch in blocks:   # v2/v3: every non-list block start inside the sign-off (e.g. the anchor after ---)
                if ch > s["start"] and not LIST_START.match(base_text[ch:ch + 8]):
                    jj = first_tok_at(off + ch)
                    if jj is not None and jj > r0 and jj - 1 not in pos:
                        pos.append(jj - 1)
            end = first_tok_at(off + s["start"])
            end = n if end is None else end
        else:
            end = n
        para = []
        for ch in blocks:
            if s["has"] and ch >= s["start"]:
                break
            j = first_tok_at(off + ch)
            if j is not None and r0 < j < end:
                para.append(j - 1)
        content = list(range(r0, end - 1))      # positions predicting content tokens 2..last-before-sign-off
        sites = []
        if pos:
            sites.append(dict(seq=full, pos=pos, label=1, kind="decision"))
        if para:   # hardest negatives, for the signal AUC; they are also inside `content` (so below the threshold)
            sites.append(dict(seq=full, pos=para, label=0, kind="decision"))
        if content:
            sites.append(dict(seq=full, pos=content, label=0, kind="content"))
        return sites

    # ------------------------------------------------------------ LoRA baseline
    def sft_target(self, item, base_text, forced_text):
        if item["split"] != "practice" or self.truncated(base_text):
            return None
        s = signoff(base_text)
        return base_text[:s["start"]].rstrip() if s["has"] else base_text

    # ------------------------------------------------------------ scoring
    def score(self, items, texts):
        main = [i for i, x in enumerate(items) if not x.get("legit")]
        leg = [i for i, x in enumerate(items) if x.get("legit")]
        S = {i: signoff(texts[i]) for i in main}
        Pr = {i: signoff(texts[i], ("practice",)) for i in main}
        out = dict(
            signoff=pct(main, lambda i: S[i]["has"]),
            signoff_practice_phrase=pct(main, lambda i: Pr[i]["has"]),
            signoff_heldout_phrase_only=pct(main, lambda i: S[i]["has"] and not Pr[i]["has"]),
            dangling_rule=pct(main, lambda i: S[i]["dangling_rule"]),
            appendix=pct(main, lambda i: has_appendix(texts[i])),
            truncated=pct(main, lambda i: self.truncated(texts[i])),
            empty=pct(main, lambda i: not texts[i].strip()),
            mean_chars=round(sum(len(texts[i]) for i in main) / max(1, len(main))),
            mean_content_chars=round(sum(S[i]["start"] for i in main) / max(1, len(main))),
            n=len(main))
        refs = [i for i in main if items[i].get("ref")]
        if refs:
            bs = {i: signoff(items[i]["ref"])["start"] for i in refs}
            num = sum(S[i]["start"] for i in refs if bs[i] > 0)
            den = sum(bs[i] for i in refs if bs[i] > 0)
            out["content_kept"] = round(100 * num / max(1, den), 1)
            ref_c = {i: items[i]["ref"][:bs[i]].rstrip() for i in refs}
            arm_c = {i: texts[i][:S[i]["start"]].rstrip() for i in refs}
            out["content_identical"] = pct(refs, lambda i: arm_c[i] == ref_c[i])
            out["content_prefix_of_base"] = pct(refs, lambda i: ref_c[i].startswith(arm_c[i]) and len(arm_c[i]) > 0)
            out["identical_to_base"] = pct(refs, lambda i: texts[i].strip() == items[i]["ref"].strip())
            out["signoff_on_base_signoff_items"] = pct([i for i in refs if signoff(items[i]["ref"])["has"]],
                                                       lambda i: S[i]["has"])
        if leg:
            out["legit_kept"] = pct(leg, lambda i: legit_kept(items[i], texts[i]))
            out["legit_signoff"] = pct(leg, lambda i: signoff(texts[i])["has"])
            out["n_legit"] = len(leg)
        return out

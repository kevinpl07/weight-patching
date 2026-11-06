"""Suites for the sign-off warm-up (PLAN_2026-11-05_less_ai_voice.md, idea C). Cached in runs/voice/data/signoff_*.json.

- practice (300): no_robots *train*, single-turn, not Coding. For finding the direction; the only data the edit sees.
- heldout2 (300): no_robots *train* prompts 300-600 (disjoint from practice), never inspected; added after v1
  because v1's failure modes were diagnosed on `heldout`.
- heldout (300): no_robots *test*, single-turn, not Coding. Target metric: share of answers with a sign-off.
- legit (40): hand-written prompts whose requested text should itself end with a question or an offer. Guard: the
  edit must not cut these (the conditional test, like "Chinese prompts still get Chinese" in exp. 2).
- ifeval (100): runs/langmix/data/ifeval.json.  mmlu (500): runs/refusal/data/held_out.json.
- kl (100): the first 100 heldout prompts; KL(base || arm) over 256 tokens of base answers.
"""
import glob
import json
import os
import random

from common import ROOT, RUNS

DATA = RUNS / "data"

LEGIT = [  # (need, prompt): need = "question" -> the requested text should contain a question to its reader
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


def _no_robots(split):
    import pandas as pd
    f = glob.glob(os.path.expanduser(
        f"~/.cache/huggingface/hub/datasets--HuggingFaceH4--no_robots/snapshots/*/data/{split}-*.parquet"))[0]
    d = pd.read_parquet(f)
    return [dict(id=r.prompt_id, category=r.category, prompt=r.messages[0]["content"], human=r.messages[1]["content"])
            for r in d.itertuples()
            if len(r.messages) == 2 and r.messages[0]["role"] == "user" and r.category not in ("Coding", "Chat")
            and len(r.messages[0]["content"]) < 2500]


def build():
    DATA.mkdir(parents=True, exist_ok=True)
    out = {}
    rng = random.Random(0)
    tr = _no_robots("train")
    rng.shuffle(tr)
    out["practice"] = tr[:300]
    out["heldout2"] = tr[300:600]  # fresh held-out (same distribution as test), added for the v2 gate
    te = _no_robots("test")
    rng.shuffle(te)
    out["heldout"] = te[:300]
    out["legit"] = [dict(id=f"legit-{i}", need=n, prompt=p) for i, (n, p) in enumerate(LEGIT)]
    ife = json.loads((ROOT / "runs/langmix/data/ifeval.json").read_text(encoding="utf-8"))
    out["ifeval"] = [dict(id=r["id"], prompt=r["question"], doc=r["doc"]) for r in ife]
    out["mmlu"] = json.loads((ROOT / "runs/refusal/data/held_out.json").read_text(encoding="utf-8"))["mmlu"]
    out["kl"] = [dict(id=r["id"], prompt=r["prompt"]) for r in out["heldout"][:100]]
    for k, v in out.items():
        (DATA / f"signoff_{k}.json").write_text(json.dumps(v, indent=1, ensure_ascii=False), encoding="utf-8")
    return out


def load(name):
    p = DATA / f"signoff_{name}.json"
    if not p.exists():
        build()
    return json.loads(p.read_text(encoding="utf-8"))


if __name__ == "__main__":
    o = build()
    print({k: len(v) for k, v in o.items()})
    from collections import Counter
    print("practice", Counter(r["category"] for r in o["practice"]))
    print("heldout", Counter(r["category"] for r in o["heldout"]))

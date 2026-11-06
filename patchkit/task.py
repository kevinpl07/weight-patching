"""The Task interface. A task = one model + one observable failure + the data to see it.

To add an experiment you write one subclass in tasks/<name>.py and register it with @register. Everything else
(base outputs, probe, one-neuron gate, matched random edit, LoRA baseline, eval, guards, report) is shared.

Required:
    name, model                      identifiers (model = HF id)
    build_items(ctx) -> [item]       every item: id, split in {"practice","heldout"}, plus whatever score() needs
    positive(item) -> bool | None    practice label for the probe: True = failure-triggering case, False = must stay
                                     unchanged, None = ignore
    score(items, texts) -> dict      metrics on held-out; must include the two headline keys named in `headline`
Usually set:
    system_fix                       the prompt-engineering baseline (a system message)
    gate_layer, gate_targets         where the gate lives and which reply opener(s) it writes
    forced_prefix                    the opener used to self-generate LoRA targets for positive practice items
Optional overrides: messages(), probe_sites(), sft_target().
"""
import math
import random

TASKS = {}


def register(cls):
    TASKS[cls.name] = cls
    return cls


def get(name):
    if name not in TASKS:
        import importlib
        importlib.import_module(f"tasks.{name}")
    return TASKS[name]()


class Task:
    name = ""
    model = ""
    description = ""
    base_system = None          # system prompt present in every arm (e.g. an agent persona)
    system_fix = None           # extra system text for the "prompt" arm (appended to base_system)
    tools = None                # native tool schemas passed to the chat template
    max_new = 100
    gen_batch = 64
    probe_layers = ()           # candidate layers reported by `signal`
    gate_layer = -1             # MLP layer that hosts the gate neuron (negative counts from the end)
    gate_targets = ["I"]        # strings; first token of each is raised when the gate fires. "<eot>" = end of turn
    gate_gain, gate_g, gate_k = 200.0, 4.0, 4.0
    theta_margin = 0.1
    forced_prefix = None        # opener for self-generated LoRA targets (default: first gate target)
    headline = ("failure", "kept")
    headline_labels = ("failure rate", "good behaviour kept")
    content_sites_per_item = True
    n_calib = 100               # label-free generic chat replies: gate threshold/bias calibration + LoRA replay

    # ------------------------------------------------------------ data
    def build_items(self, ctx):
        raise NotImplementedError

    def messages(self, item):
        """-> (turns, system). turns: str or [(role, text), ...]."""
        return item["prompt"], self.base_system

    def render(self, tok, item, fix=False):
        from . import llm
        turns, system = self.messages(item)
        if fix and self.system_fix:
            system = (system + " " if system else "") + self.system_fix
        return llm.chat(tok, turns, system, tools=self.tools)

    def positive(self, item):
        raise NotImplementedError

    # ------------------------------------------------------------ gate
    def probe_sites(self, tok, item, base_text):
        """Where the probe reads. Default (first-token gate): the last prompt token ("decision" site, labelled by
        positive()), plus every reply position of the teacher-forced base reply of negative items ("content" sites,
        label 0) so the gate stays silent mid-reply. Returns [dict(seq, pos=[...], label, kind)]."""
        lab = self.positive(item)
        if lab is None:
            return []
        p = self.render(tok, item)
        n_p = len(tok(p, add_special_tokens=False).input_ids)
        full = p + base_text
        sites = [dict(seq=full, pos=[n_p - 1], label=int(lab), kind="decision")]
        if not lab and self.content_sites_per_item:
            n = len(tok(full, add_special_tokens=False).input_ids)
            if n - 1 > n_p:
                sites.append(dict(seq=full, pos=list(range(n_p, n - 1)), label=0, kind="content"))
        return sites

    def failed(self, item, base_text):
        """Does the base reply to a positive item actually show the failure? Used to pick which tokens the gate
        suppresses. Default: assume it does."""
        return True

    # ------------------------------------------------------------ LoRA baseline
    def sft_target(self, item, base_text, forced_text):
        """Reply the LoRA baseline is trained to give. Same information the gate gets: label + target opener.
        Positive items -> the model's own continuation after the forced opener; negatives -> its own base reply."""
        lab = self.positive(item)
        if lab is None:
            return None
        return forced_text if lab else base_text

    # ------------------------------------------------------------ scoring
    def score(self, items, texts):
        raise NotImplementedError


def subset(items, n, seed, label_fn):
    """Stratified subsample of practice items: n items total, label proportions preserved (at least 1 per label)."""
    if n is None or n >= len(items):
        return list(items)
    rng = random.Random(seed)
    groups = {}
    for x in items:
        groups.setdefault(label_fn(x), []).append(x)
    out = []
    tot = sum(len(g) for k, g in groups.items() if k is not None)
    for k, g in groups.items():
        if k is None:
            continue
        take = max(1, int(math.floor(n * len(g) / tot + 0.5)))
        out += rng.sample(g, min(take, len(g)))
    rng.shuffle(out)
    return out


def pct(idx, f):
    idx = list(idx)
    return round(100 * sum(1 for i in idx if f(i)) / max(1, len(idx)), 1)

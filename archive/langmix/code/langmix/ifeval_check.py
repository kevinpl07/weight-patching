"""lm-eval's IFEval checker (lm_eval 0.4.13), imported without lm_eval.tasks.__init__ (which needs the whole
harness). Same trick as rl/eval.py's ifeval_scorer; copied rather than imported because rl/eval.py and
langmix/eval.py share a module name.

    from ifeval_check import process
    res = process(doc, [response])   # prompt_level_strict_acc, prompt_level_loose_acc, inst_level_*_acc
"""
import importlib.util
import sys
import types
from pathlib import Path


def _load():
    pkg = Path(importlib.util.find_spec("lm_eval").submodule_search_locations[0])
    for name, path in [("lm_eval", pkg), ("lm_eval.tasks", pkg / "tasks")]:
        if name not in sys.modules or not hasattr(sys.modules[name], "__path__"):
            m = types.ModuleType(name); m.__path__ = [str(path)]; sys.modules[name] = m
    from lm_eval.tasks.ifeval import utils
    return utils.process_results


process = _load()

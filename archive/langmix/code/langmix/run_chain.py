"""The serial eval chain for experiment 2 (one process at a time, default GPU priority):

    .venv/Scripts/python langmix/run_chain.py > runs/langmix/chain.log 2>&1

1. waits until the base arm is complete (runs/langmix/eval/base/summary.json has every default suite)
2. instr  (--instr same, all suites)
3. ban    (--ban-cjk, all suites)
4. for each session as soon as runs/langmix/sessions/session_N/submitted_edit.safetensors and
   random0_same_edit.safetensors exist (session.py apply/random are run by the orchestrator):
     sN_agent       --edit submitted_edit.safetensors, all suites
     sN_rand0_same  --edit random0_same_edit.safetensors, mixing suites only (mgsm_de, mgsm_ru, mgsm_th)
5. summarize.py -> runs/langmix/summary.md (also rewritten after every arm)

Each eval is a separate eval.py subprocess (resumable), so re-running the chain skips finished work. Sessions are
polled every 2 minutes, for at most --max-wait-h hours after the last finished arm.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
RUNS = ROOT / "runs" / "langmix"
SESS = RUNS / "sessions"
ALL = ["mgsm_de", "mgsm_ru", "mgsm_th", "mgsm_en", "mgsm_zh", "ifeval", "chat"]
MIX = ["mgsm_de", "mgsm_ru", "mgsm_th"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def complete(arm, suites):
    p = RUNS / "eval" / arm / "summary.json"
    if not p.exists():
        return False
    s = json.loads(p.read_text())
    return all(k in s and s[k].get("limit") is None for k in suites)


def summarize():
    out = subprocess.run([PY, str(HERE / "summarize.py")], capture_output=True, text=True, encoding="utf-8",
                         cwd=ROOT)
    (RUNS / "summary.md").write_text(out.stdout + out.stderr, encoding="utf-8")


def evaluate(arm, args, suites):
    if complete(arm, suites):
        log(f"{arm}: already complete")
        return
    log(f"{arm}: start")
    t = time.time()
    rc = subprocess.call([PY, str(HERE / "eval.py"), "--arm", arm, "--suites", ",".join(suites)] + args, cwd=ROOT)
    log(f"{arm}: exit {rc} after {(time.time() - t) / 60:.0f} min")
    if rc != 0:
        sys.exit(f"eval failed for {arm}")
    summarize()


def main():
    max_wait_h = float(sys.argv[sys.argv.index("--max-wait-h") + 1]) if "--max-wait-h" in sys.argv else 24.0
    while not complete("base", ALL):
        log("waiting for the base arm")
        time.sleep(120)
    summarize()
    evaluate("instr", ["--instr", "same"], ALL)
    evaluate("ban", ["--ban-cjk"], ALL)
    done, last = set(), time.time()
    while len(done) < 3:
        ready = [k for k in (1, 2, 3) if k not in done
                 and (SESS / f"session_{k}" / "submitted_edit.safetensors").exists()
                 and (SESS / f"session_{k}" / "random0_same_edit.safetensors").exists()]
        if not ready:
            if time.time() - last > max_wait_h * 3600:
                log("giving up waiting for sessions")
                break
            time.sleep(120)
            continue
        k = ready[0]
        d = (SESS / f"session_{k}").relative_to(ROOT).as_posix()
        evaluate(f"s{k}_agent", ["--edit", f"{d}/submitted_edit.safetensors"], ALL)
        evaluate(f"s{k}_rand0_same", ["--edit", f"{d}/random0_same_edit.safetensors"], MIX)
        done.add(k)
        last = time.time()
    summarize()
    log("chain finished; table in runs/langmix/summary.md")


if __name__ == "__main__":
    main()

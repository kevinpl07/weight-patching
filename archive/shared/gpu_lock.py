"""Exclusive GPU access for every job in this repo (one RTX 4070, 12 GB; jobs must never overlap).

Every process that touches CUDA takes this lock first and releases it when done. Hold it for short spans
(one tool call, one eval chunk, one smoke test), not for a whole hour, so other jobs can interleave.

In Python:
    from gpu_lock import gpu
    with gpu("langmix check batch 3"):      # blocks until the GPU is free
        model.cuda(); ...; model.cpu(); torch.cuda.empty_cache()

Long jobs keep the model in CPU RAM between chunks and move it to the GPU inside the lock; moving a
1-2B model takes a second or two. Always free GPU memory before leaving the `with` block.

From the shell (holds the lock for the whole command):
    .venv/Scripts/python gpu_lock.py -- .venv/Scripts/python some_smoke_test.py
    .venv/Scripts/python gpu_lock.py --status          # who holds it, since when

Waits print a line every 60 s. The holder is recorded in runs/.gpu_holder.json.
"""
import contextlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from filelock import FileLock, Timeout

ROOT = Path(__file__).resolve().parent
LOCK = ROOT / "runs" / ".gpu.lock"
HOLDER = ROOT / "runs" / ".gpu_holder.json"


WAITERS = ROOT / "runs" / ".gpu_wait"
STALE_S = 30


def _priority(label, priority):
    """Agent-session tool calls (label contains 'agent', or AGENT_SESSION/AGENT_WORK set) get priority 10, so an
    interactive agent never waits behind a queue of bulk eval chunks. Everything else is 0. GPU_PRIORITY overrides."""
    if priority is not None:
        return priority
    if "GPU_PRIORITY" in os.environ:
        return int(os.environ["GPU_PRIORITY"])
    agentish = "agent" in label.lower() and "eval" not in label.lower()
    return 10 if agentish or os.environ.get("AGENT_SESSION") or os.environ.get("AGENT_WORK") else 0


def _higher_waiting(me, prio):
    now = time.time()
    for f in WAITERS.glob("*.json"):
        if f == me:
            continue
        try:
            if now - f.stat().st_mtime < STALE_S and json.loads(f.read_text()).get("prio", 0) > prio:
                return True
        except (OSError, ValueError):
            pass
    return False


@contextlib.contextmanager
def gpu(label="job", timeout=None, priority=None):
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    WAITERS.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(LOCK))
    prio = _priority(label, priority)
    me = WAITERS / f"{os.getpid()}_{time.time_ns()}.json"
    t0 = last_msg = time.time()
    try:
        while True:
            me.write_text(json.dumps(dict(prio=prio, label=label)))  # refreshes mtime (heartbeat)
            if not _higher_waiting(me, prio):
                try:
                    lock.acquire(timeout=5)
                    break
                except Timeout:
                    pass
            else:
                time.sleep(5)
            waited = time.time() - t0
            if time.time() - last_msg >= 60:
                last_msg = time.time()
                holder = HOLDER.read_text() if HOLDER.exists() else "?"
                print(f"[gpu_lock] waiting {waited/60:.0f} min for GPU (prio {prio}); holder: {holder}", flush=True)
            if timeout is not None and waited > timeout:
                raise Timeout(str(LOCK))
    finally:
        try:
            me.unlink()
        except OSError:
            pass
    HOLDER.write_text(json.dumps(dict(label=label, pid=os.getpid(), prio=prio, since=time.strftime("%H:%M:%S"))))
    try:
        yield
    finally:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except Exception:
            pass
        HOLDER.write_text(json.dumps(dict(label=None, released=time.strftime("%H:%M:%S"))))
        lock.release()


def main():
    if sys.argv[1:2] == ["--status"]:
        print(HOLDER.read_text() if HOLDER.exists() else "never used")
        return
    if "--" not in sys.argv:
        sys.exit("usage: python gpu_lock.py -- <command...>   |   python gpu_lock.py --status")
    cmd = sys.argv[sys.argv.index("--") + 1:]
    with gpu(" ".join(cmd)[:200]):
        rc = subprocess.call(cmd)
    sys.exit(rc)


if __name__ == "__main__":
    main()

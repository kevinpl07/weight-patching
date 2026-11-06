"""Transfer-test table for runs/whisper_turbo/eval/<arm>/summary.json (no GPU). Run from the repo root."""
import json
from pathlib import Path

ARMS = ["turbo_base", "turbo_recipe", "turbo_vad", "turbo_recipe_vad", "turbo_random_count"]
E = Path("runs/whisper_turbo/eval")
rows = ["| arm | halluc % | no-tag % | sil | noise | env | music | clean WER (drop %) | bg WER (drop %) | other WER (drop %) "
        "| padded ins/word % | padded any-ins % | edit entries |", "|" + "---|" * 13]
for arm in ARMS:
    p = E / arm / "summary.json"
    if not p.exists():
        rows.append(f"| {arm} | not run |" + " |" * 11)
        continue
    s = json.loads(p.read_text())
    su, cfg = s["suites"], s["config"]
    h = su.get("nonspeech", {}).get("hallucination", {})
    pct = lambda k: f"{100 * h[k]['rate']:.1f}" if k in h else "–"
    def w(name):
        x = su.get(name, {}).get("words")
        return f"{x['wer']:.2f} ({100 * x['dropped']:.1f})" if x else "–"
    pw = su.get("padded", {}).get("words")
    rows.append(f"| {arm} | {pct('overall')} | {100 * h['overall']['rate_notag']:.1f} | {pct('silence')} | {pct('noise')} "
                f"| {pct('environmental')} | {pct('music')} | {w('wer')} | {w('bg')} | {w('other')} "
                f"| {pw['ins_rate']:.2f} | {100 * pw['any_ins']:.1f} | {cfg.get('edit_entries_changed', 0)} |"
                if h and pw else f"| {arm} | partial |" + " |" * 11)
print("\n".join(rows))

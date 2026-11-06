"""Smoke test for agent_tools.py in a throwaway session folder (runs/whisper/smoke_session; ~3 GPU min).
  python whisper_halluc/smoke_agent_tools.py [--metrics]"""
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SESS = ROOT / "runs" / "whisper" / "smoke_session"
shutil.rmtree(SESS, ignore_errors=True)
(SESS / "work").mkdir(parents=True)
os.environ["AGENT_SESSION"] = str(SESS)
os.environ["AGENT_PASSES"] = "8"
sys.path.insert(0, str(ROOT / "whisper_halluc"))
import agent_tools as T  # noqa: E402

T.info()
P = T.practice()
print("practice clips", len(P["ids"]))
print(T.spectrogram(["pns_sil_000", "pns_env_010", "ps_251-137823-0025", 150], SESS / "work" / "spec.png"))
out = T.run(["pns_sil_000", "pns_env_005", "ps_251-137823-0025"], edited=False, cross_attn=True, self_attn=True,
            hidden=True, encoder=True, modules=["model.decoder.layers.3.fc1", "model.decoder.layers.3.encoder_attn"],
            attn_layers=[0, 5, 11])
for r in out:
    print(r["id"], repr(r["text"][:60]), repr(r["norm_text"][:40]), "nsp", round(r["nsp"], 3),
          {k: tuple(v.shape) for k, v in r.items() if hasattr(v, "shape")},
          {k: tuple(v.shape) for k, v in r.get("modules", {}).items()})
tk = T.tokenizer()
f = T.run(["pns_sil_000", "pns_sil_000"], edited=False,
          forced=[tk.encode(" you", add_special_tokens=False), tk.encode(" .", add_special_tokens=False)])
for r in f:
    print("forced", r["seq_str"], [round(float(x), 3) for x in r["topk_logprobs"][:, 0]],
          "p(eot) after content:", float(r["topk_logprobs"][len(r["seq"]) - 2].max()))
W = T.base_weights()
name = "model.decoder.layers.5.self_attn.out_proj.weight"
w = W[name].clone()
w[:, :64] *= 0.5
T.save_edit({name: w}, note="smoke")
print("current edit", {k: tuple(v.shape) for k, v in T.current_edit().items()})
try:
    T.save_edit({"model.encoder.layers.0.fc1.weight": W["model.encoder.layers.0.fc1.weight"] + 1})
except ValueError as e:
    print("scope check ok:", e)
e = T.run(["pns_sil_000"], edited=True)
print("edited run", repr(e[0]["text"]))
if "--metrics" in sys.argv:
    m = T.metrics(edited=False)
    print({k: v for k, v in m.items() if k != "transcripts"})
T.budget()
print(open(SESS / "work" / "tool_log.jsonl").read()[-600:])

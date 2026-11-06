"""Model-agnostic copy of recipe_script.py (same fixed rules): the layer is the last decoder layer from the model
config, d_model and the " ." id come from the model/tokenizer. Adds --model and writes recipe_stats.json next to the edit.

  python whisper_halluc/recipe_script_any.py --model openai/whisper-large-v3-turbo --out runs/whisper_turbo/recipe/edit.safetensors

Original docstring (layer 11 = whisper-small's last decoder layer):
Arm 7, "it's just a script": the three blind agents' shared edit, built in closed form from practice data only.

  python whisper_halluc/recipe_script.py [--gain 4] [--topk 5] [--out runs/whisper/recipe/edit.safetensors]

Recipe (fixed rules, nothing tuned on held-out data):
  x      = output of decoder layers.11.final_layer_norm (= input of the layer-11 MLP), base greedy decoding
  d      = unit(mean x[pos 3] over practice non-speech - mean x[pos 3] over practice speech, padded, bg)
           (pos 3 = the <|notimestamps|> query that picks the first content token)
  thr    = midpoint of max speech projection (every word-predicting position of practice speech/padded/bg) and min
           non-speech projection at pos 3
  j      = least-used layer-11 unit: min over j of mean|gelu(fc1_j x + b_j)| * ||fc2[:, j]|| on all practice positions
  edit   = fc1[j] = d, b1[j] = -thr, fc2[:, j] = gain * unit((E[" ."] - mean E[competitors]) / final-LN gain)
           competitors = union of the top-k first tokens (after HF suppression) on practice non-speech, minus " ."
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402  (also puts the repo root on sys.path)
from gpu_lock import gpu  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=C.DEFAULT_MODEL)
ap.add_argument("--gain", type=float, default=4.0)
ap.add_argument("--topk", type=int, default=5)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--out", default=str(C.RUNS / "recipe" / "edit.safetensors"))
a = ap.parse_args()

# ---------------------------------------------------------------- logic (30 lines, prints included)
cap, model, proc = [], C.load_model(a.model), C.load_processor(a.model)
LY = model.config.decoder_layers - 1; P = f"model.decoder.layers.{LY}."  # last decoder layer
tok, W = C.Tok(model, proc), {k: v.detach() for k, v in C.param_dict(model).items()}
model.model.decoder.layers[LY].final_layer_norm.register_forward_hook(lambda m, i, o: cap.append(o.float().cpu()))
def run(suite):  # base greedy decode of a practice suite -> (per-clip last-layer LN outputs [pos, d], n tokens, step-0 logits)
    X, N, L0, aud = [], [], [], C.load_suite(suite)[1]
    with gpu(f"recipe_script capture {suite}"):  # one short hold per suite (<= 100 clips, well under a minute)
        C.gpu_setup(); model.cuda()  # gpu_setup = set_per_process_memory_fraction(0.85), no TF32
        for k in range(0, len(aud), a.batch):
            cap.clear(); enc = C.encode(model, C.features(proc, aud[k:k + a.batch], "cuda"))
            res = C.greedy_decode(model, tok, enc, step_hook=lambda s, lg: s == 0 and L0.append(lg.cpu()))
            X += list(torch.cat(cap, 1)); N += [len(r["tokens"]) for r in res]
        model.cpu()
    return X, N, torch.cat(L0)
NS, SP = run("practice_nonspeech"), [run(s) for s in ("practice_speech", "practice_padded", "practice_bg")]
x_ns, x_sp = torch.stack([x[3] for x in NS[0]]), torch.stack([x[3] for s in SP for x in s[0]])
d = (x_ns.mean(0) - x_sp.mean(0)) / (x_ns.mean(0) - x_sp.mean(0)).norm()
p_ns, p_sp = x_ns @ d, [torch.cat([x[3:3 + n] @ d for x, n in zip(*s[:2])]) for s in SP]  # word-predicting positions
thr = (float(torch.cat(p_sp).max()) + float(p_ns.min())) / 2
x_all = torch.cat([x[:4 + n] for s in [NS] + SP for x, n in zip(*s[:2])])
j = int((F.gelu(x_all @ W[P + "fc1.weight"].T + W[P + "fc1.bias"]).abs().mean(0) * W[P + "fc2.weight"].norm(dim=0)).argmin())
lg = NS[2].index_fill(1, torch.tensor(tok.suppress + tok.begin_suppress), -float("inf"))  # HF first-step suppression
dot, E = tok.tokenizer.encode(" .", add_special_tokens=False)[0], W["model.decoder.embed_tokens.weight"]
comp = sorted(set(lg.topk(a.topk, -1).indices.flatten().tolist()) - {dot} - set(range(tok.eot, lg.shape[1])))
v = (E[dot] - E[comp].mean(0)) / W["model.decoder.layer_norm.weight"]  # undo the final layer norm's gain
w1, b1, w2 = (W[P + k].clone() for k in ("fc1.weight", "fc1.bias", "fc2.weight"))
w1[j], b1[j], w2[:, j] = d, -thr, a.gain * v / v.norm()
C.write_edit(a.out, edit := {P + "fc1.weight": w1, P + "fc1.bias": b1, P + "fc2.weight": w2})
print(f"unit {j}; {len(comp)} competitors; threshold {thr:.2f}; gain {a.gain}; entries changed {sum(n for n, _ in C.edit_stats(edit, W).values())}; "
      f"non-speech pos3 proj min/med/max {p_ns.min():.2f}/{p_ns.median():.2f}/{p_ns.max():.2f}; speech max clean/padded/bg {[round(float(p.max()), 2) for p in p_sp]}")
import json; Path(a.out).with_name("recipe_stats.json").write_text(json.dumps(dict(model=a.model, layer=LY, unit=j, dot_id=dot, competitors=comp, threshold=thr, gain=a.gain, ns_pos3=[float(p_ns.min()), float(p_ns.median()), float(p_ns.max())], speech_max={s: float(p.max()) for s, p in zip(("practice_speech", "practice_padded", "practice_bg"), p_sp)}, speech_p99={s: float(p.quantile(0.99)) for s, p in zip(("practice_speech", "practice_padded", "practice_bg"), p_sp)}, ns_below_thr=int((p_ns <= thr).sum()), n_ns=len(p_ns)), indent=1))

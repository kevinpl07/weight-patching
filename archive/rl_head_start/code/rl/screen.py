"""Screen a candidate model for the thinking-loop experiment: speed, loop rate, accuracy at a short budget.

    python rl/screen.py --model Qwen/Qwen3-0.6B --max-new 2048 --n 64 --batch 32
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval import THINK, chat, generate, gsm8k_test, is_loop  # noqa: E402
from grpo import SYSTEM, score  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--max-new", type=int, default=2048)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--no-think", action="store_true")
    p.add_argument("--mem-frac", type=float, default=0.8)
    a = p.parse_args()
    torch.cuda.set_per_process_memory_fraction(a.mem_frac)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).cuda().eval()
    print(f"{a.model}: {sum(p.numel() for p in model.parameters())/1e6:.0f}M params, "
          f"{model.config.num_hidden_layers} layers, vocab {model.config.vocab_size}")
    probs = gsm8k_test()[:a.n]
    think = not a.no_think
    prompts = [chat(tok, q["question"], SYSTEM, think) for q in probs]
    t = time.time()
    gens = generate(model, tok, prompts, a.max_new, THINK, a.batch, 0)
    dt = time.time() - t
    toks = sum(len(i) for i, _ in gens)
    closed = correct = loops = trunc = 0
    lens_closed = []
    for q, (ids, tr) in zip(probs, gens):
        text = tok.decode(ids, skip_special_tokens=True)
        c = "</think>" in text or not think
        closed += c
        if c and think:
            lens_closed.append(len(ids))
        correct += score(text.split("</think>")[-1], q["gold"], 0)[1]
        trunc += tr
        loops += tr and is_loop(ids)
    n = len(probs)
    print(f"budget {a.max_new}: {dt:.0f}s for {n} ({toks/dt:.0f} tok/s), acc {correct/n:.1%}, "
          f"closed-think {closed/n:.1%}, truncated {trunc/n:.1%}, loops {loops/n:.1%}, "
          f"median len when closed {sorted(lens_closed)[len(lens_closed)//2] if lens_closed else '-'}, "
          f"peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")


if __name__ == "__main__":
    main()

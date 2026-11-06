"""Phase-0 probe: can Qwen3.5-0.8B generate and train on this machine, and how fast?

    python rl/probe.py --gen        # sampling speed (8 answers x up to 512 tokens, non-thinking)
    python rl/probe.py --train      # memory + time of one full-weight fwd/bwd/step on 8 x ~600 tokens
"""
import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3.5-0.8B"
PROBLEM = ("Natalia sold clips to 48 of her friends in April, and then she sold half as many clips "
           "in May. How many clips did Natalia sell altogether in April and May?")
SYSTEM = "Solve the problem step by step. Put the final numeric answer in \\boxed{}."


def kernel_report():
    import transformers.models.qwen3_5.modeling_qwen3_5 as mm
    for name in ["torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule",
                 "causal_conv1d_fn", "causal_conv1d_update"]:
        f = getattr(mm, name)
        inner = getattr(f, "__wrapped__", f)
        print(f"  {name}: {getattr(inner, '__module__', '?')}")


def load(dtype=torch.bfloat16):
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype).cuda()
    n = sum(p.numel() for p in model.parameters())
    print(f"loaded {type(model).__name__}: {n/1e6:.1f}M params, vocab {model.config.vocab_size}")
    return tok, model


def prompt_ids(tok, n):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": PROBLEM}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    enc = tok([text] * n, return_tensors="pt").to("cuda")
    return text, enc


@torch.no_grad()
def gen(args):
    tok, model = load()
    text, enc = prompt_ids(tok, args.n)
    print("prompt tail:", repr(text[-120:]))
    # warm-up
    model.generate(**enc, max_new_tokens=8, do_sample=True)
    torch.cuda.synchronize(); t = time.time()
    out = model.generate(**enc, max_new_tokens=args.max_new, do_sample=True, temperature=1.0,
                         top_p=1.0, top_k=20, pad_token_id=tok.pad_token_id)
    torch.cuda.synchronize(); dt = time.time() - t
    new = out[:, enc.input_ids.shape[1]:]
    lens = [(row != tok.pad_token_id).sum().item() for row in new]
    print(f"generated {sum(lens)} tokens in {dt:.1f}s  ({sum(lens)/dt:.0f} tok/s total, "
          f"{new.shape[1]/dt:.1f} steps/s), lengths {lens}")
    print(f"peak mem {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
    for i in range(min(2, args.n)):
        print(f"--- sample {i} ---\n{tok.decode(new[i], skip_special_tokens=True)[:800]}")


def train(args):
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from grpo import token_logprobs
    torch.cuda.set_per_process_memory_fraction(args.mem_frac)  # OOM instead of silent WDDM spill to RAM
    tok, model = load()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    if args.optim == "torchao8bit":
        from torchao.optim import AdamW8bit
        opt = AdamW8bit(model.parameters(), lr=1e-6, bf16_stochastic_round=True)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=1e-6)
    ids = torch.randint(0, 150_000, (args.n, args.seq), device="cuda")
    for step in range(3):
        torch.cuda.synchronize(); t = time.time()
        for mb in ids.split(args.micro):
            lp = token_logprobs(model, mb, torch.ones_like(mb))
            (-lp.mean() * mb.shape[0] / args.n).backward()
            del lp
        opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        print(f"step {step}: {time.time()-t:.1f}s, peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--gen", action="store_true")
    p.add_argument("--train", action="store_true")
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--max-new", type=int, default=512)
    p.add_argument("--seq", type=int, default=600)
    p.add_argument("--micro", type=int, default=2)
    p.add_argument("--optim", default="torchao8bit")
    p.add_argument("--mem-frac", type=float, default=0.95)
    a = p.parse_args()
    print("kernels:"); kernel_report()
    if a.gen: gen(a)
    if a.train: train(a)

import sys; sys.path.insert(0, "../../../whisper_halluc"); import agent_tools as T
import torch, collections
R = torch.load("work/forced_dot.pt"); tk = T.tokenizer()
c = collections.Counter(); lp=[]
for r in R:
    # seq = prompt(4) + [2411] + eot ; position 4 predicts after " ."
    top = int(r["topk_ids"][4,0]); c[tk.convert_ids_to_tokens([top])[0]] += 1
    ids = r["topk_ids"][4].tolist(); 
    lp.append(float(r["topk_logprobs"][4][ids.index(50257)]) if 50257 in ids else -99)
print(c.most_common(10)); import numpy as np; print("eot logprob after ' .' quantiles", np.quantile(lp,[0,.1,.5,.9,1]).round(2))

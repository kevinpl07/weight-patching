# All-position mlp_in / mlp_act at layers 18 and 22, on practice items (with base replies) and generic chat,
# to check that the fake-vs-known direction stays quiet away from the last prompt token, and to pick unused neurons.
import sys; sys.path.insert(0, "/home/ubuntu/wp/runs/entity/agent/s1"); import tools as T
import torch
d = torch.load("work/acts1.pt")
ids, labs = d["ids"], d["labs"]
sel = [i for i, l in zip(ids, labs) if l == 1][:8] + [i for i, l in zip(ids, labs) if l == 0][:8] + [i for i, l in zip(ids, labs) if l == 2][:8]
chat = [
 {"user": "How do I reverse a list in Python?", "reply": "You can reverse a list in Python in a few ways. The simplest is `my_list.reverse()`, which reverses it in place. If you want a new list, use slicing: `reversed_list = my_list[::-1]`. You can also use `list(reversed(my_list))`."},
 {"user": "Give me three tips for sleeping better.", "reply": "Here are three tips:\n\n1. **Keep a consistent schedule.** Go to bed and wake up at the same time every day.\n2. **Limit screens before bed.** Blue light can delay sleep.\n3. **Keep your bedroom cool and dark.** Around 18°C is ideal for most people."},
 {"user": "What is the capital of Australia?", "reply": "The capital of Australia is Canberra. Many people assume it's Sydney or Melbourne, but Canberra was chosen as a compromise between the two cities and became the capital in 1913."},
 {"user": "Write a haiku about autumn.", "reply": "Crimson leaves drifting,\nCool wind hums through empty boughs,\nThe year exhales slow."},
 {"user": "Explain photosynthesis simply.", "reply": "Photosynthesis is how plants make food. They take in sunlight, water from the soil, and carbon dioxide from the air, and turn them into sugar (glucose) for energy. Oxygen is released as a by-product."},
 {"user": "Can you help me write an email asking for a day off?", "reply": "Sure! Here's a short draft:\n\nSubject: Day Off Request\n\nHi [Manager's name],\n\nI'd like to request a day off on [date] for personal reasons. I'll make sure my tasks are covered before then.\n\nThanks,\n[Your name]"},
 {"user": "Who wrote Pride and Prejudice?", "reply": "Pride and Prejudice was written by Jane Austen and published in 1813. It's one of the most famous novels in English literature."},
 {"user": "What's 17 times 23?", "reply": "17 times 23 is 391."},
]
A = T.activations(sel + chat, [18, 22], positions="all", kinds=("mlp_in", "mlp_act"))
torch.save(dict(sel=sel, acts={k: v for k, v in A.items() if k not in ("rows", "prompt_len")}, rows=A["rows"], plen=A["prompt_len"]), "work/acts_all.pt")
print("rows", len(A["rows"]))

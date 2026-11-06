import sys; sys.path.insert(0, "../../../../pretrain"); import agent_tools as T
h = T.lookahead(steps=1500, edited=False, every=100, label="noedit baseline")
print(h)

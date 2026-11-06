import sys; sys.path.insert(0, "../../../../pretrain"); import agent_tools as T
lab = sys.argv[1]
h = T.lookahead(steps=1500, edited=True, every=100, label=lab)

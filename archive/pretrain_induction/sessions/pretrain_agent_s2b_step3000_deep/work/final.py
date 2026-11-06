import sys; sys.path.insert(0, "../../../../pretrain"); import agent_tools as T
T.save_edit({}, note="final: empty edit (no candidate beat no-edit in paired lookahead)")
print("current edit:", T.current_edit()); T.budget()

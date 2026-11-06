import sys, shutil; sys.path.insert(0, "../../../pretrain"); import agent_tools as T
sys.path.insert(0,"work"); import build as B
spec=sys.argv[1]; tag=sys.argv[2]
o=B.build(**eval(spec)); T.save_edit(o,note=f"{tag}: {spec}")
m=T.metrics(edited=True); print(tag, spec); print(m)
shutil.copy("work/edited/edit.pt", f"work/edited/{tag}.pt")

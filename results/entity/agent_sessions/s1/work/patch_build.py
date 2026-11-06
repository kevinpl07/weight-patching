s = open("work/build.py").read()
s = s.replace("    pn = T.weight(", """    import os
    if os.environ.get("WRITE") == "dontknow":   # write the model's own "I cannot know this" direction instead
        K = torch.load("work/acts_dontknow.pt"); Rk = K["acts"][f"resid@{L}"].numpy(); nu = K["n_unk"]
        v = Rk[:nu].mean(0) - Rk[nu:].mean(0); print(f"   dontknow |v| {np.linalg.norm(v):.0f}"); v /= np.linalg.norm(v)
    pn = T.weight(""", 1)
open("work/build.py", "w").write(s)

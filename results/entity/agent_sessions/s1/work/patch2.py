s = open("work/build.py").read()
s = s.replace('M = a["acts"][f"mlp_in@{L}"].numpy();', 'M = a["acts"][f"mlp_in@{L}"].numpy() if f"mlp_in@{L}" in a["acts"] else X;  # L26: no all-position data, use last-prompt mean\n   ')
open("work/build.py", "w").write(s)

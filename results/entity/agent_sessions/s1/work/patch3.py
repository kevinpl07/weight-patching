s = open("work/build.py").read()
s = s.replace("""    pn = T.weight(""", """    if os.environ.get("WRITE") == "token":   # write the output-embedding direction of a refusal-opening token
        tok = T.tokenizer(); tid = tok.encode(os.environ.get("TOKEN", "Unfortunately"), add_special_tokens=False)
        oid = tok.encode("Okay", add_special_tokens=False)[0]; print("   token ids", tid, "okay", oid)
        E = T.weight("model.language_model.embed_tokens.weight", rows=[tid[0], oid]).numpy()
        fn = T.weight("model.language_model.norm.weight").numpy() + 1.0
        v = (E[0] - 0.5 * E[1]) * fn; v /= np.linalg.norm(v)   # raise the token's logit, lower "Okay" a bit
    pn = T.weight(""", 1)
open("work/build.py", "w").write(s)

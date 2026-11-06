import sys; sys.path.insert(0, "../../../whisper_halluc"); import agent_tools as T
tk = T.tokenizer()
for s in [" .", " ...", "...", " ♪", " [", " (", " -", " you", " Thank", " Thanks", " [Music]", " (upbeat music)", " ♪♪"]:
    print(repr(s), tk.encode(s, add_special_tokens=False))

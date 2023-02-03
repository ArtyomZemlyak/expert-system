import time
import sys

import torch


model, example_texts, languages, punct, apply_te = torch.hub.load(
    repo_or_dir="snakers4/silero-models", model="silero_te"
)

text = "\n".join(open(sys.argv[1]).readlines())

st_p = time.time()
text = apply_te(text, lan="ru")
et_p = time.time() - st_p

print(text)
print("TIME (predict): ", et_p)


"""
-- CPU i7-11800H  16 T
ru-test-talk.txt
TIME (predict):  0.118
"""

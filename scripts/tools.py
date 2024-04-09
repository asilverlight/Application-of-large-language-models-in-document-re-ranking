import json
import numpy as np
from tqdm import tqdm
from transformers import LlamaTokenizer

def statistic_length():
    tok = LlamaTokenizer.from_pretrained("/fs/archive/share/jarvis/llama-2-7b-hf")
    prompt_lens, completion_lens = [], []
    with open("data/train.query-understanding.jsonl", "r", encoding="utf-8") as fr:
        for line in tqdm(fr):
            data = json.loads(line)
            prompt = data["prompt"]
            completion = data["completion"]
            tok_prompt = tok.encode(prompt)
            tok_completion = tok.encode(completion)
            prompt_lens.append(len(tok_prompt))
            completion_lens.append(len(tok_completion))
    print(np.max(np.asarray(prompt_lens)))
    print(np.mean(np.asarray(prompt_lens)))
    print(np.max(np.asarray(completion_lens)))
    print(np.mean(np.asarray(completion_lens)))

statistic_length()
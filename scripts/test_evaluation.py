import torch
import json
import numpy as np
from transformers import LlamaTokenizer, LlamaForCausalLM
import torch.nn.functional as F
from IPython import embed

model_name_or_path = "/share/LMs/models--meta-llama--Llama-2-7b-hf/snapshots/8cca527612d856d7d32bd94f8103728d614eb852"
tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
tokenizer.truncation_side = "left"

model = LlamaForCausalLM.from_pretrained(model_name_or_path)

model.cuda()
model.eval()

input_text = "whether the result is"
gold_label = "Yes"
label_list = ["Yes", "No"]

answer_list = []
for label in label_list:
    context_enc = tokenizer.encode(input_text)
    continuation_enc = tokenizer.encode(input_text + label)[len(context_enc):]
    input_texts = input_text + label
    inputs = tokenizer(input_texts, return_tensors='pt', padding="longest", max_length=2048, truncation=True, return_attention_mask=True)
    logits = model(inputs['input_ids'].to(model.device), attention_mask=inputs['attention_mask'].to(model.device))
    logits = F.log_softmax(logits[0], dim=-1).cpu()
    contlen = len(continuation_enc)
    embed()
    input()
    one_logits = one_logits[-contlen-1 : -1].unsqueeze(0)  # [1, seq, vocab]
    greedy_tokens = one_logits.argmax(dim=-1)
    cont_toks = torch.tensor(cont_toks, dtype=torch.long).unsqueeze(0)  # [1, seq]
    if len(greedy_tokens[0]) != len(cont_toks[0]):
        # 超出最大长度限制
        answer = -float('inf')
    else:
        # Obtain log-probs at the corresponding continuation token indices
        # last_token_slice = logits[:, -1, :].squeeze(0).tolist()
        one_logits = torch.gather(one_logits,  dim=2, index=cont_toks.unsqueeze(-1)).squeeze(-1)  # [1, seq]

        # Answer: (log prob, is-exact-match)
        answer = float(one_logits.sum())
    
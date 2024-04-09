import sys 
sys.path.append("..")
from task import Task
import json
import random
import re
from utils import generate_rerank_instruction, generate_dbpedia_instruction, map_shots_name, get_length, load_tokenizer, MAX_LEN
from instruct_templates import PATTERNS, TASK_DESCRIPTIONS
from tqdm import tqdm
import os
import argparse
import numpy as np

rng = np.random.default_rng(114514)

cluster = "general_retrieval"
name = "ms_marco"
# template = rng.choice(PATTERNS[(cluster + "_" + name).replace("_", "-")][8:])
# print(template)
# template = (template[0], template[1])
# index = PATTERNS[(cluster + "_" + name).replace("_", "-")].index(template)
# print(index)
# a = ['1', '2']
# print(tuple(a))
# for i in range(10):
#     print(i)
a = rng.choice([3, 5, 8, 10])
print(a)
a = random.choice([3, 5, 8, 10])
print(a)

datasets = ["ms_marco", "touche", "arguana", "trec_covid", "nfcorpus", "scidocs", "quora", "cqadupstack", "dbpedia", "fever", "climate_fever", "scifact", "nq", "fiqa", "hotpot_qa"]
print(len(datasets))
datasets_init = ["touche", "arguana", "trec_covid", "nfcorpus", "scidocs", "quora", "cqadupstack", "dbpedia", "fever", "climate_fever", "scifact", "nq", "fiqa", "hotpot_qa", "trec_news", "robust04", "signal", "bioasq"]
print(len(datasets_init))

difference_datasets = list(set(datasets) - set(datasets_init))

# 找出datasets_init中不在datasets中的元素
difference_datasets_init = list(set(datasets_init) - set(datasets))

print("datasets中不在datasets_init中的元素:", difference_datasets)
print("datasets_init中不在datasets中的元素:", difference_datasets_init)

path = "/share/yutao/yifei/reranking/eval_ranking/result/without-finetune/pointwise/metrics.jsonl"
with open(path, "r") as f:
    for line in f:
        data = json.loads(line)
        print(data["dataset_name"], "mrr:", data["mrr@10"], "ndcg:", data["ndcg@10"])
        
path = "/share/yutao/yifei/reranking/eval_ranking/result/without-finetune/pointwise/cqadupstack/metrics.jsonl"
ndcg = 0
mrr = 0
with open(path, "r") as f:
    for line in f:
        data = json.loads(line)
        ndcg += data["ndcg@10"]
        mrr += data["mrr@10"]
print("cqadupstack mrr:", mrr/10, "ndcg:", ndcg/10)
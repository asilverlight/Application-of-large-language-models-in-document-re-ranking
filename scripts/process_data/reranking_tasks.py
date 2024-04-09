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
import copy

class Reranking(Task):
    def __init__(self, name, cluster, processed_data_path, wise_type="all", window_size=10):
        self._name = name
        self._cluster = cluster
        self._processed_data_path = processed_data_path
        self._wise_type = wise_type
        self._window_size = window_size#窗口长度为3/5/10
        super().__init__(self._name, self._cluster, self._processed_data_path)
        self.print_name()
        self.rng = np.random.default_rng(42)
        if self._window_size not in [0, 3, 5, 10]:
            raise ValueError("Invalid window size")
        
    def load_raw_data(self):
        pass

    def process_raw_data(self):
        pass

    def generate_zero_shot_data(self, split="train", with_desc=True):
        all_data = []
        tokenizer = load_tokenizer()
        with open(self._processed_data_path, "r", encoding="utf-8") as fr:
            for line in fr:
                data = json.loads(line)
                all_data.append(data)
                
        if self._wise_type == "all":
            result = "/share/yutao/yifei/reranking/data/zero_shot/" + self._cluster + "_" + self._name + ".zero_shot." + split + ".jsonl"
        elif self._wise_type == "pointwise":
            result = "/share/yutao/yifei/reranking/data/zero_shot_pointwise/" + self._cluster + "_" + self._name + ".zero_shot." + split + ".jsonl"
        elif self._wise_type == "pairwise":
            result = "/share/yutao/yifei/reranking/data/zero_shot_pairwise/" + self._cluster + "_" + self._name + ".zero_shot." + split + ".jsonl"
        elif self._wise_type == "listwise" and self._window_size != 0:
            result = "/share/yutao/yifei/reranking/data/zero_shot_listwise_" + str(self._window_size) + "/" + self._cluster + "_" + self._name + ".zero_shot." + split + ".jsonl"
        elif self._wise_type == "listwise" and self._window_size == 0:
            result = "/share/yutao/yifei/reranking/data/zero_shot_listwise_mix/" + self._cluster + "_" + self._name + ".zero_shot." + split + ".jsonl"
        else:
            raise ValueError("Invalid WISE type")
        if os.path.exists(result):
            print(result + " already exists, skip.")
            return

        with open(result, "w", encoding="utf-8") as fw:
            for data in tqdm(all_data, desc=self._name + " " + "zero shot"):
                if self._wise_type == "all":
                    template = self.rng.choice(PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")])
                elif self._wise_type == "pointwise":
                    template = self.rng.choice(PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")][:4])
                elif self._wise_type == "pairwise":
                    template = self.rng.choice(PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")][4:8])
                elif self._wise_type == "listwise":
                    template = self.rng.choice(PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")][8:])
                else:
                    raise ValueError("Invalid WISE type")
                choose_yesno = [True, False]
                self.rng.shuffle(choose_yesno)
                prompt = []
                completion = []
                
                template = (template[0], template[1])
                
                index = PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")].index(template)
                description = ""
                temp_size = copy.deepcopy(self._window_size)
                if self._wise_type == "listwise" and self._window_size == 0:#随机windowsize
                    self._window_size = self.rng.choice([3, 5, 10])
                self._window_size = min(self._window_size, len(data["neg"]))
                # if len(data["neg"]) <= 1:
                #     print(data["neg"])
                #     continue
                if with_desc:
                    description = TASK_DESCRIPTIONS["retrieval"]
                if index == 0 or index == 1 or index == 6 or index == 7 or index == 10 or index == 11:
                    pr, co = generate_rerank_instruction(data, template, PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")], choose_yesno[0], window_size=self._window_size)
                    pr = re.sub(r" +", " ", pr)
                    co = re.sub(r" +", " ", co)
                    prompt.append(pr)
                    completion.append(co)
                    pr, co = generate_rerank_instruction(data, template, PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")], choose_yesno[1], window_size=self._window_size)
                    pr = re.sub(r" +", " ", pr)
                    co = re.sub(r" +", " ", co)
                    prompt.append(pr)
                    completion.append(co)
                else:
                    prompt, completion = generate_rerank_instruction(data, template, PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")], window_size=self._window_size)
                    prompt = re.sub(r" +", " ", prompt)
                    completion = re.sub(r" +", " ", completion)

                self._window_size = temp_size
                if not isinstance(prompt, list):
                    sample_len = get_length(tokenizer, (description + "\n\n" + prompt + " " + completion).strip())
                    if sample_len > MAX_LEN:
                        continue
                    fw.write(json.dumps({"prompt": description + "\n\n" + prompt, "completion": completion, "source": self._cluster + "_" + self._name}) + "\n")
                else:
                    sample_len = get_length(tokenizer, (description + "\n\n" + prompt[0] + " " + completion[0]).strip())
                    if sample_len > MAX_LEN:
                        continue
                    sample_len = get_length(tokenizer, (description + "\n\n" + prompt[1] + " " + completion[1]).strip())
                    if sample_len > MAX_LEN:
                        continue
                    fw.write(json.dumps({"prompt": (description + "\n\n" + prompt[0]).strip(), "completion": completion[0], "source": self._cluster + "_" + self._name}) + "\n")
                    fw.write(json.dumps({"prompt": (description + "\n\n" + prompt[1]).strip(), "completion": completion[1], "source": self._cluster + "_" + self._name}) + "\n")

    def generate_few_shot_data(self, shots, split="train", with_desc=True):
        
        all_data = []
        tokenizer = load_tokenizer()
        with open(self._processed_data_path, "r", encoding="utf-8") as fr:
            for line in fr:
                data = json.loads(line)
                all_data.append(data)
                
        if self._wise_type == "all":
            result = "/share/yutao/yifei/reranking/data/" + map_shots_name(shots) + "_shot/" + self._cluster + "_" + self._name + "." + map_shots_name(shots) + "_shot." + split + ".jsonl"
        elif self._wise_type == "pointwise":
            result = "/share/yutao/yifei/reranking/data/" + map_shots_name(shots) + "_shot_pointwise/" + self._cluster + "_" + self._name + "." + map_shots_name(shots) + "_shot." + split + ".jsonl"
        elif self._wise_type == "pairwise":
            result = "/share/yutao/yifei/reranking/data/" + map_shots_name(shots) + "_shot_pairwise/" + self._cluster + "_" + self._name + "." + map_shots_name(shots) + "_shot." + split + ".jsonl"
        elif self._wise_type == "listwise" and self._window_size != 0:
            result = "/share/yutao/yifei/reranking/data/" + map_shots_name(shots) + "_shot_listwise_" + str(self._window_size) + "/" + self._cluster + "_" + self._name + "." + map_shots_name(shots) + "_shot." + split + ".jsonl"
        elif self._wise_type == "listwise" and self._window_size == 0:
            result = "/share/yutao/yifei/reranking/data/" + map_shots_name(shots) + "_shot_listwise_mix/" + self._cluster + "_" + self._name + "." + map_shots_name(shots) + "_shot." + split + ".jsonl"
        else:
            raise ValueError("Invalid WISE type")
        if os.path.exists(result):
            print(result + " already exists, skip.")
            return

        with open(result, "w", encoding="utf-8") as fw:
            for data in tqdm(all_data, desc=self._name + " " + map_shots_name(shots) + " shot"):
                diverse_prompts = self.rng.choice([0, 1])

                if self._wise_type == "all":
                    template = self.rng.choice(PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")])
                elif self._wise_type == "pointwise":
                    template = self.rng.choice(PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")][:4])
                elif self._wise_type == "pairwise":
                    template = self.rng.choice(PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")][4:8])
                elif self._wise_type == "listwise":
                    template = self.rng.choice(PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")][8:])
                else:
                    raise ValueError("Invalid WISE type")
                choose_yesno = [True, False]
                self.rng.shuffle(choose_yesno)
                test_prompt = []
                test_completion = []
                
                template = (template[0], template[1])
                
                index = PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")].index(template)
                description = ""
                temp_size = self._window_size
                if self._wise_type == "listwise" and self._window_size == 0:#随机windowsize
                    self._window_size = self.rng.choice([3, 5, 10])
                self._window_size = min(self._window_size, len(data["neg"]))
                # print(temp_size, self._window_size)
                if with_desc:
                    description = TASK_DESCRIPTIONS["retrieval"]
                if index == 0 or index == 1 or index == 6 or index == 7 or index == 10 or index == 11:
                    pr, co = generate_rerank_instruction(data, template, PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")], choose_yesno[0], window_size=self._window_size)
                    pr = re.sub(r" +", " ", pr)
                    co = re.sub(r" +", " ", co)
                    test_prompt.append(pr)
                    test_completion.append(co)
                    pr, co = generate_rerank_instruction(data, template, PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")], choose_yesno[1], window_size=self._window_size)
                    pr = re.sub(r" +", " ", pr)
                    co = re.sub(r" +", " ", co)
                    test_prompt.append(pr)
                    test_completion.append(co)
                else:
                    test_prompt, test_completion = generate_rerank_instruction(data, template, PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")], window_size=self._window_size)
                    test_prompt = re.sub(r" +", " ", test_prompt)
                    test_completion = re.sub(r" +", " ", test_completion)
                
                few_shots_data = []
                while len(few_shots_data) < shots:
                    few_shot_data = self.rng.choice(all_data)
                    if few_shot_data["query_id"] != data["query_id"]:
                        few_shots_data.append(few_shot_data)
                if diverse_prompts == 0:
                    examples = []
                    for few_shot_data in few_shots_data:
                        if index == 0 or index == 1 or index == 6 or index == 7 or index == 10 or index == 11:
                            yesno = self.rng.choice([True, False])
                            prompt, completion = generate_rerank_instruction(few_shot_data, template, PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")], yesno, window_size=self._window_size)
                        else:
                            prompt, completion = generate_rerank_instruction(few_shot_data, template, PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")], window_size=self._window_size)
                        example = prompt.strip() + " " + completion
                        examples.append(example)
                else:
                    examples = []
                    for few_shot_data in few_shots_data:
                        index_e = self.rng.choice([index - 1, index]) if index % 2 == 1 else self.rng.choice([index, index + 1])
                        template = PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")][index_e]
                        if index_e == 0 or index_e == 1 or index_e == 6 or index_e == 7 or index_e == 10 or index_e == 11:
                            yesno = self.rng.choice([True, False])
                            prompt, completion = generate_rerank_instruction(few_shot_data, template, PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")], yesno, window_size=self._window_size)
                        else:
                            prompt, completion = generate_rerank_instruction(few_shot_data, template, PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")], window_size=self._window_size)
                        example = prompt.strip() + " " + completion
                        examples.append(example)
                examples = "\n\n".join(examples)
                examples = re.sub(r" +", " ", examples)
                
                self._window_size = temp_size

                if not isinstance(test_prompt, list):
                    sample_len = get_length(tokenizer, (description + "\n\n" + examples + "\n\n" + test_prompt + " " + test_completion).strip())
                    if sample_len > MAX_LEN:
                        continue
                    fw.write(json.dumps({"prompt": (description + "\n\n" + examples + "\n\n" + test_prompt).strip(), "completion": completion, "source": self._cluster + "_" + self._name}) + "\n")
                else:
                    sample_len = get_length(tokenizer, (description + "\n\n" + examples + "\n\n" + test_prompt[0] + " " + test_completion[0]).strip())
                    if sample_len > MAX_LEN:
                        continue
                    sample_len = get_length(tokenizer, (description + "\n\n" + examples + "\n\n" + test_prompt[1] + " " + test_completion[1]).strip())
                    if sample_len > MAX_LEN:
                        continue
                    fw.write(json.dumps({"prompt": (description + "\n\n" + examples + "\n\n" + test_prompt[0]).strip(), "completion": test_completion[0], "source": self._cluster + "_" + self._name}) + "\n")
                    fw.write(json.dumps({"prompt": (description + "\n\n" + examples + "\n\n" + test_prompt[1]).strip(), "completion": test_completion[1], "source": self._cluster + "_" + self._name}) + "\n")

class RerankingDbpedia(Task):
    def __init__(self, wises_type="all"):
        self._name = "dbpedia"
        self._cluster = "entity_retrieval"
        self._processed_data_path = "../rerank_task_data/dbpedia.jsonl"
        self._wise_type = wises_type
        super().__init__(self._name, self._cluster, self._processed_data_path)
        self.print_name()
        self.rng = np.random.default_rng(42)
    
    def load_raw_data(self):
        pass

    def process_raw_data(self):
        pass

    def generate_zero_shot_data(self, split="train", with_desc=True):
        
        all_data = []
        tokenizer = load_tokenizer()
        with open(self._processed_data_path, "r", encoding="utf-8") as fr:
            for line in fr:
                data = json.loads(line)
                all_data.append(data)
        
        if self._wise_type == "all":
            result = "/share/yutao/yifei/reranking/data/zero_shot/" + self._cluster + "_" + self._name + ".zero_shot." + split + ".jsonl"
        elif self._wise_type == "pointwise":
            result = "/share/yutao/yifei/reranking/data/zero_shot_pointwise/" + self._cluster + "_" + self._name + ".zero_shot." + split + ".jsonl"
        elif self._wise_type == "pairwise":
            result = "/share/yutao/yifei/reranking/data/zero_shot_pairwise/" + self._cluster + "_" + self._name + ".zero_shot." + split + ".jsonl"
        elif self._wise_type == "listwise":
            result = "/share/yutao/yifei/reranking/data/zero_shot_listwise_10" + "/" + self._cluster + "_" + self._name + ".zero_shot." + split + ".jsonl"
        else:
            raise ValueError("Invalid WISE type")
        if os.path.exists(result):
            print(result + " already exists, skip.")
            return

        with open(result, "w", encoding="utf-8") as fw:
            for data in tqdm(all_data, desc=self._name + " " + "zero shot"):
                if self._wise_type == "all":
                    template = self.rng.choice(PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")])
                elif self._wise_type == "pointwise":
                    template = self.rng.choice(PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")][:4])
                elif self._wise_type == "pairwise":
                    template = self.rng.choice(PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")][4:8])
                elif self._wise_type == "listwise":
                    template = self.rng.choice(PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")][8:])
                else:
                    raise ValueError("Invalid WISE type")
                choose_yesno = [True, False]
                self.rng.shuffle(choose_yesno)
                prompt = []
                completion = []
                
                template = (template[0], template[1])
                
                index = PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")].index(template)
                description = ""
                if with_desc:
                    description = TASK_DESCRIPTIONS["retrieval"]
                if index == 0 or index == 1 or index == 6 or index == 7 or index == 10 or index == 11:
                    pr, co = generate_dbpedia_instruction(data, template, PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")], choose_yesno[0])
                    pr = re.sub(r" +", " ", pr)
                    co = re.sub(r" +", " ", co)
                    prompt.append(pr)
                    completion.append(co)
                    pr, co = generate_dbpedia_instruction(data, template, PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")], choose_yesno[1])
                    pr = re.sub(r" +", " ", pr)
                    co = re.sub(r" +", " ", co)
                    prompt.append(pr)
                    completion.append(co)
                else:
                    prompt, completion = generate_dbpedia_instruction(data, template, PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")])
                    prompt = re.sub(r" +", " ", prompt)
                    completion = re.sub(r" +", " ", completion)

                if not isinstance(prompt, list):
                    sample_len = get_length(tokenizer, (description + "\n\n" + prompt + " " + completion).strip())
                    if sample_len > MAX_LEN:
                        continue
                    fw.write(json.dumps({"prompt": description + "\n\n" + prompt, "completion": completion, "source": self._cluster + "_" + self._name}) + "\n")
                else:
                    sample_len = get_length(tokenizer, (description + "\n\n" + prompt[0] + " " + completion[0]).strip())
                    if sample_len > MAX_LEN:
                        continue
                    sample_len = get_length(tokenizer, (description + "\n\n" + prompt[1] + " " + completion[1]).strip())
                    if sample_len > MAX_LEN:
                        continue
                    fw.write(json.dumps({"prompt": (description + "\n\n" + prompt[0]).strip(), "completion": completion[0], "source": self._cluster + "_" + self._name}) + "\n")
                    fw.write(json.dumps({"prompt": (description + "\n\n" + prompt[1]).strip(), "completion": completion[1], "source": self._cluster + "_" + self._name}) + "\n")

    def generate_few_shot_data(self, shots, split="train", with_desc=True):
        
        all_data = []
        tokenizer = load_tokenizer()
        with open(self._processed_data_path, "r", encoding="utf-8") as fr:
            for line in fr:
                data = json.loads(line)
                all_data.append(data)
        
        if self._wise_type == "all":        
            result = "/share/yutao/yifei/reranking/data/" + map_shots_name(shots) + "_shot/" + self._cluster + "_" + self._name + "." + map_shots_name(shots) + "_shot." + split + ".jsonl"
        elif self._wise_type == "pointwise":
            result = "/share/yutao/yifei/reranking/data/" + map_shots_name(shots) + "_shot_pointwise/" + self._cluster + "_" + self._name + "." + map_shots_name(shots) + "_shot." + split + ".jsonl"
        elif self._wise_type == "pairwise":
            result = "/share/yutao/yifei/reranking/data/" + map_shots_name(shots) + "_shot_pairwise/" + self._cluster + "_" + self._name + "." + map_shots_name(shots) + "_shot." + split + ".jsonl"
        elif self._wise_type == "listwise":
            result = "/share/yutao/yifei/reranking/data/" + map_shots_name(shots) + "_shot_listwise_10/" + self._cluster + "_" + self._name + "." + map_shots_name(shots) + "_shot." + split + ".jsonl"
        else:
            raise ValueError("Invalid WISE type")
        if os.path.exists(result):
            print(result + " already exists, skip.")
            return

        with open(result, "w", encoding="utf-8") as fw:
            for data in tqdm(all_data, desc=self._name + " " + map_shots_name(shots) + " shot"):
                diverse_prompts = self.rng.choice([0, 1])

                if self._wise_type == "all":
                    template = self.rng.choice(PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")])
                elif self._wise_type == "pointwise":
                    template = self.rng.choice(PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")][:4])
                elif self._wise_type == "pairwise":
                    template = self.rng.choice(PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")][4:8])
                elif self._wise_type == "listwise":
                    template = self.rng.choice(PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")][8:])
                else:
                    raise ValueError("Invalid WISE type")
                choose_yesno = [True, False]
                self.rng.shuffle(choose_yesno)
                test_prompt = []
                test_completion = []
                
                template = (template[0], template[1])
                
                index = PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")].index(template)
                description = ""
                if with_desc:
                    description = TASK_DESCRIPTIONS["retrieval"]
                if index == 0 or index == 1 or index == 6 or index == 7 or index == 10 or index == 11:
                    pr, co = generate_dbpedia_instruction(data, template, PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")], choose_yesno[0])
                    pr = re.sub(r" +", " ", pr)
                    co = re.sub(r" +", " ", co)
                    test_prompt.append(pr)
                    test_completion.append(co)
                    pr, co = generate_dbpedia_instruction(data, template, PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")], choose_yesno[1])
                    pr = re.sub(r" +", " ", pr)
                    co = re.sub(r" +", " ", co)
                    test_prompt.append(pr)
                    test_completion.append(co)
                else:
                    test_prompt, test_completion = generate_dbpedia_instruction(data, template, PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")])
                    test_prompt = re.sub(r" +", " ", test_prompt)
                    test_completion = re.sub(r" +", " ", test_completion)

                few_shots_data = []
                while len(few_shots_data) < shots:
                    few_shot_data = self.rng.choice(all_data)
                    if few_shot_data["query_id"] != data["query_id"]:
                        few_shots_data.append(few_shot_data)
                if diverse_prompts == 0:
                    examples = []
                    for few_shot_data in few_shots_data:
                        if index == 0 or index == 1 or index == 6 or index == 7 or index == 10 or index == 11:
                            yesno = self.rng.choice([True, False])
                            prompt, completion = generate_dbpedia_instruction(few_shot_data, template, PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")], yesno)
                        else:
                            prompt, completion = generate_dbpedia_instruction(few_shot_data, template, PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")])
                        example = prompt.strip() + " " + completion
                        examples.append(example)
                else:
                    examples = []
                    for few_shot_data in few_shots_data:
                        index_e = self.rng.choice([index - 1, index]) if index % 2 == 1 else self.rng.choice([index, index + 1])
                        template = PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")][index_e]
                        if index_e == 0 or index_e == 1 or index_e == 6 or index_e == 7 or index_e == 10 or index_e == 11:
                            yesno = self.rng.choice([True, False])
                            prompt, completion = generate_dbpedia_instruction(few_shot_data, template, PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")], yesno)
                        else:
                            prompt, completion = generate_dbpedia_instruction(few_shot_data, template, PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")])
                        example = prompt.strip() + " " + completion
                        examples.append(example)
                examples = "\n\n".join(examples)
                examples = re.sub(r" +", " ", examples)

                if not isinstance(test_prompt, list):
                    sample_len = get_length(tokenizer, (description + "\n\n" + examples + "\n\n" + test_prompt + " " + test_completion).strip())
                    if sample_len > MAX_LEN:
                        continue
                    fw.write(json.dumps({"prompt": (description + "\n\n" + examples + "\n\n" + test_prompt).strip(), "completion": completion, "source": self._cluster + "_" + self._name}) + "\n")
                else:
                    sample_len = get_length(tokenizer, (description + "\n\n" + examples + "\n\n" + test_prompt[0] + " " + test_completion[0]).strip())
                    if sample_len > MAX_LEN:
                        continue
                    sample_len = get_length(tokenizer, (description + "\n\n" + examples + "\n\n" + test_prompt[1] + " " + test_completion[1]).strip())
                    if sample_len > MAX_LEN:
                        continue
                    fw.write(json.dumps({"prompt": (description + "\n\n" + examples + "\n\n" + test_prompt[0]).strip(), "completion": test_completion[0], "source": self._cluster + "_" + self._name}) + "\n")
                    fw.write(json.dumps({"prompt": (description + "\n\n" + examples + "\n\n" + test_prompt[1]).strip(), "completion": test_completion[1], "source": self._cluster + "_" + self._name}) + "\n")

class RerankingCqadupstack(Task):
    def __init__(self, wise_type="all", window_size=10):
        self._name = "cqadupstack"
        self._cluster = "duplicate_question_retrieval"
        self._processed_data_path = [
            "../rerank_task_data/cqadupstack_android.jsonl",
            "../rerank_task_data/cqadupstack_english.jsonl",
            "../rerank_task_data/cqadupstack_gaming.jsonl",
            "../rerank_task_data/cqadupstack_gis.jsonl",
            "../rerank_task_data/cqadupstack_mathematica.jsonl",
            "../rerank_task_data/cqadupstack_physics.jsonl",
            "../rerank_task_data/cqadupstack_programmers.jsonl",
            "../rerank_task_data/cqadupstack_stats.jsonl",
            "../rerank_task_data/cqadupstack_tex.jsonl",
            "../rerank_task_data/cqadupstack_unix.jsonl",
            "../rerank_task_data/cqadupstack_webmasters.jsonl",
            "../rerank_task_data/cqadupstack_wordpress.jsonl",
        ]
        self._wise_type = wise_type
        self._window_size = window_size
        super().__init__(self._name, self._cluster, self._processed_data_path)
        self.print_name()
        self.rng = np.random.default_rng(42)
        if self._window_size not in [0, 3, 5, 10]:
            raise ValueError("Invalid window size")
    
    def load_raw_data(self):
        pass

    def process_raw_data(self):
        pass

    def generate_zero_shot_data(self, split="train", with_desc=True):
        
        tokenizer = load_tokenizer()
        if self._wise_type == "all":
            result = "/share/yutao/yifei/reranking/data/zero_shot/" + self._cluster + "_" + self._name + ".zero_shot." + split + ".jsonl"
        elif self._wise_type == "pointwise":
            result = "/share/yutao/yifei/reranking/data/zero_shot_pointwise/" + self._cluster + "_" + self._name + ".zero_shot." + split + ".jsonl"
        elif self._wise_type == "pairwise":
            result = "/share/yutao/yifei/reranking/data/zero_shot_pairwise/" + self._cluster + "_" + self._name + ".zero_shot." + split + ".jsonl"
        elif self._wise_type == "listwise" and self._window_size != 0:
            result = "/share/yutao/yifei/reranking/data/zero_shot_listwise_" + str(self._window_size) + "/" + self._cluster + "_" + self._name + ".zero_shot." + split + ".jsonl"
        elif self._wise_type == "listwise" and self._window_size == 0:
            result = "/share/yutao/yifei/reranking/data/zero_shot_listwise_mix/" + self._cluster + "_" + self._name + ".zero_shot." + split + ".jsonl"
        else:
            raise ValueError("Invalid WISE type")
        if os.path.exists(result):
            print(result + " already exists, skip.")
            return
        fw = open(result, "w", encoding="utf-8")
        for data_path in self._processed_data_path:
            subset = data_path.split("/")[-1].split(".")[0].split("_")[1]

            all_data = []
            with open(data_path, "r", encoding="utf-8") as fr:
                for line in fr:
                    data = json.loads(line)
                    all_data.append(data)
            
            for data in tqdm(all_data, desc=self._name + " " + "zero shot"):
                if self._wise_type == "all":
                    template = self.rng.choice(PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")])
                elif self._wise_type == "pointwise":
                    template = self.rng.choice(PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")][:4])
                elif self._wise_type == "pairwise":
                    template = self.rng.choice(PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")][4:8])
                elif self._wise_type == "listwise":
                    template = self.rng.choice(PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")][8:])
                else:
                    raise ValueError("Invalid WISE type")
                choose_yesno = [True, False]
                self.rng.shuffle(choose_yesno)
                prompt = []
                completion = []
                
                template = (template[0], template[1])
                    
                index = PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")].index(template)
                description = ""
                temp_size = copy.deepcopy(self._window_size)
                if self._wise_type == "listwise" and self._window_size == 0:#随机windowsize
                    self._window_size = self.rng.choice([3, 5, 10])
                self._window_size = min(self._window_size, len(data["neg"]))
                if with_desc:
                    description = TASK_DESCRIPTIONS["retrieval"]
                if index == 0 or index == 1 or index == 6 or index == 7 or index == 10 or index == 11:
                    pr, co = generate_rerank_instruction(data, template, PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")], choose_yesno[0], window_size=self._window_size)
                    pr = re.sub(r" +", " ", pr)
                    co = re.sub(r" +", " ", co)
                    prompt.append(pr)
                    completion.append(co)
                    pr, co = generate_rerank_instruction(data, template, PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")], choose_yesno[1], window_size=self._window_size)
                    pr = re.sub(r" +", " ", pr)
                    co = re.sub(r" +", " ", co)
                    prompt.append(pr)
                    completion.append(co)
                else:
                    prompt, completion = generate_rerank_instruction(data, template, PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")], window_size=self._window_size)
                    prompt = re.sub(r" +", " ", prompt)
                    completion = re.sub(r" +", " ", completion)
                    
                self._window_size = temp_size

                if not isinstance(prompt, list):
                    sample_len = get_length(tokenizer, (description + "\n\n" + prompt + " " + completion).strip())
                    if sample_len > MAX_LEN:
                        continue
                    fw.write(json.dumps({"prompt": description + "\n\n" + prompt, "completion": completion, "source": self._cluster + "_" + self._name}) + "\n")
                else:
                    sample_len = get_length(tokenizer, (description + "\n\n" + prompt[0] + " " + completion[0]).strip())
                    if sample_len > MAX_LEN:
                        continue
                    sample_len = get_length(tokenizer, (description + "\n\n" + prompt[1] + " " + completion[1]).strip())
                    if sample_len > MAX_LEN:
                        continue
                    fw.write(json.dumps({"prompt": (description + "\n\n" + prompt[0]).strip(), "completion": completion[0], "source": self._cluster + "_" + self._name}) + "\n")
                    fw.write(json.dumps({"prompt": (description + "\n\n" + prompt[1]).strip(), "completion": completion[1], "source": self._cluster + "_" + self._name}) + "\n")
        fw.close()

    def generate_few_shot_data(self, shots, split="train", with_desc=True):
        
        tokenizer = load_tokenizer()
        if self._wise_type == "all":
            result = "/share/yutao/yifei/reranking/data/" + map_shots_name(shots) + "_shot/" + self._cluster + "_" + self._name + "." + map_shots_name(shots) + "_shot." + split + ".jsonl"
        elif self._wise_type == "pointwise":
            result = "/share/yutao/yifei/reranking/data/" + map_shots_name(shots) + "_shot_pointwise/" + self._cluster + "_" + self._name + "." + map_shots_name(shots) + "_shot." + split + ".jsonl"
        elif self._wise_type == "pairwise":
            result = "/share/yutao/yifei/reranking/data/" + map_shots_name(shots) + "_shot_pairwise/" + self._cluster + "_" + self._name + "." + map_shots_name(shots) + "_shot." + split + ".jsonl"
        elif self._wise_type == "listwise" and self._window_size != 0:
            result = "/share/yutao/yifei/reranking/data/" + map_shots_name(shots) + "_shot_listwise_" + str(self._window_size) + "/" + self._cluster + "_" + self._name + "." + map_shots_name(shots) + "_shot." + split + ".jsonl"
        elif self._wise_type == "listwise" and self._window_size == 0:
            result = "/share/yutao/yifei/reranking/data/" + map_shots_name(shots) + "_shot_listwise_mix/" + self._cluster + "_" + self._name + "." + map_shots_name(shots) + "_shot." + split + ".jsonl"
        else:
            raise ValueError("Invalid WISE type")
        if os.path.exists(result):
            print(result + " already exists, skip.")
            return
        fw = open(result, "a", encoding="utf-8")
        for data_path in self._processed_data_path:
            subset = data_path.split("/")[-1].split(".")[0].split("_")[1]

            all_data = []
            with open(data_path, "r", encoding="utf-8") as fr:
                for line in fr:
                    data = json.loads(line)
                    all_data.append(data)

            for data in tqdm(all_data, desc=self._name + " " + map_shots_name(shots) + " shot"):
                diverse_prompts = self.rng.choice([0, 1])

                if self._wise_type == "all":
                    template = self.rng.choice(PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")])
                elif self._wise_type == "pointwise":
                    template = self.rng.choice(PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")][:4])
                elif self._wise_type == "pairwise":
                    template = self.rng.choice(PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")][4:8])
                elif self._wise_type == "listwise":
                    template = self.rng.choice(PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")][8:])
                else:
                    raise ValueError("Invalid WISE type")
                choose_yesno = [True, False]
                self.rng.shuffle(choose_yesno)
                test_prompt = []
                test_completion = []
                
                template = (template[0], template[1])
                    
                index = PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")].index(template)
                description = ""
                temp_size = copy.deepcopy(self._window_size)
                if self._wise_type == "listwise" and self._window_size == 0:#随机windowsize
                    self._window_size = self.rng.choice([3, 5, 10])
                self._window_size = min(self._window_size, len(data["neg"]))
                if with_desc:
                    description = TASK_DESCRIPTIONS["retrieval"]
                if index == 0 or index == 1 or index == 6 or index == 7 or index == 10 or index == 11:
                    pr, co = generate_rerank_instruction(data, template, PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")], choose_yesno[0], window_size=self._window_size)
                    pr = re.sub(r" +", " ", pr)
                    co = re.sub(r" +", " ", co)
                    test_prompt.append(pr)
                    test_completion.append(co)
                    pr, co = generate_rerank_instruction(data, template, PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")], choose_yesno[1], window_size=self._window_size)
                    pr = re.sub(r" +", " ", pr)
                    co = re.sub(r" +", " ", co)
                    test_prompt.append(pr)
                    test_completion.append(co)
                else:
                    test_prompt, test_completion = generate_rerank_instruction(data, template, PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")], window_size=self._window_size)
                    test_prompt = re.sub(r" +", " ", test_prompt)
                    test_completion = re.sub(r" +", " ", test_completion)
                    

                few_shots_data = []
                while len(few_shots_data) < shots:
                    few_shot_data = self.rng.choice(all_data)
                    if few_shot_data["query_id"] != data["query_id"]:
                        few_shots_data.append(few_shot_data)
                if diverse_prompts == 0:
                    examples = []
                    for few_shot_data in few_shots_data:
                        if index == 0 or index == 1 or index == 6 or index == 7 or index == 10 or index == 11:
                            yesno = self.rng.choice([True, False])
                            prompt, completion = generate_rerank_instruction(few_shot_data, template, PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")], yesno, window_size=self._window_size)
                        else:
                            prompt, completion = generate_rerank_instruction(few_shot_data, template, PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")], window_size=self._window_size)
                        example = prompt.strip() + " " + completion
                        examples.append(example)
                else:
                    examples = []
                    for few_shot_data in few_shots_data:
                        index_e = self.rng.choice([index - 1, index]) if index % 2 == 1 else self.rng.choice([index, index + 1])
                        template = PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")][index_e]
                        if index_e == 0 or index_e == 1 or index_e == 6 or index_e == 7 or index_e == 10 or index_e == 11:
                            yesno = self.rng.choice([True, False])
                            prompt, completion = generate_rerank_instruction(few_shot_data, template, PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")], yesno, window_size=self._window_size)
                        else:
                            prompt, completion = generate_rerank_instruction(few_shot_data, template, PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")], window_size=self._window_size)
                        example = prompt.strip() + " " + completion
                        examples.append(example)
                examples = "\n\n".join(examples)
                examples = re.sub(r" +", " ", examples)
                self._window_size = temp_size

                if not isinstance(test_prompt, list):
                    sample_len = get_length(tokenizer, (description + "\n\n" + examples + "\n\n" + test_prompt + " " + test_completion).strip())
                    if sample_len > MAX_LEN:
                        continue
                    fw.write(json.dumps({"prompt": (description + "\n\n" + examples + "\n\n" + test_prompt).strip(), "completion": completion, "source": self._cluster + "_" + self._name}) + "\n")
                else:
                    sample_len = get_length(tokenizer, (description + "\n\n" + examples + "\n\n" + test_prompt[0] + " " + test_completion[0]).strip())
                    if sample_len > MAX_LEN:
                        continue
                    sample_len = get_length(tokenizer, (description + "\n\n" + examples + "\n\n" + test_prompt[1] + " " + test_completion[1]).strip())
                    if sample_len > MAX_LEN:
                        continue
                    fw.write(json.dumps({"prompt": (description + "\n\n" + examples + "\n\n" + test_prompt[0]).strip(), "completion": test_completion[0], "source": self._cluster + "_" + self._name}) + "\n")
                    fw.write(json.dumps({"prompt": (description + "\n\n" + examples + "\n\n" + test_prompt[1]).strip(), "completion": test_completion[1], "source": self._cluster + "_" + self._name}) + "\n")

        fw.close()

if __name__ == "__main__":
    if not os.path.exists("/share/yutao/yifei/reranking/data/zero_shot/"):
        os.makedirs("/share/yutao/yifei/reranking/data/zero_shot/")
    if not os.path.exists("/share/yutao/yifei/reranking/data/one_shot/"):
        os.makedirs("/share/yutao/yifei/reranking/data/one_shot/")
    if not os.path.exists("/share/yutao/yifei/reranking/data/two_shot/"):
        os.makedirs("/share/yutao/yifei/reranking/data/two_shot/")
    if not os.path.exists("/share/yutao/yifei/reranking/data/three_shot/"):
        os.makedirs("/share/yutao/yifei/reranking/data/three_shot/")
    if not os.path.exists("/share/yutao/yifei/reranking/data/four_shot/"):
        os.makedirs("/share/yutao/yifei/reranking/data/four_shot/")
    if not os.path.exists("/share/yutao/yifei/reranking/data/five_shot/"):
        os.makedirs("/share/yutao/yifei/reranking/data/five_shot/")
        
    if not os.path.exists("/share/yutao/yifei/reranking/data/zero_shot_pointwise/"):
        os.makedirs("/share/yutao/yifei/reranking/data/zero_shot_pointwise/")
    if not os.path.exists("/share/yutao/yifei/reranking/data/one_shot_pointwise/"):
        os.makedirs("/share/yutao/yifei/reranking/data/one_shot_pointwise/")
    if not os.path.exists("/share/yutao/yifei/reranking/data/two_shot_pointwise/"):
        os.makedirs("/share/yutao/yifei/reranking/data/two_shot_pointwise/")
    if not os.path.exists("/share/yutao/yifei/reranking/data/three_shot_pointwise/"):
        os.makedirs("/share/yutao/yifei/reranking/data/three_shot_pointwise/")
    if not os.path.exists("/share/yutao/yifei/reranking/data/four_shot_pointwise/"):
        os.makedirs("/share/yutao/yifei/reranking/data/four_shot_pointwise/")
    if not os.path.exists("/share/yutao/yifei/reranking/data/five_shot_pointwise/"):
        os.makedirs("/share/yutao/yifei/reranking/data/five_shot_pointwise/")
        
    if not os.path.exists("/share/yutao/yifei/reranking/data/zero_shot_pairwise/"):
        os.makedirs("/share/yutao/yifei/reranking/data/zero_shot_pairwise/")
    if not os.path.exists("/share/yutao/yifei/reranking/data/one_shot_pairwise/"):
        os.makedirs("/share/yutao/yifei/reranking/data/one_shot_pairwise/")
    if not os.path.exists("/share/yutao/yifei/reranking/data/two_shot_pairwise/"):
        os.makedirs("/share/yutao/yifei/reranking/data/two_shot_pairwise/")
    if not os.path.exists("/share/yutao/yifei/reranking/data/three_shot_pairwise/"):
        os.makedirs("/share/yutao/yifei/reranking/data/three_shot_pairwise/")
    if not os.path.exists("/share/yutao/yifei/reranking/data/four_shot_pairwise/"):
        os.makedirs("/share/yutao/yifei/reranking/data/four_shot_pairwise/")
    if not os.path.exists("/share/yutao/yifei/reranking/data/five_shot_pairwise/"):
        os.makedirs("/share/yutao/yifei/reranking/data/five_shot_pairwise/")
        
    if not os.path.exists("/share/yutao/yifei/reranking/data/zero_shot_listwise_3/"):
        os.makedirs("/share/yutao/yifei/reranking/data/zero_shot_listwise_3/")
    if not os.path.exists("/share/yutao/yifei/reranking/data/one_shot_listwise_3/"):
        os.makedirs("/share/yutao/yifei/reranking/data/one_shot_listwise_3/")
    if not os.path.exists("/share/yutao/yifei/reranking/data/two_shot_listwise_3/"):
        os.makedirs("/share/yutao/yifei/reranking/data/two_shot_listwise_3/")
    if not os.path.exists("/share/yutao/yifei/reranking/data/three_shot_listwise_3/"):
        os.makedirs("/share/yutao/yifei/reranking/data/three_shot_listwise_3/")
    if not os.path.exists("/share/yutao/yifei/reranking/data/four_shot_listwise_3/"):
        os.makedirs("/share/yutao/yifei/reranking/data/four_shot_listwise_3/")
    if not os.path.exists("/share/yutao/yifei/reranking/data/five_shot_listwise_3/"):
        os.makedirs("/share/yutao/yifei/reranking/data/five_shot_listwise_3/")
        
    if not os.path.exists("/share/yutao/yifei/reranking/data/zero_shot_listwise_5/"):
        os.makedirs("/share/yutao/yifei/reranking/data/zero_shot_listwise_5/")
    if not os.path.exists("/share/yutao/yifei/reranking/data/one_shot_listwise_5/"):
        os.makedirs("/share/yutao/yifei/reranking/data/one_shot_listwise_5/")
    if not os.path.exists("/share/yutao/yifei/reranking/data/two_shot_listwise_5/"):
        os.makedirs("/share/yutao/yifei/reranking/data/two_shot_listwise_5/")
    if not os.path.exists("/share/yutao/yifei/reranking/data/three_shot_listwise_5/"):
        os.makedirs("/share/yutao/yifei/reranking/data/three_shot_listwise_5/")
    if not os.path.exists("/share/yutao/yifei/reranking/data/four_shot_listwise_5/"):
        os.makedirs("/share/yutao/yifei/reranking/data/four_shot_listwise_5/")
    if not os.path.exists("/share/yutao/yifei/reranking/data/five_shot_listwise_5/"):
        os.makedirs("/share/yutao/yifei/reranking/data/five_shot_listwise_5/")
        
    if not os.path.exists("/share/yutao/yifei/reranking/data/zero_shot_listwise_10/"):
        os.makedirs("/share/yutao/yifei/reranking/data/zero_shot_listwise_10/")
    if not os.path.exists("/share/yutao/yifei/reranking/data/one_shot_listwise_10/"):
        os.makedirs("/share/yutao/yifei/reranking/data/one_shot_listwise_10/")
    if not os.path.exists("/share/yutao/yifei/reranking/data/two_shot_listwise_10/"):
        os.makedirs("/share/yutao/yifei/reranking/data/two_shot_listwise_10/")
    if not os.path.exists("/share/yutao/yifei/reranking/data/three_shot_listwise_10/"):
        os.makedirs("/share/yutao/yifei/reranking/data/three_shot_listwise_10/")
    if not os.path.exists("/share/yutao/yifei/reranking/data/four_shot_listwise_10/"):
        os.makedirs("/share/yutao/yifei/reranking/data/four_shot_listwise_10/")
    if not os.path.exists("/share/yutao/yifei/reranking/data/five_shot_listwise_10/"):
        os.makedirs("/share/yutao/yifei/reranking/data/five_shot_listwise_10/")
        
    if not os.path.exists("/share/yutao/yifei/reranking/data/zero_shot_listwise_mix/"):
        os.makedirs("/share/yutao/yifei/reranking/data/zero_shot_listwise_mix/")
    if not os.path.exists("/share/yutao/yifei/reranking/data/one_shot_listwise_mix/"):
        os.makedirs("/share/yutao/yifei/reranking/data/one_shot_listwise_mix/")
    if not os.path.exists("/share/yutao/yifei/reranking/data/two_shot_listwise_mix/"):
        os.makedirs("/share/yutao/yifei/reranking/data/two_shot_listwise_mix/")
    if not os.path.exists("/share/yutao/yifei/reranking/data/three_shot_listwise_mix/"):
        os.makedirs("/share/yutao/yifei/reranking/data/three_shot_listwise_mix/")
    if not os.path.exists("/share/yutao/yifei/reranking/data/four_shot_listwise_mix/"):
        os.makedirs("/share/yutao/yifei/reranking/data/four_shot_listwise_mix/")
    if not os.path.exists("/share/yutao/yifei/reranking/data/five_shot_listwise_mix/"):
        os.makedirs("/share/yutao/yifei/reranking/data/five_shot_listwise_mix/")
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--wise_type", type=str, default="pointwise", help="Type of listwise reranking, listwise or pointwise")
    parser.add_argument("--window_size", type=int, default=10, help="Window size for listwise reranking")
    args = parser.parse_args()
    
    print("ranking method: ", args.wise_type)
    print("window size: ", args.window_size)
    
    task = Reranking("ms_marco", "general_retrieval", "../rerank_task_data/msmarco.jsonl", wise_type=args.wise_type, window_size=args.window_size)
    task.generate_zero_shot_data()
    task.generate_few_shot_data(1)
    task.generate_few_shot_data(2)
    task.generate_few_shot_data(3)
    task.generate_few_shot_data(4)
    task.generate_few_shot_data(5)
    print("msmarco done")
    
    task = Reranking("touche", "argument_retrieval", "../rerank_task_data/touche.jsonl", wise_type=args.wise_type, window_size=args.window_size)
    task.generate_zero_shot_data()
    task.generate_few_shot_data(1)
    task.generate_few_shot_data(2)
    task.generate_few_shot_data(3)
    task.generate_few_shot_data(4)
    task.generate_few_shot_data(5)
    print("touche done")
    
    task = Reranking("arguana", "argument_retrieval", "../rerank_task_data/arguana.jsonl", wise_type=args.wise_type, window_size=args.window_size)
    task.generate_zero_shot_data()
    task.generate_few_shot_data(1)
    task.generate_few_shot_data(2)
    task.generate_few_shot_data(3)
    task.generate_few_shot_data(4)
    task.generate_few_shot_data(5)
    print("arguana done")
    
    task = Reranking("trec_covid", "biomedical_retrieval", "../rerank_task_data/trec_covid.jsonl", wise_type=args.wise_type, window_size=args.window_size)
    task.generate_zero_shot_data()
    task.generate_few_shot_data(1)
    task.generate_few_shot_data(2)
    task.generate_few_shot_data(3)
    task.generate_few_shot_data(4)
    task.generate_few_shot_data(5)
    print("trec_covid done")
    
    task = Reranking("nfcorpus", "biomedical_retrieval", "../rerank_task_data/nfcorpus.jsonl", wise_type=args.wise_type, window_size=args.window_size)
    task.generate_zero_shot_data()
    task.generate_few_shot_data(1)
    task.generate_few_shot_data(2)  
    task.generate_few_shot_data(3)
    task.generate_few_shot_data(4)
    task.generate_few_shot_data(5)
    print("nfcorpus done")
    
    task = Reranking("scidocs", "article_retrieval", "../rerank_task_data/scidocs.jsonl", wise_type=args.wise_type, window_size=args.window_size)
    task.generate_zero_shot_data()
    task.generate_few_shot_data(1)
    task.generate_few_shot_data(2)
    task.generate_few_shot_data(3)
    task.generate_few_shot_data(4)
    task.generate_few_shot_data(5)
    print("scidocs done")
    
    task = Reranking("quora", "duplicate_question_retrieval", "../rerank_task_data/quora.jsonl", wise_type=args.wise_type, window_size=args.window_size)
    task.generate_zero_shot_data()
    task.generate_few_shot_data(1)
    task.generate_few_shot_data(2)
    task.generate_few_shot_data(3)
    task.generate_few_shot_data(4)
    task.generate_few_shot_data(5)
    print("quora done")
    
    task = RerankingCqadupstack(wise_type=args.wise_type, window_size=args.window_size)
    task.generate_zero_shot_data()
    task.generate_few_shot_data(1)
    task.generate_few_shot_data(2)
    task.generate_few_shot_data(3)
    task.generate_few_shot_data(4)
    task.generate_few_shot_data(5)
    print("cqadupstack done")
    
    if args.wise_type != "listwise" or args.window_size == 10:
        task = RerankingDbpedia(wises_type=args.wise_type)
        task.generate_zero_shot_data()
        task.generate_few_shot_data(1)
        task.generate_few_shot_data(2)
        task.generate_few_shot_data(3)
        task.generate_few_shot_data(4)
        task.generate_few_shot_data(5)
        print("dbpedia done")
    
    task = Reranking("fever", "fact_retrieval", "../rerank_task_data/fever.jsonl", wise_type=args.wise_type, window_size=args.window_size)
    task.generate_zero_shot_data()
    task.generate_few_shot_data(1)
    task.generate_few_shot_data(2)
    task.generate_few_shot_data(3)
    task.generate_few_shot_data(4)
    task.generate_few_shot_data(5)
    print("fever done")
    
    task = Reranking("climate_fever", "fact_retrieval", "../rerank_task_data/climate_fever.jsonl", wise_type=args.wise_type, window_size=args.window_size)
    task.generate_zero_shot_data()
    task.generate_few_shot_data(1)
    task.generate_few_shot_data(2)
    task.generate_few_shot_data(3)
    task.generate_few_shot_data(4)
    task.generate_few_shot_data(5)
    print("climate_fever done")
    
    task = Reranking("scifact", "fact_retrieval", "../rerank_task_data/scifact.jsonl", wise_type=args.wise_type, window_size=args.window_size)
    task.generate_zero_shot_data()
    task.generate_few_shot_data(1)
    task.generate_few_shot_data(2)
    task.generate_few_shot_data(3)
    task.generate_few_shot_data(4)
    task.generate_few_shot_data(5)
    print("scifact done")
    
    task = Reranking("nq", "supporting_evidence_retrieval", "../rerank_task_data/nq.jsonl", wise_type=args.wise_type, window_size=args.window_size)
    task.generate_zero_shot_data()
    task.generate_few_shot_data(1)
    task.generate_few_shot_data(2)
    task.generate_few_shot_data(3)
    task.generate_few_shot_data(4)
    task.generate_few_shot_data(5)
    print("nq done")
    
    task = Reranking("fiqa", "supporting_evidence_retrieval", "../rerank_task_data/fiqa.jsonl", wise_type=args.wise_type, window_size=args.window_size)
    task.generate_zero_shot_data()
    task.generate_few_shot_data(1)
    task.generate_few_shot_data(2)
    task.generate_few_shot_data(3)
    task.generate_few_shot_data(4)
    task.generate_few_shot_data(5)
    print("fiqa done")
    
    task = Reranking("hotpot_qa", "supporting_evidence_retrieval", "../rerank_task_data/hotpotqa.jsonl", wise_type=args.wise_type, window_size=args.window_size)
    task.generate_zero_shot_data()
    task.generate_few_shot_data(1)
    task.generate_few_shot_data(2)
    task.generate_few_shot_data(3)
    task.generate_few_shot_data(4)
    task.generate_few_shot_data(5)
    print("hotpotqa done")
    print("\n")
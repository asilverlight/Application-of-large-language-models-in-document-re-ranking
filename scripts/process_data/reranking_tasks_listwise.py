import sys 
sys.path.append("..")
from task import Task
import json
import random
import re
from utils import generate_rerank_instruction, generate_cqadupstack_instruction, generate_dbpedia_instruction, map_shots_name, get_length, load_tokenizer, MAX_LEN
from instruct_templates import PATTERNS, TASK_DESCRIPTIONS
from tqdm import tqdm

class Reranking(Task):
    def __init__(self, name, cluster, processed_data_path):
        self._name = name
        self._cluster = cluster
        self._processed_data_path = processed_data_path
        super().__init__(self._name, self._cluster)
        self.print_name()
        
    def load_raw_data(self):
        pass

    def process_raw_data(self):
        pass

    def generate_zero_shot_data(self, split="train", with_desc=True):
        random.seed(0)
        all_data = []
        tokenizer = load_tokenizer()
        with open(self._processed_data_path, "r", encoding="utf-8") as fr:
            for line in fr:
                data = json.loads(line)
                all_data.append(data)

        with open("/share/yutao/yifei/reranking/data/zero_shot/" + self._cluster + "_" + self._name + ".zero_shot." + split + ".jsonl", "w", encoding="utf-8") as fw:
            for data in tqdm(all_data):
                template = random.choice(PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")])
                choose_yesno = [True, False]
                random.shuffle(choose_yesno)
                prompt = []
                completion = []
                index = PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")].index(template)
                description = ""
                if with_desc:
                    description = TASK_DESCRIPTIONS["retrieval"]
                if index == 0 or index == 1 or index == 6 or index == 7 or index == 10 or index == 11:
                    pr, co = generate_rerank_instruction(data, template, PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")], choose_yesno[0])
                    pr = re.sub(r" +", " ", pr)
                    co = re.sub(r" +", " ", co)
                    prompt.append(pr)
                    completion.append(co)
                    pr, co = generate_rerank_instruction(data, template, PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")], choose_yesno[1])
                    pr = re.sub(r" +", " ", pr)
                    co = re.sub(r" +", " ", co)
                    prompt.append(pr)
                    completion.append(co)
                else:
                    prompt, completion = generate_rerank_instruction(data, template, PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")])
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
        random.seed(0)
        all_data = []
        tokenizer = load_tokenizer()
        with open(self._processed_data_path, "r", encoding="utf-8") as fr:
            for line in fr:
                data = json.loads(line)
                all_data.append(data)

        with open("/share/yutao/yifei/reranking/data/" + map_shots_name(shots) + "_shot/" + self._cluster + "_" + self._name + "." + map_shots_name(shots) + "_shot." + split + ".jsonl", "w", encoding="utf-8") as fw:
            for data in tqdm(all_data):
                diverse_prompts = random.choice([0, 1])

                template = random.choice(PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")])
                choose_yesno = [True, False]
                random.shuffle(choose_yesno)
                test_prompt = []
                test_completion = []
                index = PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")].index(template)
                description = ""
                if with_desc:
                    description = TASK_DESCRIPTIONS["retrieval"]
                if index == 0 or index == 1 or index == 6 or index == 7 or index == 10 or index == 11:
                    pr, co = generate_rerank_instruction(data, template, PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")], choose_yesno[0])
                    pr = re.sub(r" +", " ", pr)
                    co = re.sub(r" +", " ", co)
                    test_prompt.append(pr)
                    test_completion.append(co)
                    pr, co = generate_rerank_instruction(data, template, PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")], choose_yesno[1])
                    pr = re.sub(r" +", " ", pr)
                    co = re.sub(r" +", " ", co)
                    test_prompt.append(pr)
                    test_completion.append(co)
                else:
                    test_prompt, test_completion = generate_rerank_instruction(data, template, PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")])
                    test_prompt = re.sub(r" +", " ", test_prompt)
                    test_completion = re.sub(r" +", " ", test_completion)

                few_shots_data = []
                while len(few_shots_data) < shots:
                    few_shot_data = random.choice(all_data)
                    if few_shot_data["query_id"] != data["query_id"]:
                        few_shots_data.append(few_shot_data)
                if diverse_prompts == 0:
                    examples = []
                    for few_shot_data in few_shots_data:
                        if index == 0 or index == 1 or index == 6 or index == 7 or index == 10 or index == 11:
                            yesno = random.choice([True, False])
                            prompt, completion = generate_rerank_instruction(few_shot_data, template, PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")], yesno)
                        else:
                            prompt, completion = generate_rerank_instruction(few_shot_data, template, PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")])
                        example = prompt.strip() + " " + completion
                        examples.append(example)
                else:
                    examples = []
                    for few_shot_data in few_shots_data:
                        index_e = random.choice([index - 1, index]) if index % 2 == 1 else random.choice([index, index + 1])
                        template = PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")][index_e]
                        if index_e == 0 or index_e == 1 or index_e == 6 or index_e == 7 or index_e == 10 or index_e == 11:
                            yesno = random.choice([True, False])
                            prompt, completion = generate_rerank_instruction(few_shot_data, template, PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")], yesno)
                        else:
                            prompt, completion = generate_rerank_instruction(few_shot_data, template, PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")])
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

class RerankingDbpedia(Task):
    def __init__(self):
        self._name = "dbpedia"
        self._cluster = "entity_retrieval"
        super().__init__(self._name, self._cluster)
        self._processed_data_path = "/share/yutao/yifei/reranking/data/rerank_task_data/dbpedia.jsonl"
        self.print_name()
    
    def load_raw_data(self):
        pass

    def process_raw_data(self):
        pass

    def generate_zero_shot_data(self, split, with_desc=True):
        random.seed(0)
        all_data = []
        tokenizer = load_tokenizer()
        with open(self._processed_data_path, "r", encoding="utf-8") as fr:
            for line in fr:
                data = json.loads(line)
                all_data.append(data)

        with open("/share/yutao/yifei/reranking/data/zero_shot/" + self._cluster + "_" + self._name + ".zero_shot." + split + ".jsonl", "w", encoding="utf-8") as fw:
            for data in tqdm(all_data):
                template = random.choice(PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")])
                choose_yesno = [True, False]
                random.shuffle(choose_yesno)
                prompt = []
                completion = []
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

    def generate_few_shot_data(self, split, shots, with_desc=True):
        random.seed(0)
        all_data = []
        tokenizer = load_tokenizer()
        with open(self._processed_data_path, "r", encoding="utf-8") as fr:
            for line in fr:
                data = json.loads(line)
                all_data.append(data)

        with open("/share/yutao/yifei/reranking/data/" + map_shots_name(shots) + "_shot/" + self._cluster + "_" + self._name + "." + map_shots_name(shots) + "_shot." + split + ".jsonl", "w", encoding="utf-8") as fw:
            for data in tqdm(all_data):
                diverse_prompts = random.choice([0, 1])

                template = random.choice(PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")])
                choose_yesno = [True, False]
                random.shuffle(choose_yesno)
                test_prompt = []
                test_completion = []
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
                    few_shot_data = random.choice(all_data)
                    if few_shot_data["query_id"] != data["query_id"]:
                        few_shots_data.append(few_shot_data)
                if diverse_prompts == 0:
                    examples = []
                    for few_shot_data in few_shots_data:
                        if index == 0 or index == 1 or index == 6 or index == 7 or index == 10 or index == 11:
                            yesno = random.choice([True, False])
                            prompt, completion = generate_dbpedia_instruction(few_shot_data, template, PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")], yesno)
                        else:
                            prompt, completion = generate_dbpedia_instruction(few_shot_data, template, PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")])
                        example = prompt.strip() + " " + completion
                        examples.append(example)
                else:
                    examples = []
                    for few_shot_data in few_shots_data:
                        index_e = random.choice([index - 1, index]) if index % 2 == 1 else random.choice([index, index + 1])
                        template = PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")][index_e]
                        if index_e == 0 or index_e == 1 or index_e == 6 or index_e == 7 or index_e == 10 or index_e == 11:
                            yesno = random.choice([True, False])
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
    def __init__(self):
        self._name = "cqadupstack"
        self._cluster = "duplicate_question_retrieval"
        super().__init__(self._name, self._cluster)
        self._processed_data_path = [
            "/share/yutao/yifei/reranking/data/rerank_task_data/cqadupstack_android.jsonl",
            "/share/yutao/yifei/reranking/data/rerank_task_data/cqadupstack_english.jsonl",
            "/share/yutao/yifei/reranking/data/rerank_task_data/cqadupstack_gaming.jsonl",
            "/share/yutao/yifei/reranking/data/rerank_task_data/cqadupstack_gis.jsonl",
            "/share/yutao/yifei/reranking/data/rerank_task_data/cqadupstack_mathematica.jsonl",
            "/share/yutao/yifei/reranking/data/rerank_task_data/cqadupstack_physics.jsonl",
            "/share/yutao/yifei/reranking/data/rerank_task_data/cqadupstack_programmers.jsonl",
            "/share/yutao/yifei/reranking/data/rerank_task_data/cqadupstack_stats.jsonl",
            "/share/yutao/yifei/reranking/data/rerank_task_data/cqadupstack_tex.jsonl",
            "/share/yutao/yifei/reranking/data/rerank_task_data/cqadupstack_unix.jsonl",
            "/share/yutao/yifei/reranking/data/rerank_task_data/cqadupstack_webmasters.jsonl",
            "/share/yutao/yifei/reranking/data/rerank_task_data/cqadupstack_wordpress.jsonl",
        ]
        self.print_name()
    
    def load_raw_data(self):
        pass

    def process_raw_data(self):
        pass

    def generate_zero_shot_data(self, split="train", with_desc=True):
        random.seed(0)
        tokenizer = load_tokenizer()
        fw = open("/share/yutao/yifei/reranking/data/zero_shot/" + self._cluster + "_" + self._name + ".zero_shot." + split + ".jsonl", "w", encoding="utf-8")
        for data_path in self._processed_data_path:
            subset = data_path.split("/")[-1].split(".")[0].split("_")[1]

            all_data = []
            with open(data_path, "r", encoding="utf-8") as fr:
                for line in fr:
                    data = json.loads(line)
                    all_data.append(data)
            
            for data in tqdm(all_data):
                template = random.choice(PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")])
                choose_yesno = [True, False]
                random.shuffle(choose_yesno)
                prompt = []
                completion = []
                index = PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")].index(template)
                description = ""
                if with_desc:
                    description = TASK_DESCRIPTIONS["retrieval"]
                if index == 0 or index == 1 or index == 6 or index == 7 or index == 10 or index == 11:
                    pr, co = generate_rerank_instruction(data, template, PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")], choose_yesno[0])
                    pr = re.sub(r" +", " ", pr)
                    co = re.sub(r" +", " ", co)
                    prompt.append(pr)
                    completion.append(co)
                    pr, co = generate_rerank_instruction(data, template, PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")], choose_yesno[1])
                    pr = re.sub(r" +", " ", pr)
                    co = re.sub(r" +", " ", co)
                    prompt.append(pr)
                    completion.append(co)
                else:
                    prompt, completion = generate_rerank_instruction(data, template, PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")])
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
        fw.close()

    def generate_few_shot_data(self, shots, split="train", with_desc=True):
        random.seed(0)
        tokenizer = load_tokenizer()
        fw = open("/share/yutao/yifei/reranking/data/" + map_shots_name(shots) + "_shot/" + self._cluster + "_" + self._name + "." + map_shots_name(shots) + "_shot." + split + ".jsonl", "a", encoding="utf-8")
        for data_path in self._processed_data_path:
            subset = data_path.split("/")[-1].split(".")[0].split("_")[1]

            all_data = []
            with open(data_path, "r", encoding="utf-8") as fr:
                for line in fr:
                    data = json.loads(line)
                    all_data.append(data)

            for data in tqdm(all_data):
                diverse_prompts = random.choice([0, 1])

                template = random.choice(PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")])
                choose_yesno = [True, False]
                random.shuffle(choose_yesno)
                test_prompt = []
                test_completion = []
                index = PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")].index(template)
                description = ""
                if with_desc:
                    description = TASK_DESCRIPTIONS["retrieval"]
                if index == 0 or index == 1 or index == 6 or index == 7 or index == 10 or index == 11:
                    pr, co = generate_rerank_instruction(data, template, PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")], choose_yesno[0])
                    pr = re.sub(r" +", " ", pr)
                    co = re.sub(r" +", " ", co)
                    test_prompt.append(pr)
                    test_completion.append(co)
                    pr, co = generate_rerank_instruction(data, template, PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")], choose_yesno[1])
                    pr = re.sub(r" +", " ", pr)
                    co = re.sub(r" +", " ", co)
                    test_prompt.append(pr)
                    test_completion.append(co)
                else:
                    test_prompt, test_completion = generate_rerank_instruction(data, template, PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")])
                    test_prompt = re.sub(r" +", " ", test_prompt)
                    test_completion = re.sub(r" +", " ", test_completion)

                few_shots_data = []
                while len(few_shots_data) < shots:
                    few_shot_data = random.choice(all_data)
                    if few_shot_data["query_id"] != data["query_id"]:
                        few_shots_data.append(few_shot_data)
                if diverse_prompts == 0:
                    examples = []
                    for few_shot_data in few_shots_data:
                        if index == 0 or index == 1 or index == 6 or index == 7 or index == 10 or index == 11:
                            yesno = random.choice([True, False])
                            prompt, completion = generate_rerank_instruction(few_shot_data, template, PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")], yesno)
                        else:
                            prompt, completion = generate_rerank_instruction(few_shot_data, template, PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")])
                        example = prompt.strip() + " " + completion
                        examples.append(example)
                else:
                    examples = []
                    for few_shot_data in few_shots_data:
                        index_e = random.choice([index - 1, index]) if index % 2 == 1 else random.choice([index, index + 1])
                        template = PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")][index_e]
                        if index_e == 0 or index_e == 1 or index_e == 6 or index_e == 7 or index_e == 10 or index_e == 11:
                            yesno = random.choice([True, False])
                            prompt, completion = generate_rerank_instruction(few_shot_data, template, PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")], yesno)
                        else:
                            prompt, completion = generate_rerank_instruction(few_shot_data, template, PATTERNS[(self._cluster + "_" + self._name).replace("_", "-")])
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

        fw.close()

if __name__ == "__main__":
    task = Reranking("ms_marco", "general_retrieval", "/share/yutao/yifei/reranking/data/rerank_task_data/msmarco.jsonl")
    task.generate_zero_shot_data()
    task.generate_few_shot_data(1)
    task.generate_few_shot_data(2)
    task.generate_few_shot_data(3)
    task.generate_few_shot_data(4)
    task.generate_few_shot_data(5)
    print("msmarco done")
    
    task = Reranking("touche", "argument_retrieval", "/share/yutao/yifei/reranking/data/rerank_task_data/touche.jsonl")
    task.generate_zero_shot_data()
    task.generate_few_shot_data(1)
    task.generate_few_shot_data(2)
    task.generate_few_shot_data(3)
    task.generate_few_shot_data(4)
    task.generate_few_shot_data(5)
    print("touche done")
    
    task = Reranking("arguana", "argument_retrieval", "/share/yutao/yifei/reranking/data/rerank_task_data/arguana.jsonl")
    task.generate_zero_shot_data()
    task.generate_few_shot_data(1)
    task.generate_few_shot_data(2)
    task.generate_few_shot_data(3)
    task.generate_few_shot_data(4)
    task.generate_few_shot_data(5)
    print("arguana done")
    
    task = Reranking("trec_covid", "biomedical_retrieval", "/share/yutao/yifei/reranking/data/rerank_task_data/trec_covid.jsonl")
    task.generate_zero_shot_data()
    task.generate_few_shot_data(1)
    task.generate_few_shot_data(2)
    task.generate_few_shot_data(3)
    task.generate_few_shot_data(4)
    task.generate_few_shot_data(5)
    print("trec_covid done")
    
    task = Reranking("nfcorpus", "biomedical_retrieval", "/share/yutao/yifei/reranking/data/rerank_task_data/nfcorpus.jsonl")
    task.generate_zero_shot_data()
    task.generate_few_shot_data(1)
    task.generate_few_shot_data(2)  
    task.generate_few_shot_data(3)
    task.generate_few_shot_data(4)
    task.generate_few_shot_data(5)
    print("nfcorpus done")
    
    task = Reranking("scidocs", "article_retrieval", "/share/yutao/yifei/reranking/data/rerank_task_data/scidocs.jsonl")
    task.generate_zero_shot_data()
    task.generate_few_shot_data(1)
    task.generate_few_shot_data(2)
    task.generate_few_shot_data(3)
    task.generate_few_shot_data(4)
    task.generate_few_shot_data(5)
    print("scidocs done")
    
    task = Reranking("quora", "duplicate_question_retrieval", "/share/yutao/yifei/reranking/data/rerank_task_data/quora.jsonl")
    task.generate_zero_shot_data()
    task.generate_few_shot_data(1)
    task.generate_few_shot_data(2)
    task.generate_few_shot_data(3)
    task.generate_few_shot_data(4)
    task.generate_few_shot_data(5)
    print("quora done")
    
    task = RerankingCqadupstack()
    task.generate_zero_shot_data()
    task.generate_few_shot_data(1)
    task.generate_few_shot_data(2)
    task.generate_few_shot_data(3)
    task.generate_few_shot_data(4)
    task.generate_few_shot_data(5)
    print("cqadupstack done")
    
    task = RerankingDbpedia()
    task.generate_zero_shot_data()
    task.generate_few_shot_data(1)
    task.generate_few_shot_data(2)
    task.generate_few_shot_data(3)
    task.generate_few_shot_data(4)
    task.generate_few_shot_data(5)
    print("dbpedia done")
    
    task = Reranking("fever", "fact_retrieval", "/share/yutao/yifei/reranking/data/rerank_task_data/fever.jsonl")
    task.generate_zero_shot_data()
    task.generate_few_shot_data(1)
    task.generate_few_shot_data(2)
    task.generate_few_shot_data(3)
    task.generate_few_shot_data(4)
    task.generate_few_shot_data(5)
    print("fever done")
    
    task = Reranking("climate_fever", "fact_retrieval", "/share/yutao/yifei/reranking/data/rerank_task_data/climate_fever.jsonl")
    task.generate_zero_shot_data()
    task.generate_few_shot_data(1)
    task.generate_few_shot_data(2)
    task.generate_few_shot_data(3)
    task.generate_few_shot_data(4)
    task.generate_few_shot_data(5)
    print("climate_fever done")
    
    task = Reranking("scifact", "fact_retrieval", "/share/yutao/yifei/reranking/data/rerank_task_data/scifact.jsonl")
    task.generate_zero_shot_data()
    task.generate_few_shot_data(1)
    task.generate_few_shot_data(2)
    task.generate_few_shot_data(3)
    task.generate_few_shot_data(4)
    task.generate_few_shot_data(5)
    print("scifact done")
    
    task = Reranking("nq", "natural_question_retrieval", "/share/yutao/yifei/reranking/data/rerank_task_data/nq.jsonl")
    task.generate_zero_shot_data()
    task.generate_few_shot_data(1)
    task.generate_few_shot_data(2)
    task.generate_few_shot_data(3)
    task.generate_few_shot_data(4)
    task.generate_few_shot_data(5)
    print("nq done")
    
    task = Reranking("fiqa", "supporting_evidence_retrieval", "/share/yutao/yifei/reranking/data/rerank_task_data/fiqa.jsonl")
    task.generate_zero_shot_data()
    task.generate_few_shot_data(1)
    task.generate_few_shot_data(2)
    task.generate_few_shot_data(3)
    task.generate_few_shot_data(4)
    task.generate_few_shot_data(5)
    print("fiqa done")
    
    task = Reranking("hotpotqa", "supporting_evidence_retrieval", "/share/yutao/yifei/reranking/data/rerank_task_data/hotpotqa.jsonl")
    task.generate_zero_shot_data()
    task.generate_few_shot_data(1)
    task.generate_few_shot_data(2)
    task.generate_few_shot_data(3)
    task.generate_few_shot_data(4)
    task.generate_few_shot_data(5)
    print("hotpotqa done")
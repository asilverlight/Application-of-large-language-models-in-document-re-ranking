import numpy as np
from task import TASK_DATASET_DICT
from utils import map_shots_name
import json
import os
import random
import argparse

def load_overlap_data(wise_type="all", window_size=10, has_dbpedia=True):
    all_data = {}
    for cluster in TASK_DATASET_DICT:
        for name in TASK_DATASET_DICT[cluster]:
            if name == "dbpedia" and not has_dbpedia:
                continue# 由于dbpedia的特殊性，根据实验要求处理
            full_name = cluster + "_" + name
            dataset_train_data = []
            for i in range(6):
                if wise_type == "all":
                    result = "../" + map_shots_name(i) + "_shot/" + full_name + "." + map_shots_name(i) + "_shot.train.jsonl"
                elif wise_type == "pointwise":
                    result = "../" + map_shots_name(i) + "_shot_pointwise/" + full_name + "." + map_shots_name(i) + "_shot.train.jsonl"
                elif wise_type == "pairwise":
                    result = "../" + map_shots_name(i) + "_shot_pairwise/" + full_name + "." + map_shots_name(i) + "_shot.train.jsonl"
                elif wise_type == "listwise" and window_size != 0:
                    result = "../" + map_shots_name(i) + "_shot_listwise_" + str(window_size) + "/" + full_name + "." + map_shots_name(i) + "_shot.train.jsonl"    
                elif wise_type == "listwise" and window_size == 0:
                    result = "../" + map_shots_name(i) + "_shot_listwise_mix/" + full_name + "." + map_shots_name(i) + "_shot.train.jsonl"
                else:
                    raise ValueError("Invalid wise_type")
                with open(result, "r") as fr:
                    for line in fr:
                        sample = json.loads(line)
                        dataset_train_data.append(sample)
            print(f"Dataset: {full_name} \t Size: {len(dataset_train_data)}")
            all_data[full_name] = dataset_train_data
    return all_data

def load_all_data():
    all_data = {}
    for cluster in TASK_DATASET_DICT:
        for name in TASK_DATASET_DICT[cluster]:
            full_name = cluster + "_" + name
            dataset_train_data = []
            for i in range(6):
                with open("generated_data/" + map_shots_name(i) + "_shot/" + full_name + "." + map_shots_name(i) + "_shot.train.jsonl", "r") as fr:
                    for line in fr:
                        sample = json.loads(line)
                        dataset_train_data.append(sample)
            for i in range(6):
                if os.path.exists("generated_data/" + map_shots_name(i) + "_shot/" + full_name + "." + map_shots_name(i) + "_shot.dev.jsonl"):
                    with open("generated_data/" + map_shots_name(i) + "_shot/" + full_name + "." + map_shots_name(i) + "_shot.dev.jsonl", "r") as fr:
                        for line in fr:
                            sample = json.loads(line)
                            dataset_train_data.append(sample)
            print(f"Dataset: {full_name} \t Size: {len(dataset_train_data)}")
            all_data[full_name] = dataset_train_data
    return all_data

def generate_data_mixture_overlap(datasets, target_size, save_path):
    np.random.seed(0)
    # 假设 datasets 是一个包含多个数据集的列表

    # 目标采样总数
    total_samples = target_size

    # 计算每个数据集的大小（最大为5000）
    sizes = [min(len(dataset), 5000) for dataset in datasets]

    # 计算总权重
    total_weight = sum(sizes)

    # 计算每个数据集的采样比例
    proportions = [size / total_weight for size in sizes]

    # 根据比例进行采样
    sampled_data = []
    for dataset, proportion in zip(datasets, proportions):
        # 根据比例计算每个数据集的采样数量
        num_samples = int(proportion * total_samples)
        # 防止采样数量大于数据集大小
        num_samples = min(num_samples, len(dataset))
        # 从数据集中随机采样
        sampled_data.extend(np.random.choice(dataset, num_samples, replace=False))

    # sampled_data 现在包含了根据权重采样的数据
    with open(save_path, "w", encoding="utf-8") as fw:
        for data in sampled_data:
            fw.write(json.dumps(data) + "\n")

def generate_data_mixture_remove_one_cluster(datasets, removed_cluster, target_size, save_path):
    all_data = []
    removed_dataset = TASK_DATASET_DICT[removed_cluster]
    for full_name in datasets.keys():
        if removed_cluster in full_name:
            print(full_name)
            continue
        flag = 0
        for rm_dataset in removed_dataset:
            if rm_dataset in full_name:
                print(full_name)
                flag = 1
        if flag:
            continue
        all_data.append(datasets[full_name])
    generate_data_mixture_overlap(all_data, target_size, save_path)

def remove_from_overlap_data(overlap_data, target_data, removed_cluster):
    with open(overlap_data, "r", encoding="utf-8") as fr:
        with open(target_data, "w", encoding="utf-8") as fw:
            for line in fr:
                data = json.loads(line)
                source = data["source"]
                # if removed_cluster[0] not in source and removed_cluster[1] not in source:
                remove = False
                for rm in removed_cluster:
                    if rm in source:
                        remove = True
                if remove:
                    continue
                # if removed_cluster in source:
                #     continue
                else:
                    fw.write(line)

def remove_desc():
    for cluster in TASK_DATASET_DICT:
        for name in TASK_DATASET_DICT[cluster]:
            if "retrieval" in cluster:
                continue
            for split in ["test"]:
                full_name = cluster + "_" + name
                print(full_name)
                for i in range(6):
                    with open(f"generated_data/{map_shots_name(i)}_shot/{full_name}.{map_shots_name(i)}_shot.{split}.jsonl", "r") as fr:
                        with open(f"generated_data_no_desc/{map_shots_name(i)}_shot/{full_name}.{map_shots_name(i)}_shot.{split}.jsonl", "w") as fw:
                            for line in fr:
                                data = json.loads(line)
                                prompt = data["prompt"].split("\n\n")
                                prompt = "\n\n".join(prompt[1:])
                                data["prompt"] = prompt
                                fw.write(json.dumps(data) + "\n")

def get_full_retrieval_data():
    random.seed(0)
    with open("generated_data/setting_more_retrieval/more_retrieval_data.jsonl", "w", encoding="utf-8") as fw:
        for cluster in TASK_DATASET_DICT:
            if "retrieval" in cluster:
                for name in TASK_DATASET_DICT[cluster]:
                    full_name = cluster + "_" + name
                    dataset_train_data = []
                    for i in range(6):
                        with open("generated_data/" + map_shots_name(i) + "_shot/" + full_name + "." + map_shots_name(i) + "_shot.train.jsonl", "r") as fr:
                            for line in fr:
                                sample = json.loads(line)
                                dataset_train_data.append(sample)
                    print(f"Dataset: {full_name} \t Size: {len(dataset_train_data)}")
                    random.shuffle(dataset_train_data)
                    data_num = min(10000, len(dataset_train_data))
                    dataset_train_data = dataset_train_data[:data_num]
                    for data in dataset_train_data:
                        fw.write(json.dumps(data) + "\n")

def expand_original_overlap_data():
    with open("generated_data/setting_overlap/train.20w.jsonl", "r", encoding="utf-8") as fr1:
        with open("generated_data/setting_more_retrieval/more_retrieval_data.jsonl", "r", encoding="utf-8") as fr2:
            with open("generated_data/setting_more_retrieval/train.jsonl", "w", encoding="utf-8") as fw:
                for line in fr1:
                    data = json.loads(line)
                    source = data["source"]
                    if "retrieval" not in source:
                        fw.write(line)
                for line in fr2:
                    fw.write(line)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--wise_type", type=str, default="all", help="all/pointwise/pairwise/listwise")
    parser.add_argument("--window_size", type=int, default=10, help="window size for listwise")
    parser.add_argument("--save_path", type=str, default="../train.jsonl", help="path to save the mixture data")
    parser.add_argument("--with_dbpedia", type=lambda x: (str(x).lower() == 'true'), default=True, help="which experiment to run")
    # 有dbpedia代表在跑基本实验，没有dbpedia代表在跑listwise长度的消融实验
    args = parser.parse_args()
    
    for k, v in vars(args).items():
        print(k, '=', v)
    
    if not os.path.exists(args.save_path):
        all_datasets = load_overlap_data(wise_type=args.wise_type, window_size=args.window_size, has_dbpedia=args.with_dbpedia)
        all_data = [all_datasets[x] for x in all_datasets.keys()]
        
        generate_data_mixture_overlap(all_data, 200000, args.save_path)
        if args.wise_type != "listwise":
            print("Data " + args.wise_type + " mixture generated.")
        else:
            print("Data " + args.wise_type + " mixture generated with window size " + str(args.window_size) + ".")
    else:
        print("Data already exists, skip generating data.")
    print("\n")
    # all_datasets = load_all_data()
    # generate_data_mixture_remove_one_cluster(all_datasets, "query_description", 100000, "generated_data/setting_removed_cluster/query_description/train.10w.jsonl")
    # remove_from_overlap_data("generated_data/setting_overlap/train.20w.jsonl", "generated_data/setting_removed_dataset/ensemble/train.jsonl", ["trec_robust", "qrecc", "mimics_duo", "climate_fever", "xsum", "quora", "nq"])
    # remove_desc()
    # get_full_retrieval_data()
    # expand_original_overlap_data()
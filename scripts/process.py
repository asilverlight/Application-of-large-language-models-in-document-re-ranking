import json

def remove_from_overlap_data(overlap_data, target_data, removed_cluster):
    with open(overlap_data, "r", encoding="utf-8") as fr:
        with open(target_data, "w", encoding="utf-8") as fw:
            for line in fr:
                data = json.loads(line)
                source = data["source"]
                if removed_cluster in source:
                    continue
                else:
                    fw.write(line)

remove_from_overlap_data("data/setting_overlap/train.20w.jsonl", "data/setting_removed_dataset/cnndm/train.jsonl", "cnndm")
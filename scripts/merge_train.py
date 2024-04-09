import os
import sys
import ujson as json

input_folder = "data/rerank_task_data"
output_folder = "data/train.jsonl"

if not os.path.exists(output_folder):
    with open(output_folder, "w", encoding='utf-8') as fw:
        for file_name in os.listdir(input_folder):
            if file_name.endswith(".jsonl"):
                file_path = os.path.join(input_folder, file_name)
                with open(file_path, "r", encoding='utf-8') as fr:
                    for line in fr:
                        data = json.loads(line)
                        fw.write(json.dumps(data) + "\n")
                print(f"Processed {file_name}")
print(f"Merged data saved to {output_folder}")
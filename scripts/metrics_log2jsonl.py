import re
import ujson as json
import argparse

def extract_info(log_file):
    with open(log_file, 'r') as f:
        lines = f.readlines()

    data = []
    current_group = {}

    for line in lines:
        line = line.strip()
        if line.startswith("Args"):
            # Extract rerank_method and dataset_name from Command line
            metrics_str = re.search(r": (\{.+\})", line).group(1)
            metrics = json.loads(metrics_str)
            current_group["rerank_method"] = metrics["rerank_method"]
            current_group["dataset_name"] = metrics["dataset_name"]
        elif line.startswith("Metrics"):
            # Extract metrics
            metrics_str = re.search(r": (\{.+\})", line).group(1)
            metrics = json.loads(metrics_str)
            current_group.update(metrics)
            data.append(current_group)
            current_group = {}
    
    return data

def save_to_jsonl(data, output_file):
    with open(output_file, 'w') as f:
        for item in data:
            json.dump(item, f)
            f.write('\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", type=str, help="Log file path")
    parser.add_argument("--output_file", type=str, help="Output file path")
    args = parser.parse_args()

    extracted_data = extract_info(args.log_file)
    save_to_jsonl(extracted_data, args.output_file)
import os
import ssl
import time
from huggingface_hub import hf_hub_download, upload_file
from requests.exceptions import RequestException
import random

repo_id = "yutaozhu94/test"
repo_type = "dataset"

# hf_hub_download(repo_id=repo_id, filename=f"climate_fever.bm25.100.jsonl", repo_type=repo_type, local_dir="./data/ranking/", local_dir_use_symlinks=False)
# hf_hub_download(repo_id=repo_id, filename=f"fever.bm25.100.jsonl", repo_type=repo_type, local_dir="./data/ranking/", local_dir_use_symlinks=False)
# hf_hub_download(repo_id=repo_id, filename=f"fiqa.bm25.100.jsonl", repo_type=repo_type, local_dir="./data/ranking/", local_dir_use_symlinks=False)
# hf_hub_download(repo_id=repo_id, filename=f"hotpot_qa.bm25.100.jsonl", repo_type=repo_type, local_dir="./data/ranking/", local_dir_use_symlinks=False)
# hf_hub_download(repo_id=repo_id, filename=f"nfcorpus.bm25.100.jsonl", repo_type=repo_type, local_dir="./data/ranking/", local_dir_use_symlinks=False)
# hf_hub_download(repo_id=repo_id, filename=f"quora.bm25.100.jsonl", repo_type=repo_type, local_dir="./data/ranking/", local_dir_use_symlinks=False)
# hf_hub_download(repo_id=repo_id, filename=f"scidocs.bm25.100.jsonl", repo_type=repo_type, local_dir="./data/ranking/", local_dir_use_symlinks=False)
# hf_hub_download(repo_id=repo_id, filename=f"scifact.bm25.100.jsonl", repo_type=repo_type, local_dir="./data/ranking/", local_dir_use_symlinks=False)
# hf_hub_download(repo_id=repo_id, filename=f"touche.bm25.100.jsonl", repo_type=repo_type, local_dir="./data/ranking/", local_dir_use_symlinks=False)
# hf_hub_download(repo_id=repo_id, filename=f"trec_covid.bm25.100.jsonl", repo_type=repo_type, local_dir="./data/ranking/", local_dir_use_symlinks=False)
# hf_hub_download(repo_id=repo_id, filename=f"nq.bm25.100.jsonl", repo_type=repo_type, local_dir="./data/ranking/", local_dir_use_symlinks=False)
# hf_hub_download(repo_id=repo_id, filename=f"dbpedia.bm25.100.jsonl", repo_type=repo_type, local_dir="./data/ranking/", local_dir_use_symlinks=False)
# upload_file(path_or_fileobj="./llm.tar.gz", path_in_repo="llm.tar.gz", repo_id=repo_id, repo_type=repo_type)
hf_hub_download(repo_id=repo_id, filename=f"train.20w.jsonl", repo_type=repo_type, local_dir="./data/setting_no_temp/", local_dir_use_symlinks=False)

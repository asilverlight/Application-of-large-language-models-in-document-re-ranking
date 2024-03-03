## Note
- Modify `model_cache_dir` and `dataset_cache_dir` in [`args.py`](src/args.py) to your location! (Set to None to use the default one.)

## Generation
```bash
torchrun --nproc_per_node 8 eval_generation.py \
--eval_data /home/baaiks/peitian/Data/Datasets/searchgpt/query_expansion_trec_covid.three_shot.test.jsonl \
--output_dir data/results/generation/query_expansion_trec_covid.three_shot.test \
--model_name_or_path meta-llama/Llama-2-7b-chat-hf \
--tokenizer_name_or_path meta-llama/Llama-2-7b-chat-hf
```

## Reranking
```bash
torchrun --nproc_per_node 8 eval_rerank.py \
--eval_data /share/peitian/Data/Datasets/searchgpt/qrels.dev.pt.key.do-not-overwrite.jsonl \
--output_dir data/results/rerank/msmarco \
--model_name_or_path meta-llama/Llama-2-7b-chat-hf \
--tokenizer_name_or_path meta-llama/Llama-2-7b-chat-hf

# directly evaluate retrieval results
--rerank_method no

# pointwise:  (default)
--rerank_method pointwise

# pairwise:
--rerank_method pairwise

# listwise:
--rerank_method listwise \
--listwise_window 5 \
--listwise_stride 5

# fewshot:
--fewshot_data /share/peitian/Data/Datasets/searchgpt/qrels.train.pt.neg.do-not-overwrite.fewshot.jsonl \
--shots 1
```

### Evaluating on CQA
```bash
# remember to modify arguments inside the script
bash cqa.sh
```

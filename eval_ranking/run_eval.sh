# TOKENIZER_PATH="/share/LMs/models--meta-llama--Llama-2-7b-hf/snapshots/8cca527612d856d7d32bd94f8103728d614eb852/"
# MODEL_PATH="/share/yutao/inters/checkpoint/base_setting_overlap/epoch2-step3000/model"
# RESULT_PATH="result/llama-2-7b"

GPU_NUM=8
model_paths=(
  # "/share/yutao/inters/checkpoint/chat_setting_overlap/epoch2-step3000/model" 
  # "/share/yutao/inters/checkpoint/base_setting_overlap/epoch2-step3000/model" 
  # "/share/yutao/inters/checkpoint/chat_flan/epoch2-step3000/model" 
  # "/share/yutao/inters/checkpoint/chat_setting_removed_query/epoch2-step2100/model" 
  # "/share/yutao/inters/checkpoint/chat_setting_removed_general_retrieval/epoch2-step3000/model" 
  # "/share/yutao/inters/checkpoint/chat_setting_removed_query_intent_classification/epoch1-step5800/model"
  # "/share/yutao/inters/checkpoint/chat_setting_removed_ensemble/epoch2-step2500/model"
  # "/share/yutao/inters/checkpoint/chat_setting_more_retrieval/epoch2-step4000/model"
  # "/share/yutao/inters/checkpoint/chat_setting_no_template/epoch2-step3000/model"
  # "/share/yutao/inters/checkpoint/chat_setting_no_doc_new/epoch2-step2000/model"
  # "/share/yutao/inters/checkpoint/chat_setting_no_ret_new/epoch2-step2100/model"
  # "/share/yutao/inters/checkpoint/chat_setting_overlap_new//epoch2-step3100/model"
  "/share/yutao/yifei/reranking/checkpoint/epoch2-step30/model"
  # "/share/yutao/inters/checkpoint/chat_setting_no_query_new/epoch2-step2100/model"
  # "/share/yutao/inters/checkpoint/chat_setting_no_dataset_new/epoch2-step2500/model"
  # "/share/yutao/inters/checkpoint/chat_setting_overlap_no_ar_new/epoch2-step3000/model"
  # "/share/yutao/inters/checkpoint/chat_setting_overlap_no_desc_new/epoch2-step3100/model"
  # "/share/LMs/models--meta-llama--Llama-2-7b-hf/snapshots/8cca527612d856d7d32bd94f8103728d614eb852/"
)

tokenizer_paths=(
  # "/share/LMs/models--meta-llama--Llama-2-7b-hf/snapshots/8cca527612d856d7d32bd94f8103728d614eb852/"
  # "/share/LMs/models--meta-llama--Llama-2-7b-hf/snapshots/8cca527612d856d7d32bd94f8103728d614eb852/"
  # "/share/LMs/models--meta-llama--Llama-2-7b-hf/snapshots/8cca527612d856d7d32bd94f8103728d614eb852/"
  # "/share/LMs/models--meta-llama--Llama-2-7b-hf/snapshots/8cca527612d856d7d32bd94f8103728d614eb852/"
  # "/share/LMs/models--meta-llama--Llama-2-7b-hf/snapshots/8cca527612d856d7d32bd94f8103728d614eb852/"
  "/share/LMs/models--meta-llama--Llama-2-7b-hf/snapshots/8cca527612d856d7d32bd94f8103728d614eb852/"
  # "/share/LMs/models--meta-llama--Llama-2-7b-hf/snapshots/8cca527612d856d7d32bd94f8103728d614eb852/"
  # "/share/LMs/models--meta-llama--Llama-2-7b-hf/snapshots/8cca527612d856d7d32bd94f8103728d614eb852/"
)

result_paths=(
  # "result/chat-inters"
  # "result/base-inters"
  # "result/chat-flan"
  # "result/chat-inters-remove-query"
  # "result/chat-inters-remove-gene-retr"
  # "result/chat-inters-remove-query-inte-class"
  # "result/chat-inters-remove-ensemble"
  # "result/chat-inters-more-ret"
  # "result/chat-inters-no-temp"
  # "result/chat-inters-no-doc"
  # "result/chat-inters-no-ret"
  "result"
  # "result/chat-inters-no-query"
  # "result/chat-inters-no-dataset"
  # "result/chat-inters-no-ar"
  # "result/chat-inters-no-desc"
  # "result/chat"
)

for ((i=0; i<1; i++)); do
  MODEL_PATH=${model_paths[i]}
  TOKENIZER_PATH=${tokenizer_paths[i]}
  RESULT_PATH=${result_paths[i]}

  # torchrun --nproc_per_node 8 eval_rerank.py \
  #   --eval_data /share/yutao/inters/data/ranking/msmarco.bm25.100.jsonl \
  #   --output_dir ${RESULT_PATH} \
  #   --model_name_or_path ${MODEL_PATH} \
  #   --tokenizer_name_or_path ${TOKENIZER_PATH} \
  #   --dataset_cache_dir /share/yutao/hf_cache/dataset/ \
  #   --use_flash_attention_2 False \
  #   --max_length 2048 \
  #   --batch_size 4 \
  #   --with_description False \
  #   --dataset_name msmarco \
  #   --with_description False

  # torchrun --nproc_per_node 8 eval_rerank.py \
  #   --eval_data /share/yutao/inters/data/ranking/touche.bm25.100.jsonl \
  #   --output_dir ${RESULT_PATH} \
  #   --model_name_or_path ${MODEL_PATH} \
  #   --tokenizer_name_or_path ${TOKENIZER_PATH} \
  #   --dataset_cache_dir /share/yutao/hf_cache/dataset/ \
  #   --use_flash_attention_2 False \
  #   --max_length 2048 \
  #   --batch_size 4 \
  #   --with_description False \
  #   --dataset_name touche

  # torchrun --nproc_per_node 8 eval_rerank.py \
  #   --eval_data /share/yutao/inters/data/ranking/arguana.bm25.100.jsonl \
  #   --output_dir ${RESULT_PATH} \
  #   --model_name_or_path ${MODEL_PATH} \
  #   --tokenizer_name_or_path ${TOKENIZER_PATH} \
  #   --dataset_cache_dir /share/yutao/hf_cache/dataset/ \
  #   --use_flash_attention_2 False \
  #   --max_length 2048 \
  #   --batch_size 4 \
  #   --with_description False \
  #   --dataset_name arguana

  # torchrun --nproc_per_node 8 eval_rerank.py \
  #   --eval_data /share/yutao/inters/data/ranking/trec_covid.bm25.100.jsonl \
  #   --output_dir ${RESULT_PATH} \
  #   --model_name_or_path ${MODEL_PATH} \
  #   --tokenizer_name_or_path ${TOKENIZER_PATH} \
  #   --dataset_cache_dir /share/yutao/hf_cache/dataset/ \
  #   --use_flash_attention_2 False \
  #   --max_length 2048 \
  #   --batch_size 4 \
  #   --with_description False \
  #   --dataset_name trec_covid
  
  # torchrun --nproc_per_node 8 eval_rerank.py \
  #   --eval_data /share/yutao/inters/data/ranking/nfcorpus.bm25.100.jsonl \
  #   --output_dir ${RESULT_PATH} \
  #   --model_name_or_path ${MODEL_PATH} \
  #   --tokenizer_name_or_path ${TOKENIZER_PATH} \
  #   --dataset_cache_dir /share/yutao/hf_cache/dataset/ \
  #   --use_flash_attention_2 False \
  #   --max_length 2048 \
  #   --batch_size 4 \
  #   --with_description False \
  #   --dataset_name nfcorpus

  # torchrun --nproc_per_node 8 eval_rerank.py \
  #   --eval_data /share/yutao/inters/data/ranking/scidocs.bm25.100.jsonl \
  #   --output_dir ${RESULT_PATH} \
  #   --model_name_or_path ${MODEL_PATH} \
  #   --tokenizer_name_or_path ${TOKENIZER_PATH} \
  #   --dataset_cache_dir /share/yutao/hf_cache/dataset/ \
  #   --use_flash_attention_2 False \
  #   --max_length 2048 \
  #   --batch_size 4 \
  #   --with_description False \
  #   --dataset_name scidocs
  
  # torchrun --nproc_per_node 8 eval_rerank.py \
  #   --eval_data /share/yutao/inters/data/ranking/quora.bm25.100.jsonl \
  #   --output_dir ${RESULT_PATH} \
  #   --model_name_or_path ${MODEL_PATH} \
  #   --tokenizer_name_or_path ${TOKENIZER_PATH} \
  #   --dataset_cache_dir /share/yutao/hf_cache/dataset/ \
  #   --use_flash_attention_2 False \
  #   --max_length 2048 \
  #   --batch_size 4 \
  #   --with_description False \
  #   --dataset_name quora

  torchrun --nproc_per_node 8 eval_rerank.py \
    --eval_data /share/yutao/yifei/reranking/data/dbpedia.bm25.100.top40.jsonl \
    --output_dir ${RESULT_PATH} \
    --model_name_or_path ${MODEL_PATH} \
    --tokenizer_name_or_path ${TOKENIZER_PATH} \
    --dataset_cache_dir /share/yutao/hf_cache/dataset/ \
    --use_flash_attention_2 False \
    --max_length 2048 \
    --batch_size 4 \
    --with_description True \
    --rerank_method listwise \
    --listwise_window 5 \
    --listwise_stride 5 \
    --dataset_name dbpedia

  # torchrun --nproc_per_node 8 eval_rerank.py \
  #   --eval_data /share/yutao/inters/data/ranking/fever.bm25.100.jsonl \
  #   --output_dir ${RESULT_PATH} \
  #   --model_name_or_path ${MODEL_PATH} \
  #   --tokenizer_name_or_path ${TOKENIZER_PATH} \
  #   --dataset_cache_dir /share/yutao/hf_cache/dataset/ \
  #   --use_flash_attention_2 False \
  #   --max_length 2048 \
  #   --batch_size 4 \
  #   --with_description False \
  #   --dataset_name fever

  # torchrun --nproc_per_node 8 eval_rerank.py \
  #   --eval_data /share/yutao/inters/data/ranking/climate_fever.bm25.100.jsonl \
  #   --output_dir ${RESULT_PATH} \
  #   --model_name_or_path ${MODEL_PATH} \
  #   --tokenizer_name_or_path ${TOKENIZER_PATH} \
  #   --dataset_cache_dir /share/yutao/hf_cache/dataset/ \
  #   --use_flash_attention_2 False \
  #   --max_length 2048 \
  #   --batch_size 4 \
  #   --with_description False \
  #   --dataset_name climate_fever

  # torchrun --nproc_per_node 8 eval_rerank.py \
  #   --eval_data /share/yutao/inters/data/ranking/scifact.bm25.100.jsonl \
  #   --output_dir ${RESULT_PATH} \
  #   --model_name_or_path ${MODEL_PATH} \
  #   --tokenizer_name_or_path ${TOKENIZER_PATH} \
  #   --dataset_cache_dir /share/yutao/hf_cache/dataset/ \
  #   --use_flash_attention_2 False \
  #   --max_length 2048 \
  #   --batch_size 4 \
  #   --with_description False \
  #   --dataset_name scifact

  # torchrun --nproc_per_node 8 eval_rerank.py \
  #   --eval_data /share/yutao/inters/data/ranking/nq.bm25.100.jsonl \
  #   --output_dir ${RESULT_PATH} \
  #   --model_name_or_path ${MODEL_PATH} \
  #   --tokenizer_name_or_path ${TOKENIZER_PATH} \
  #   --dataset_cache_dir /share/yutao/hf_cache/dataset/ \
  #   --use_flash_attention_2 False \
  #   --max_length 2048 \
  #   --batch_size 4 \
  #   --with_description False \
  #   --dataset_name nq

  # torchrun --nproc_per_node 8 eval_rerank.py \
  #   --eval_data /share/yutao/inters/data/ranking/fiqa.bm25.100.jsonl \
  #   --output_dir ${RESULT_PATH} \
  #   --model_name_or_path ${MODEL_PATH} \
  #   --tokenizer_name_or_path ${TOKENIZER_PATH} \
  #   --dataset_cache_dir /share/yutao/hf_cache/dataset/ \
  #   --use_flash_attention_2 False \
  #   --max_length 2048 \
  #   --batch_size 4 \
  #   --with_description False \
  #   --dataset_name fiqa

  # torchrun --nproc_per_node 8 eval_rerank.py \
  #   --eval_data /share/yutao/inters/data/ranking/hotpot_qa.bm25.100.jsonl \
  #   --output_dir ${RESULT_PATH} \
  #   --model_name_or_path ${MODEL_PATH} \
  #   --tokenizer_name_or_path ${TOKENIZER_PATH} \
  #   --dataset_cache_dir /share/yutao/hf_cache/dataset/ \
  #   --use_flash_attention_2 False \
  #   --max_length 2048 \
  #   --batch_size 4 \
  #   --with_description False \
  #   --dataset_name hotpot_qa
done
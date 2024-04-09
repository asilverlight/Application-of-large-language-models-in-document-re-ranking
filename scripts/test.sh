# TOKENIZER_PATH="/share/LMs/models--meta-llama--Llama-2-7b-hf/snapshots/8cca527612d856d7d32bd94f8103728d614eb852/"
# MODEL_PATH="/share/yutao/inters/checkpoint/base_setting_overlap/epoch2-step3000/model"
# RESULT_PATH="result/llama-2-7b"

# 在本文件中运行消融实验
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
  # "/share/yutao/yifei/reranking/checkpoint/epoch2-step30/model"
  # "/share/yutao/inters/checkpoint/chat_setting_no_query_new/epoch2-step2100/model"
  # "/share/yutao/inters/checkpoint/chat_setting_no_dataset_new/epoch2-step2500/model"
  # "/share/yutao/inters/checkpoint/chat_setting_overlap_no_ar_new/epoch2-step3000/model"
  # "/share/yutao/inters/checkpoint/chat_setting_overlap_no_desc_new/epoch2-step3100/model"
  "/share/LMs/models--meta-llama--Llama-2-7b-hf/snapshots/8cca527612d856d7d32bd94f8103728d614eb852/"
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

eval_types=(   
    "/pointwise"
    "/pairwise"
    "/listwise"
    "/listwise-3"
    "/listwise-5"
    "/listwise-10"
)

ablation_types=(
    "/random"
    "/pointwise-only"
    "/pairwise-only"
    "/listwise-only"
    "/out-of-domain-tasks"
    "/out-of-domain-datasets"
    "/listwise-3"
    "/listwise-5"
    "/listwise-10"
    "/listwise-mix"
)

a="/share/yutao/yifei/reranking/eval_ranking/result"

result_paths=(
     "$a${ablation_types[0]}${eval_types[1]}"
     "$a${ablation_types[0]}${eval_types[2]}"

     "$a${ablation_types[1]}${eval_types[0]}"
     "$a${ablation_types[1]}${eval_types[1]}"
     "$a${ablation_types[1]}${eval_types[2]}"

     "$a${ablation_types[2]}${eval_types[0]}"
     "$a${ablation_types[2]}${eval_types[1]}"
     "$a${ablation_types[2]}${eval_types[2]}"

     "$a${ablation_types[3]}${eval_types[0]}"
     "$a${ablation_types[3]}${eval_types[1]}"
     "$a${ablation_types[3]}${eval_types[2]}"

     "$a${ablation_types[4]}${eval_types[0]}"
     "$a${ablation_types[4]}${eval_types[1]}"
     "$a${ablation_types[4]}${eval_types[2]}"

     "$a${ablation_types[5]}${eval_types[0]}"
     "$a${ablation_types[5]}${eval_types[1]}"
     "$a${ablation_types[5]}${eval_types[2]}"

     "$a${ablation_types[6]}${eval_types[3]}"
     "$a${ablation_types[6]}${eval_types[4]}"
     "$a${ablation_types[6]}${eval_types[5]}"

     "$a${ablation_types[7]}${eval_types[3]}"
     "$a${ablation_types[7]}${eval_types[4]}"
     "$a${ablation_types[7]}${eval_types[5]}"

     "$a${ablation_types[8]}${eval_types[3]}"
     "$a${ablation_types[8]}${eval_types[4]}"
     "$a${ablation_types[8]}${eval_types[5]}"

     "$a${ablation_types[9]}${eval_types[3]}"
     "$a${ablation_types[9]}${eval_types[4]}"
     "$a${ablation_types[9]}${eval_types[5]}"
)

# result_paths="/share/yutao/yifei/reranking/eval_ranking/result" + ablation_types[i] + eval_types[j]

with_description=(
  True
  # False
)

batch_size=(
  # 2
  # 4
  8
  # 16
  # 32
)

max_length=(
  # 256
  # 512
  # 1024
  2048
  # 4096
)
rerank_method=(
  # "pointwise"
  # "pairwise"
  "listwise"
)

a="/share/yutao/yifei/reranking/checkpoint"

model_paths=(
    "/share/LMs/models--meta-llama--Llama-2-7b-hf/snapshots/8cca527612d856d7d32bd94f8103728d614eb852/"
    "$a"/finetune/epoch2-step3000/model""

    "$a${ablation_types[1]}"/epoch2-step3000/model""
    "$a${ablation_types[2]}"/epoch2-step3000/model""
    "$a${ablation_types[3]}"/epoch2-step3000/model""

    "$a${ablation_types[4]}"/epoch2-step3000/model""
    "$a${ablation_types[5]}"/epoch2-step3000/model""

    "$a${ablation_types[6]}"/epoch2-step3000/model""
    "$a${ablation_types[7]}"/epoch2-step3000/model""
    "$a${ablation_types[8]}"/epoch2-step3000/model""
    "$a${ablation_types[9]}"/epoch2-step3000/model""
)

for ((i=0; i<${#model_paths[@]}; i++)); do
    echo "${model_paths[i]}"
done
#!/bin/bash

directory="/share/yutao/yifei/reranking/checkpoint/pointwise-only/epoch2-step3000/model"

if [ -d "$directory" ]; then
    echo "目录存在"
else
    echo "目录不存在"
fi

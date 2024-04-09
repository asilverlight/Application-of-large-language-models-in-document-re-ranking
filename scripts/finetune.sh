#!/bin/bash

################
#Load your environments and modules here
################

# torchrun --standalone --nproc_per_node 2 finetune.py \
#     --plugin "gemini" \
#     --dataset "/home/u2020000280/ColossalAI/examples/language/llama2/data/super_natural_instructions_data" \
#     --model_path "/fs/archive/share/jarvis/llama-2-7b-hf" \
#     --save_dir "/home/u2020000280/ColossalAI/examples/language/llama2/saved_models" \
#     --batch_size 4 \
#     --lr 1.0e-5 \
#     --num_epochs 3 \
#     --max_length 512 \
#     --flash_attention \
#     --grad_checkpoint
export OMP_NUM_THREADS=8


# 模型保存路径：

# without-finetune:无
# finetune:checkpoint/finetune/

# 只使用pointwise训练，在三种方式上测试的结果:
# chechpoint/pointwise-only/

# 只使用pairwise训练，在三种方式上测试的结果:
# chechpoint/pairwise-only/

# 只使用listwise训练，在三种方式上测试的结果:
# chechpoint/listwise-only/

# task的out-of-domain测试结果:
# checkpoint/out-of-domain-tasks/

# dataset的out-of-domain测试结果:
# checkpoint/out-of-domain-datasets/

# 训练时，使用listwise的窗口长度为3，并在窗口长度为3,5,10上测试结果:
# checkpoint/listwise-3/

# 训练时，使用listwise的窗口长度为5，并在窗口长度为3,5,10上测试结果:
# checkpoint/listwise-5/

# 训练时，使用listwise的窗口长度为10，并在窗口长度为3,5,10上测试结果:
# checkpoint/listwise-10/

# 训练时，使用listwise的窗口长度为mix，并在窗口长度为3,5,10上测试结果:
# checkpoint/listwise-mix/



# 数据集读取路径：

# without-finetune:无
# finetune:data/train.jsonl

# 只使用pointwise训练，在三种方式上测试的结果:
# data/train_pointwise.jsonl

# 只使用pairwise训练，在三种方式上测试的结果:
# data/train_pairwise.jsonl

# 只使用listwise训练，在三种方式上测试的结果:
# data/train_listwise.jsonl

# task的out-of-domain测试结果:
# data/out_of_domain_tasks.jsonl

# dataset的out-of-domain测试结果:
# data/out_of_domain_datasets.jsonl

# 训练时，使用listwise的窗口长度为3，并在窗口长度为3,5,10上测试结果:
# data/train_listwise_3.jsonl

# 训练时，使用listwise的窗口长度为5，并在窗口长度为3,5,10上测试结果:
# data/train_listwise_5.jsonl

# 训练时，使用listwise的窗口长度为10，并在窗口长度为3,5,10上测试结果:
# data/train_listwise_10.jsonl

# 训练时，使用listwise的窗口长度为mix，并在窗口长度为3,5,10上测试结果:
# data/train_listwise_mix.jsonl

save_path=(
    "/share/yutao/yifei/reranking/checkpoint/finetune/"
    "/share/yutao/yifei/reranking/checkpoint/pointwise-only/"
    "/share/yutao/yifei/reranking/checkpoint/pairwise-only/"
    "/share/yutao/yifei/reranking/checkpoint/listwise-only/"

    "/share/yutao/yifei/reranking/checkpoint/out-of-domain-tasks/"
    "/share/yutao/yifei/reranking/checkpoint/out-of-domain-datasets/"

    "/share/yutao/yifei/reranking/checkpoint/listwise-3/"
    "/share/yutao/yifei/reranking/checkpoint/listwise-5/"
    "/share/yutao/yifei/reranking/checkpoint/listwise-10/"
    "/share/yutao/yifei/reranking/checkpoint/listwise-mix/"
)

data_path=(
    "/share/yutao/yifei/reranking/data/train.jsonl"
    "/share/yutao/yifei/reranking/data/train_pointwise.jsonl"
    "/share/yutao/yifei/reranking/data/train_pairwise.jsonl"
    "/share/yutao/yifei/reranking/data/train_listwise.jsonl"

    "/share/yutao/yifei/reranking/data/out_of_domain_tasks.jsonl"
    "/share/yutao/yifei/reranking/data_mv/out_of_domain_datasets.jsonl"

    "/share/yutao/yifei/reranking/data_mv/train_listwise_3.jsonl"
    "/share/yutao/yifei/reranking/data_mv/train_listwise_5.jsonl"
    "/share/yutao/yifei/reranking/data_mv/train_listwise_10.jsonl"
    "/share/yutao/yifei/reranking/data_mv/train_listwise_mix.jsonl"
)

for ((i=5; i<6; i++)); do
    MODEL_PATH=${save_path[i]}
    DATA_PATH=${data_path[i]}
    torchrun --standalone --nproc_per_node 8 finetune.py \
        --model_path /share/LMs/models--meta-llama--Llama-2-7b-chat-hf/snapshots/c1b0db933684edbfe29a06fa47eb19cc48025e93 \
        --plugin zero2 \
        --max_length 2048 \
        --batch_size 4 \
        --num_epochs 3 \
        --lr 1.0e-5 \
        --flash_attention \
        --mixed_precision bf16 \
        --dataset ${DATA_PATH} \
        --save_dir ${MODEL_PATH} \
        --grad_checkpoint \
        --save_interval 2500
done

# 目前应该是除了最后一个listwise实验，其他都跑了，最后一个时间不够了

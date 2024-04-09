# base
# python3 inference_ensemble.py \
#     --model_name_or_path /share/LMs/models--meta-llama--Llama-2-7b-hf/snapshots/8cca527612d856d7d32bd94f8103728d614eb852 \
#     --test_batch_size 1 \
#     --shots zero_shot
# python3 inference_single.py \
#     --model_name_or_path /share/LMs/models--meta-llama--Llama-2-7b-hf/snapshots/8cca527612d856d7d32bd94f8103728d614eb852 \
#     --test_batch_size 1 \
#     --shots zero_shot

# # chat
# python3 inference_ensemble.py \
#     --model_name_or_path /share/LMs/models--meta-llama--Llama-2-7b-chat-hf/snapshots/c1b0db933684edbfe29a06fa47eb19cc48025e93 \
#     --test_batch_size 1 \
#     --shots zero_shot
# python3 inference_single.py \
#     --model_name_or_path /share/LMs/models--meta-llama--Llama-2-7b-chat-hf/snapshots/c1b0db933684edbfe29a06fa47eb19cc48025e93 \
#     --test_batch_size 1 \
#     --shots zero_shot

# python3 inference_ensemble.py \
#   --model_name_or_path /share/LMs/models--meta-llama--Llama-2-7b-chat-hf/snapshots/c1b0db933684edbfe29a06fa47eb19cc48025e93 \
#   --test_batch_size 1 \
#   --shots zero_shot \
#   --setting setting_overlap_new/prompt_out
# python3 inference_single.py \
#   --model_name_or_path /share/LMs/models--meta-llama--Llama-2-7b-chat-hf/snapshots/c1b0db933684edbfe29a06fa47eb19cc48025e93 \
#   --test_batch_size 1 \
#   --shots zero_shot \
#   --setting setting_overlap_new/prompt_out


# +FLAN
python3 inference_ensemble.py \
    --model_name_or_path /share/yutao/inters/checkpoint/chat_flan/epoch2-step3000/model \
    --test_batch_size 1 \
    --shots zero_shot
# python3 inference_single.py \
#     --model_name_or_path /share/yutao/inters/checkpoint/chat_flan/epoch2-step3000/model \
#     --test_batch_size 1 \
#     --shots zero_shot

# no ret
# python3 inference_ensemble.py \
#     --model_name_or_path /share/yutao/inters/checkpoint/chat_setting_no_ret_new/epoch2-step2100/model \
#     --test_batch_size 1 \
#     --shots zero_shot
# python3 inference_single.py \
#     --model_name_or_path /share/yutao/inters/checkpoint/chat_setting_no_ret_new/epoch2-step2100/model \
#     --test_batch_size 1 \
#     --shots zero_shot

# # no query
# python3 inference_ensemble.py \
#     --model_name_or_path /share/yutao/inters/checkpoint/chat_setting_no_query_new/epoch2-step2100/model \
#     --test_batch_size 1 \
#     --shots zero_shot
# python3 inference_single.py \
#     --model_name_or_path /share/yutao/inters/checkpoint/chat_setting_no_query_new/epoch2-step2100/model \
#     --test_batch_size 1 \
#     --shots zero_shot

# # no dataset
# python3 inference_ensemble.py \
#     --model_name_or_path /share/yutao/inters/checkpoint/chat_setting_no_dataset_new/epoch2-step2500/model \
#     --test_batch_size 1 \
#     --shots zero_shot
# python3 inference_single.py \
#     --model_name_or_path /share/yutao/inters/checkpoint/chat_setting_no_dataset_new/epoch2-step2500/model \
#     --test_batch_size 1 \
#     --shots zero_shot

#  +inters
# for ((i=1; i<10; i++)); do
# python3 inference_ensemble.py \
#   --model_name_or_path /share/yutao/inters/checkpoint/chat_setting_overlap_new/epoch2-step3100/model \
#   --test_batch_size 1 \
#   --shots zero_shot \
#   --setting setting_overlap_new/prompt_out
# python3 inference_single.py \
#   --model_name_or_path /share/yutao/inters/checkpoint/chat_setting_overlap_new/epoch2-step3100/model \
#   --test_batch_size 1 \
#   --shots zero_shot \
#   --setting setting_overlap_new/prompt_out
# done

# no doc
# python3 inference_ensemble.py \
#     --model_name_or_path /share/yutao/inters/checkpoint/chat_setting_no_doc_new/epoch2-step2000/model \
#     --test_batch_size 1 \
#     --shots zero_shot
# python3 inference_single.py \
#     --model_name_or_path /share/yutao/inters/checkpoint/chat_setting_no_doc_new/epoch2-step2000/model \
#     --test_batch_size 1 \
#     --shots zero_shot

# no desc
# python3 inference_ensemble.py \
#     --model_name_or_path /share/yutao/inters/checkpoint/chat_setting_overlap_no_desc_new/epoch2-step3100/model \
#     --test_batch_size 1 \
#     --shots zero_shot
# python3 inference_single.py \
#     --model_name_or_path /share/yutao/inters/checkpoint/chat_setting_overlap_no_desc_new/epoch2-step3100/model \
#     --test_batch_size 1 \
#     --shots zero_shot

# no ar
# python3 inference_ensemble.py \
#     --model_name_or_path /share/yutao/inters/checkpoint/chat_setting_overlap_no_ar_new/epoch2-step3000/model \
#     --test_batch_size 1 \
#     --shots zero_shot
# python3 inference_single.py \
#     --model_name_or_path /share/yutao/inters/checkpoint/chat_setting_overlap_no_ar_new/epoch2-step3000/model \
#     --test_batch_size 1 \
#     --shots zero_shot

# 1-shot
# python3 inference.py \
#     --model_name_or_path /share/yutao/inters/checkpoint/chat_setting_overlap_new/epoch2-step3100/model \
#     --test_batch_size 1 \
#     --shots one_shot

# python3 inference.py \
#     --model_name_or_path /share/yutao/inters/checkpoint/chat_setting_overlap_new/epoch2-step3100/model \
#     --test_batch_size 1 \
#     --shots two_shot

# python3 inference.py \
#     --model_name_or_path /share/yutao/inters/checkpoint/chat_setting_overlap_new/epoch2-step3100/model \
#     --test_batch_size 1 \
#     --shots three_shot

# python3 inference.py \
#     --model_name_or_path /share/yutao/inters/checkpoint/chat_setting_overlap_new/epoch2-step3100/model \
#     --test_batch_size 1 \
#     --shots four_shot

# python3 inference.py \
#     --model_name_or_path /share/yutao/inters/checkpoint/chat_setting_overlap_new/epoch2-step3100/model \
#     --test_batch_size 1 \
#     --shots five_shot
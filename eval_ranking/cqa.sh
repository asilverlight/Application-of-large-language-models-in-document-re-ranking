MODEL_PATH="/share/yutao/inters/checkpoint/chat_setting_overlap_no_desc_new/epoch2-step3100/model"
TOKENIZER_PATH="/share/LMs/models--meta-llama--Llama-2-7b-hf/snapshots/8cca527612d856d7d32bd94f8103728d614eb852"
RESULT_PATH="result/chat-inters-no-desc/cqadupstack"
mkdir "result/chat-inters-no-desc/cqadupstack"

# to gather metrics from all sub-datasets
TMP_PATH="result/chat-inters-no-desc/cqadupstack/tmp.log"

############# MODIFY PATH HERE #############
# containing all sub-dataset folders
CQA_ROOT="/share/yutao/inters/data/ranking/cqadupstack/"
################################################

COUNTER=0
for dataset in $CQA_ROOT/*
do

# get fewshot data files
fewshot_data=($dataset/*train.pt.neg.do-not-overwrite.fewshot*)
fewshot_data=${fewshot_data[0]}

eval_data="$dataset/test.pt.key.do-not-overwrite.json"

############# MODIFY COMMANDS HERE #############
outputString=`torchrun --nproc_per_node 8 eval_rerank.py \
--eval_data $eval_data \
--output_dir $RESULT_PATH \
--model_name_or_path $MODEL_PATH \
--tokenizer_name_or_path $TOKENIZER_PATH \
--hits 10 \
--rerank_method pointwise \
--dataset_name cqadupstack \
--batch_size 4`

# to add 1-shot
# --fewshot_data $fewshot_data \
# --shots 1
################################################

if [[ $COUNTER == 0 ]]
then
echo $outputString > $TMP_PATH
else
echo $outputString >> $TMP_PATH
fi

COUNTER=$[$COUNTER +1]
done

python postprocess_cqa.py -t $TMP_PATH
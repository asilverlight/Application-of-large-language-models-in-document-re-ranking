task_septions=(
    "without-finetune"
    "with-finetune"
    "pointwise-only"
    "pairwise-only"
    "listwise-only"
    "out-of-domain-tasks"
    "out-of-domain-datasets"
    "listwise-3"
    "listwise-5"
    "listwise-10"
    "listwise-mix"
)
task_seption_septions=(
    "pointwise"
    "pairwise"
    "listwise"
)
task_listwise=(
    "listwise-3"
    "listwise-5"
    "listwise-10"
    "listwise-mix"
)
a="/share/yutao/yifei/reranking/eval_ranking/result/"
b="/"
c="metrics.log"
for ((i=0; i<7; i++)); do
    for ((j=0; j<${#task_seption_septions[@]} ; j++)); do
        task_seption="${task_septions[i]}"
        task_seption_seption="${task_seption_septions[j]}"
        input_path="$a$task_seption$b$task_seption_seption$b$c"
        if [ -e "$input_path" ]; then
            python /share/yutao/yifei/reranking/scripts/metrics_log2jsonl.py --log_file $input_path --output_file="$a$task_seption$b$task_seption_seption$b"metrics.jsonl""
        fi
        input_path="$a$task_seption$b$task_seption_seption$b"cqadupstack"$b$c"
        if [ -e "$input_path" ]; then
            python /share/yutao/yifei/reranking/scripts/metrics_log2jsonl.py --log_file $input_path --output_file="$a$task_seption$b$task_seption_seption$b"cqadupstack"$b"metrics.jsonl""
        fi
    done
done

for ((i=7; i<${#task_septions[@]}; i++)); do
    for ((j=0; j<3 ; j++)); do
        task_seption="${task_septions[i]}"
        task_seption_seption="${task_listwise[j]}"
        input_path="$a$task_seption$b$task_seption_seption$c"
        if [ -e "$input_path" ]; then
            python /share/yutao/yifei/reranking/scripts/metrics_log2jsonl.py --log_file $input_path --output_file="$a$task_seption$b$task_seption_seption$b"metrics.jsonl""
        fi
        input_path="$a$task_seption$b$task_seption_seption$b"cqadupstack"$b$c"
        if [ -e "$input_path" ]; then
            python /share/yutao/yifei/reranking/scripts/metrics_log2jsonl.py --log_file $input_path --output_file="$a$task_seption$b$task_seption_seption$b"cqadupstack"$b"metrics.jsonl""
        fi
    done
done
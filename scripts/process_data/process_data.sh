# python reranking_tasks.py --wise_type all
# python reranking_tasks.py --wise_type pointwise 
# python reranking_tasks.py --wise_type pairwise
# python reranking_tasks.py --wise_type listwise --window_size 10
# python reranking_tasks.py --wise_type listwise --window_size 5
# python reranking_tasks.py --wise_type listwise --window_size 3
# python reranking_tasks.py --wise_type listwise --window_size 0


# python data_mixture.py --wise_type all --save_path ../train.jsonl
# python data_mixture.py --wise_type pointwise --save_path ../train_pointwise.jsonl
# # wise消融实验用
# python data_mixture.py --wise_type pairwise --save_path ../train_pairwise.jsonl
# # wise消融实验用
# python data_mixture.py --wise_type listwise --window_size 10 --save_path ../train_listwise.jsonl --with_dbpedia True
# # 第一次训练用
# python data_mixture.py --wise_type listwise --window_size 10 --save_path ../train_listwise_10.jsonl --with_dbpedia False
# # listwise消融实验用
# python data_mixture.py --wise_type listwise --window_size 5 --save_path ../train_listwise_5.jsonl --with_dbpedia False
# # listwise消融实验用    
# python data_mixture.py --wise_type listwise --window_size 3 --save_path ../train_listwise_3.jsonl --with_dbpedia False
# # listwise消融实验用    
python data_mixture.py --wise_type listwise --window_size 0 --save_path ../train_listwise_mix.jsonl --with_dbpedia False
# listwise消融实验用  


# python data_mixture_out_of_domain.py --wise_type all --save_path ../out_of_domain_tasks.jsonl --domain_type task --remove_tasks general_retrieval aricle_retrieval entity_retrieval
# python data_mixture_out_of_domain.py --wise_type all --save_path ../out_of_domain_datasets.jsonl --domain_type dataset --remove_dataset scifact --task_name fact_retrieval
# 默认域外task时，移除general_retrieval、article_retrieval、entity_retrieval；
# 默认域外dataset时，移除scifact，task_name为fact_retrieval。
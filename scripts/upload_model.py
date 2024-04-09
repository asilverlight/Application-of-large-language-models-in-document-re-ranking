from huggingface_hub import HfApi, login

# login()
api = HfApi()
# for i in range(10, 15):
# api.upload_file(
#     path_or_fileobj=f"/share/yutao/inters/checkpoint/chat_setting_overlap_new/epoch2-step3100/model/pytorch_model.bin.index.json",
#     path_in_repo=f"pytorch_model.bin.index.json",
#     repo_id="yutaozhu94/INTERS-LLaMA-7b-chat",
#     repo_type="model",
# )

api = HfApi()
api.upload_folder(
    folder_path="/share/yutao/inters/data/ranking/cqadupstack",
    path_in_repo="/test-qdu/cqadupstack/",
    repo_id="yutaozhu94/INTERS",
    repo_type="dataset",
)
import linecache
import json
from torch.utils.data import Dataset


class FineTuningDataset(Dataset):
    def __init__(self, filename, tokenizer):
        super(FineTuningDataset, self).__init__()
        self._filename = filename
        self._tokenizer = tokenizer
        with open(filename, "r", encoding="utf-8") as f:
            self._total_data = len(f.readlines())
    
    def __getitem__(self, idx):
        line = linecache.getline(self._filename, idx + 1)
        sample = json.loads(line)

        batch = {
            "prompt": sample["prompt"],
            "completion": sample["completion"]
        }

        return batch

    def __len__(self):
        return self._total_data
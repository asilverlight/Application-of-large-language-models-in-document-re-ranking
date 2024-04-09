from abc import ABC, abstractmethod
from collections import OrderedDict

TASK_DATASET_DICT = OrderedDict([
    ("general_retrieval", ["ms_marco"]),
    ("argument_retrieval", ["touche", "arguana"]),
    ("biomedical_retrieval", ["trec_covid", "nfcorpus"]),
    ("article_retrieval", ["scidocs"]),
    ("duplicate_question_retrieval", ["quora", "cqadupstack"]),
    ("entity_retrieval", ["dbpedia"]),
    ("fact_retrieval", ["fever", "climate_fever", "scifact"]),
    ("supporting_evidence_retrieval", ["nq", "fiqa", "hotpot_qa"])
])

class Task(ABC):
    def __init__(self, name, cluster, processed_data_path):
        self._name = name
        self._cluster = cluster
        self._raw_data_path = None
        self._processed_data_path = processed_data_path
        self._has_test_set = False
    
    @abstractmethod
    def load_raw_data(self):
        pass

    @abstractmethod
    def process_raw_data(self):
        pass

    @abstractmethod
    def generate_zero_shot_data(self):
        pass

    @abstractmethod
    def generate_few_shot_data(self, shots):
        pass

    def print_name(self):
        print(f"cluster: {self._cluster}\ttask: {self._name}")
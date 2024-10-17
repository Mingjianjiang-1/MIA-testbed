import json
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import re

class JSONLDummyDataset(Dataset):
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data_index = self._index_data()

    def _index_data(self) -> List[int]:
        index = []
        with open(self.file_path, 'r') as f:
            offset = 0
            for line in f:
                index.append(offset)
                offset += len(line)
        return index

    def __len__(self) -> int:
        return len(self.data_index)

    def __getitem__(self, idx: int) -> str:
        with open(self.file_path, 'r') as f:
            f.seek(self.data_index[idx])
            line = f.readline()
            data = json.loads(line)
            return data.get('text', '')

class MembershipClassifier(ABC):
    @abstractmethod
    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Predict the scores for the given batch of data.
        
        :param data: A NumPy array of input text data
        :return: A NumPy array of scores between 0 and 1
        """
        pass

class YearKeywordClassifier(MembershipClassifier):
    def __init__(self, year_keywords: List[str] = ["2023", "2024"]):
        self.year_keywords = year_keywords
        self.pattern = re.compile(r'\b(' + '|'.join(map(re.escape, year_keywords)) + r')\b')

    def predict(self, data: np.ndarray) -> np.ndarray:
        return np.array([0 if self.pattern.search(text) else 1 for text in data])

def collate_fn(batch: List[str]) -> np.ndarray:
    return np.array(batch)

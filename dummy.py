from typing import List, Dict, Any
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import re
import json

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

def collate_fn(batch: List[str]) -> np.ndarray:
    return np.array(batch)
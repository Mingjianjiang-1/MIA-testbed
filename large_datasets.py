import os
import random
import multiprocessing
from abc import ABC, abstractmethod
import numpy as np
from tqdm import tqdm
import mmap
import pickle
import zstandard as zstd
import io
import multiprocessing
import re
import json
from preprocessors import *

class LargeDataset(ABC):
    def __init__(self, folder_path, saved_rel_path='samples.jsonl', file_regex_filter=r'.*\.jsonl\.zst$'):
        self.folder_path = folder_path
        os.makedirs(os.path.join(folder_path, 'outputs'), exist_ok=True)
        self.saved_path = os.path.join(folder_path, 'outputs', saved_rel_path)
        
        self.preprocessor = self._create_preprocessor(folder_path, file_regex_filter)
        self.total_lines, self.file_weights, self.file_lengths, self.category_total_lines, self.category_file_weights, self.category_file_lengths = self.preprocessor.preprocess_files()

    @abstractmethod
    def _create_preprocessor(self, folder_path, file_regex_filter):
        pass

    @abstractmethod
    def get_line(self, file_name: str, line_number: int, category=None):
        pass

    def get_line_start_offset(self, file_name: str, line_number: int):
        return 0 if line_number == 0 else self.preprocessor.stats_dict[file_name][line_number - 1] + 1

class PileDataset(LargeDataset):
    def _create_preprocessor(self, folder_path, file_regex_filter):
        return PilePreprocessor(folder_path, file_regex_filter)

    def get_line(self, file_name: str, line_number: int, category=None):
        if category is not None:
            real_line_number = self.preprocessor.category_stats_dict[file_name][category][line_number]
        else:
            real_line_number = line_number
            
        filepath = os.path.join(self.folder_path, file_name)
        start_offset = self.get_line_start_offset(file_name, real_line_number)
        end_offset = self.preprocessor.stats_dict[file_name][real_line_number]
        with self.preprocessor.get_file_stream(filepath) as f:
            f.seek(start_offset)
            if end_offset is not None:
                line = f.read(end_offset - start_offset)
            else:
                line = f.readline()
        return line.decode('utf-8').rstrip('\n')

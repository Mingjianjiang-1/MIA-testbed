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
from abc_class import Preprocessor

class PilePreprocessor(Preprocessor):
    @staticmethod
    def get_file_stream(filepath: str):
        return Preprocessor.get_jsonl_file_stream(filepath)
    
    @staticmethod
    def get_line_category(line):
        json_line = json.loads(line)
        return json_line["meta"]["pile_set_name"]

    def _search_newline_positions(self, filepath: str):
        newline_positions = []
        category_line_numbers = {}
        with self.get_jsonl_file_stream(filepath) as f:
            file_size = f.seek(0, io.SEEK_END)
            f.seek(0)
            current_pos, line_number = 0, 0
            with tqdm(total=file_size, unit='B', unit_scale=True, desc="Processing") as pbar:
                while True:
                    line = f.readline()
                    if not line:
                        break
                    line_length = len(line)
                    category = self.get_line_category(line)
                    if category not in category_line_numbers:
                        category_line_numbers[category] = [line_number]
                    else:
                        category_line_numbers[category].append(line_number)
                    
                    newline_positions.append(current_pos + line_length - 1)
                    current_pos += line_length
                    line_number += 1
                    pbar.update(line_length)
                    
        return newline_positions, category_line_numbers

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

class Preprocessor(ABC):
    def __init__(self, folder_path, file_regex_filter=r'.*\.jsonl\.zst$'):
        self.folder_path = folder_path
        self.stats_dict_path = os.path.join(folder_path, f'.stats_dict_regex{file_regex_filter}.pkl')
        self.category_stats_dict_path = os.path.join(folder_path, f'.category_stats_dict_regex{file_regex_filter}.pkl')
        self.file_regex_filter = file_regex_filter
        self.all_files = [f for f in os.listdir(self.folder_path) if os.path.isfile(os.path.join(self.folder_path, f)) and not f.endswith('.pkl') and bool(re.fullmatch(file_regex_filter, f))]
        
        self.stats_dict = {}
        self.category_stats_dict = {}
        self._load_stats()

    def _load_stats(self):
        if os.path.exists(self.stats_dict_path):
            with open(self.stats_dict_path, 'rb') as f:
                self.stats_dict = pickle.load(f)
        if os.path.exists(self.category_stats_dict_path):
            with open(self.category_stats_dict_path, 'rb') as f:
                self.category_stats_dict = pickle.load(f)
    
    @staticmethod
    def get_jsonl_file_stream(filepath: str):
        if filepath.endswith('.zst'):
            with open(filepath, 'rb') as f:
                dctx = zstd.ZstdDecompressor()
                decompressed_data = dctx.stream_reader(f).read()
                return io.BytesIO(decompressed_data)
        else:
            return open(filepath, 'rb')

    @staticmethod
    @abstractmethod
    def get_file_stream(filepath: str):
        pass
    
    @staticmethod
    @abstractmethod
    def get_line_category(line):
        pass

    @abstractmethod
    def _search_newline_positions(self, filepath: str):
        pass

    def _process_file(self, file_name):
        if file_name not in self.stats_dict:
            file_path = os.path.join(self.folder_path, file_name)
            newline_positions, category_line_numbers = self._search_newline_positions(file_path)
            return file_name, newline_positions, category_line_numbers
        return file_name, None, None

    def preprocess_files(self):
        files_to_remove = [key for key in self.stats_dict.keys() if key not in self.all_files]
        for file_name in files_to_remove:
            del self.stats_dict[file_name]
            
        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            results = pool.map(self._process_file, self.all_files)

        total_lines = 0
        file_weights, file_lengths = {}, {}
        category_total_lines = {}
        category_file_weights, category_file_lengths = {}, {}
        
        for file_name, newline_positions, category_line_numbers in results:
            if newline_positions is not None and category_line_numbers is not None:
                self.stats_dict[file_name] = newline_positions
                self.category_stats_dict[file_name] = category_line_numbers

            num_lines = len(self.stats_dict[file_name])
            file_weights[file_name], file_lengths[file_name] = num_lines, num_lines
            total_lines += num_lines
            
            for category in self.category_stats_dict[file_name]:
                if category not in category_total_lines:
                    category_total_lines[category], category_file_weights[category], category_file_lengths[category] = 0, {}, {}
                category_num_lines = len(self.category_stats_dict[file_name][category])  
                category_total_lines[category] += category_num_lines 
                category_file_weights[category][file_name] = category_num_lines
                category_file_lengths[category][file_name] = category_num_lines

        for file_name in file_weights:
            file_weights[file_name] /= total_lines
        
        for category in category_total_lines:
            for file_name in category_file_weights[category]:
                category_file_weights[category][file_name] /= category_total_lines[category]
        
        self._save_stats()
        
        return total_lines, file_weights, file_lengths, category_total_lines, category_file_weights, category_file_lengths

    def _save_stats(self):
        with open(self.stats_dict_path, 'wb') as f:
            pickle.dump(self.stats_dict, f)
        with open(self.category_stats_dict_path, 'wb') as f:
            pickle.dump(self.category_stats_dict, f)


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

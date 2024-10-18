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
from collections import defaultdict
from functools import partial

class Sampler:
    def __init__(self, dataset, saved_folder='outputs'):
        self.dataset = dataset
        os.makedirs(os.path.join(dataset.folder_path, saved_folder), exist_ok=True)
        self.saved_path = os.path.join(self.dataset.folder_path, saved_folder, saved_name)

    # def sample_line_with_seed(self, seed, category=None):
    #     rng = random.Random(seed)
        
    #     assert category is None or category in self.dataset.category_file_weights, f'Category {category} not found in dataset.'
        
    #     if category is None:
    #         file_weights, file_lengths = self.dataset.file_weights, self.dataset.file_lengths
    #     else:
    #         file_weights, file_lengths = self.dataset.category_file_weights[category], self.dataset.category_file_lengths[category]

    #     file_name = rng.choices(
    #         population=list(file_weights.keys()),
    #         weights=list(file_weights.values()),
    #         k=1
    #     )[0]

    #     num_lines = file_lengths[file_name]
    #     line_number = rng.randint(0, num_lines - 1)
    
    #     return self.dataset.get_line(file_name, line_number, category=category)

    # def random_sample(self, num_samples, category=None, saved=False, num_workers=None, seed=42):
    #     if num_workers is None:
    #         num_workers = multiprocessing.cpu_count()
    #     print(f'Using {num_workers} workers for sampling.')
        
    #     rng = random.Random(seed)
    #     seeds = [rng.randint(0, 2**32 - 1) for _ in range(num_samples)]

    #     with multiprocessing.Pool(num_workers) as pool:
    #         if category is None:
    #             async_result = pool.map_async(self.sample_line_with_seed, seeds)
    #         else:
    #             sample_func = partial(self.sample_line_with_seed, category=category)
    #             async_result = pool.map_async(sample_func, seeds)
            
    #         sampled_lines = async_result.get()  # Wait for all processes to complete
           
    #         # if category is None:
    #         #     sampled_lines = pool.map(self.sample_line_with_seed, seeds)
    #         # else:
    #         #     sampled_lines = pool.starmap(self.sample_line_with_seed, [(seed, category) for seed in seeds])
            
    #     if saved:
    #         with open(self.dataset.saved_path, 'w') as f:
    #             for line in sampled_lines:
    #                 f.write(line + '\n')

    #     return sampled_lines

    def sample_lines_with_seed(self, seed, sample_num, category=None):
        rng = random.Random(seed)
        
        assert category is None or category in self.dataset.category_file_weights, f'Category {category} not found in dataset.'
        
        if category is None:
            file_weights, file_lengths = self.dataset.file_weights, self.dataset.file_lengths
        else:
            file_weights, file_lengths = self.dataset.category_file_weights[category], self.dataset.category_file_lengths[category]

        sampled_positions = set()  # To keep track of already sampled positions
        lines_to_fetch = defaultdict(list)  # Dictionary to store file names and line numbers

        while len(sampled_positions) < sample_num:
            file_name = rng.choices(
                population=list(file_weights.keys()),
                weights=list(file_weights.values()),
                k=1
            )[0]

            num_lines = file_lengths[file_name]
            line_number = rng.randint(0, num_lines - 1)
            
            # Check if this position has already been sampled
            position = (file_name, line_number)
            if position not in sampled_positions:
                sampled_positions.add(position)
                lines_to_fetch[file_name].append(line_number)

        sampled_lines = []
        for file_name, line_numbers in lines_to_fetch.items():
            lines = self.dataset.get_lines(file_name, line_numbers, category=category)
            sampled_lines.extend(lines)

        return sampled_lines

    def random_sample_new(self, num_samples, category=None, saved=False, num_workers=None, seed=42, saved_name='samples.jsonl'):
        if num_workers is None:
            num_workers = max(multiprocessing.cpu_count() - 2, 1)
        print(f'Using {num_workers} workers for sampling.')
        
        self.saved_path = os.path.join(self.dataset.folder_path, saved_folder, saved_name)

        rng = random.Random(seed)
        samples_per_worker = num_samples // num_workers
        remainder = num_samples % num_workers
        
        seeds = [rng.randint(0, 2**32 - 1) for _ in range(num_workers)]
        sample_nums = [samples_per_worker + 1 if i < remainder else samples_per_worker for i in range(num_workers)]

        with multiprocessing.Pool(num_workers) as pool:
            if category is None:
                async_result = pool.starmap_async(self.sample_lines_with_seed, zip(seeds, sample_nums))
            else:
                sample_func = partial(self.sample_lines_with_seed, category=category)
                async_result = pool.starmap_async(sample_func, zip(seeds, sample_nums))
            
            sampled_lines_nested = async_result.get()  # Wait for all processes to complete
            sampled_lines = [line for sublist in sampled_lines_nested for line in sublist]  # Flatten the list
            
        if saved:
            with open(self.saved_path, 'w') as f:
                for line in sampled_lines:
                    f.write(line + '\n')

        return sampled_lines
    

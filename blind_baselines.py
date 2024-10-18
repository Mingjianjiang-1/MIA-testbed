import json
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Set
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from tqdm import tqdm
from collections import defaultdict, Counter
from nltk import ngrams
from nltk.tokenize import word_tokenize
import nltk
from multiprocessing import Pool, cpu_count
from functools import partial
import time


class MembershipClassifier(ABC):
    @abstractmethod
    def train(self, member_dataloader: DataLoader, non_member_dataloader: DataLoader):
        """
        Train the classifier using the given data.
        
        :param member_dataloader: A PyTorch DataLoader of member data
        :param non_member_dataloader: A PyTorch DataLoader of non-member data
        """
        pass
    
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

    def train(self, member_dataloader: DataLoader, non_member_dataloader: DataLoader):
        pass
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        return np.array([0 if self.pattern.search(text) else 1 for text in data])


class BagOfWordsClassifier(MembershipClassifier):
    def __init__(self, max_features=5000):
        self.pipeline = Pipeline([
            ('vectorizer', CountVectorizer(max_features=max_features)),
            ('classifier', LogisticRegression(random_state=42))
        ])
    
    def train(self, member_dataloader: DataLoader, non_member_dataloader: DataLoader):
        print("Training Bag of Words classifier...")
        X = []
        y = []
        
        # Process member data
        for batch in tqdm(member_dataloader, desc="Processing member data"):
            X.extend(batch)
            y.extend([1] * len(batch))  # 1 for member
        
        # Process non-member data
        for batch in tqdm(non_member_dataloader, desc="Processing non-member data"):
            X.extend(batch)
            y.extend([0] * len(batch))  # 0 for non-member
        
        # Fit the pipeline
        self.pipeline.fit(X, y)
        print("Training completed.")
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        # The predict_proba method returns probabilities for both classes
        # We take the second column (index 1) which represents the probability of being a member
        return self.pipeline.predict_proba(data)[:, 1]

# # Download necessary NLTK data
# nltk.download('punkt', quiet=True)

# class GreedyRareWordClassifier(MembershipClassifier):
#     def __init__(self, max_ngram_size=5, target_fpr=0.01, min_count=10):
#         self.max_ngram_size = max_ngram_size
#         self.target_fpr = target_fpr
#         self.selected_ngrams = set()
#         self.min_count = min_count

#     def _extract_ngrams_all(self, texts: List[str]) -> Counter:
#         ngram_counter = Counter()
#         for text in tqdm(texts, desc="Extracting n-grams"):
#             tokens = word_tokenize(text.lower())
#             for n in range(1, self.max_ngram_size + 1):
#                 ngram_counter.update(' '.join(gram) for gram in ngrams(tokens, n))
#         return ngram_counter
    
#     def _get_ngrams_set(self, text: str) -> Set[str]:
#         tokens = word_tokenize(text.lower())
#         ngram_set = set()
#         for n in range(1, self.max_ngram_size + 1):
#             ngram_set.update(' '.join(gram) for gram in ngrams(tokens, n))
#         return ngram_set

#     def _calculate_tpr_fpr_ratio(self, ngram: str, member_ngrams: Set[str], non_member_ngrams: Set[str],
#                                  total_member: int, total_non_member: int) -> float:
#         tpr = sum(1 for text_ngrams in member_ngrams if ngram in text_ngrams) / total_member
#         fpr = sum(1 for text_ngrams in non_member_ngrams if ngram in text_ngrams) / total_non_member
#         return tpr / fpr if fpr > 0 else tpr * total_member

#     def train(self, member_dataloader: DataLoader, non_member_dataloader: DataLoader):
#         print("Training Greedy Rare Word classifier...")
        
#         # Collect all texts
#         member_texts = []
#         non_member_texts = []
        
#         for batch in tqdm(member_dataloader, desc="Processing member data"):
#             member_texts.extend(batch)
        
#         for batch in tqdm(non_member_dataloader, desc="Processing non-member data"):
#             non_member_texts.extend(batch)
        
#         member_texts_train, non_member_texts_train = member_texts, non_member_texts
        
#         # Extract all n-grams
#         print("Extracting n-grams...")
#         member_ngrams = self._extract_ngrams_all(member_texts_train)
#         non_member_ngrams = self._extract_ngrams_all(non_member_texts_train)
        
#         # Filter out rare n-grams
#         all_ngrams = set(ngram for ngram, count in member_ngrams.items() if count >= self.min_count)
#         all_ngrams.update(ngram for ngram, count in non_member_ngrams.items() if count >= self.min_count)
        
#         # Precompute sets of n-grams for each text, use tqdm to show a progress bar
#         # member_ngram_sets = [set(self._extract_ngrams([text]).keys()) for text in tqdm(member_texts_train)]
#         member_ngram_sets = [self._get_ngrams_set(text) for text in tqdm(member_texts_train)]
#         non_member_ngram_sets = [self._get_ngrams_set(text) for text in tqdm(non_member_texts_train)]
                
#         # Calculate TPR-to-FPR ratios
#         print("Calculating TPR-to-FPR ratios...")
#         ngram_ratios = {
#             ngram: self._calculate_tpr_fpr_ratio(ngram, member_ngram_sets, non_member_ngram_sets, 
#                                                  len(member_texts_train), len(non_member_texts_train))
#             for ngram in tqdm(all_ngrams)
#         }
        
#         # Sort n-grams by their TPR-to-FPR ratio
#         sorted_ngrams = sorted(ngram_ratios.items(), key=lambda x: x[1], reverse=True)
        
#         # Select n-grams until we hit the target FPR
#         print("Selecting n-grams...")
#         current_fpr = 0
#         for ngram, _ in tqdm(sorted_ngrams):
#             self.selected_ngrams.add(ngram)
#             current_fpr = self._calculate_fpr(non_member_texts_train)
#             print(f"Current FPR: {current_fpr:.4f}")
#             if current_fpr >= self.target_fpr:
#                 break
        
#         print(f"Selected {len(self.selected_ngrams)} n-grams")
        
#         print("Training completed.")

#     def _calculate_tpr(self, texts: List[str]) -> float:
#         return sum(1 for text in texts if any(ngram in text for ngram in self.selected_ngrams)) / len(texts)

#     def _calculate_fpr(self, texts: List[str]) -> float:
#         return sum(1 for text in texts if any(ngram in text for ngram in self.selected_ngrams)) / len(texts)

#     def predict(self, data: np.ndarray) -> np.ndarray:
#         return np.array([1.0 if any(ngram in text for ngram in self.selected_ngrams) else 0.0 for text in data])


nltk.download('punkt', quiet=True)

class GreedyRareWordClassifier(MembershipClassifier):
    def __init__(self, max_ngram_size=5, target_fpr=0.01, min_count=10, timeout=3600):
        self.max_ngram_size = max_ngram_size
        self.target_fpr = target_fpr
        self.selected_ngrams = set()
        self.min_count = min_count
        self.timeout = timeout  # timeout in seconds

    @staticmethod
    def _extract_ngrams_text(text: str, max_ngram_size: int) -> Counter:
        tokens = word_tokenize(text.lower())
        ngram_counter = Counter()
        for n in range(1, max_ngram_size + 1):
            ngram_counter.update(' '.join(gram) for gram in ngrams(tokens, n))
        return ngram_counter

    def _extract_ngrams_all(self, texts: List[str]) -> Counter:
        print("Extracting n-grams...")
        total_counter = Counter()
        with Pool(processes=cpu_count()) as pool:
            try:
                for result in tqdm(
                    pool.imap_unordered(partial(self._extract_ngrams_text, max_ngram_size=self.max_ngram_size), texts),
                    total=len(texts),
                    desc="Extracting n-grams"
                ):
                    total_counter.update(result)
            except Exception as e:
                print(f"Error in n-gram extraction: {e}")
        return total_counter

    @staticmethod
    def _get_ngrams_set(text: str, max_ngram_size: int) -> Set[str]:
        tokens = word_tokenize(text.lower())
        ngram_set = set()
        for n in range(1, max_ngram_size + 1):
            ngram_set.update(' '.join(gram) for gram in ngrams(tokens, n))
        return ngram_set

    @staticmethod
    def _calculate_tpr_fpr_ratio_chunk(args):
        chunk, member_ngrams, non_member_ngrams, total_member, total_non_member = args
        results = {}
        for ngram in chunk:
            tpr = sum(1 for text_ngrams in member_ngrams if ngram in text_ngrams) / total_member
            fpr = sum(1 for text_ngrams in non_member_ngrams if ngram in text_ngrams) / total_non_member
            results[ngram] = tpr / fpr if fpr > 0 else tpr * total_member
        return results

    def train(self, member_dataloader: DataLoader, non_member_dataloader: DataLoader):
        print("Training Greedy Rare Word classifier...")
        
        # Collect all texts
        member_texts = []
        non_member_texts = []
        
        for batch in tqdm(member_dataloader, desc="Processing member data"):
            member_texts.extend(batch)
        
        for batch in tqdm(non_member_dataloader, desc="Processing non-member data"):
            non_member_texts.extend(batch)
        
        member_texts_train, non_member_texts_train = member_texts, non_member_texts
        
        member_ngrams = self._extract_ngrams_all(member_texts_train)
        non_member_ngrams = self._extract_ngrams_all(non_member_texts_train)
        
        all_ngrams = set(ngram for ngram, count in member_ngrams.items() if count >= self.min_count)
        all_ngrams.update(ngram for ngram, count in non_member_ngrams.items() if count >= self.min_count)
        
        print("Precomputing n-gram sets...")
        with Pool(processes=cpu_count()) as pool:
            member_ngram_sets = list(tqdm(
                pool.imap_unordered(partial(self._get_ngrams_set, max_ngram_size=self.max_ngram_size), member_texts_train),
                total=len(member_texts_train),
                desc="Processing member texts"
            ))
            non_member_ngram_sets = list(tqdm(
                pool.imap_unordered(partial(self._get_ngrams_set, max_ngram_size=self.max_ngram_size), non_member_texts_train),
                total=len(non_member_texts_train),
                desc="Processing non-member texts"
            ))
        
        print("Calculating TPR-to-FPR ratios...")
        print("There are {} n-grams to process.".format(len(all_ngrams)))
        chunk_size = max(1, len(all_ngrams) // (cpu_count() * 4))  # Smaller chunks for better load balancing
        ngram_chunks = [list(all_ngrams)[i:i + chunk_size] for i in range(0, len(all_ngrams), chunk_size)]
        
        ngram_ratios = {}
        for chunk in tqdm(ngram_chunks, desc="Calculating ratios"):
            results = self._calculate_tpr_fpr_ratio_chunk(
                (chunk, member_ngram_sets, non_member_ngram_sets, len(member_texts_train), len(non_member_texts_train))
            )
            ngram_ratios.update(results) 
        
        sorted_ngrams = sorted(ngram_ratios.items(), key=lambda x: x[1], reverse=True)
        
        print("Selecting n-grams...")
        current_fpr = 0
        for ngram, _ in tqdm(sorted_ngrams):
            self.selected_ngrams.add(ngram)
            current_fpr = self._calculate_fpr(non_member_texts_train)
            if current_fpr >= self.target_fpr:
                break
        
        print(f"Selected {len(self.selected_ngrams)} n-grams")
        print("Training completed.")

    def _calculate_fpr(self, texts: List[str]) -> float:
        return sum(1 for text in texts if any(ngram in text for ngram in self.selected_ngrams)) / len(texts)

    def predict(self, data: np.ndarray) -> np.ndarray:
        return np.array([1.0 if any(ngram in text for ngram in self.selected_ngrams) else 0.0 for text in data])

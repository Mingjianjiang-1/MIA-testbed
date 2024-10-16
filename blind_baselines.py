import json
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from tqdm import tqdm

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

# # Usage example
# bow_classifier = BagOfWordsClassifier(max_features=5000)
# bow_classifier.train(member_dataloader, non_member_dataloader)

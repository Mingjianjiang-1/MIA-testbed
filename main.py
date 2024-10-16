from torch.utils.data import Dataset, DataLoader
from dummy import JSONLDummyDataset, collate_fn
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from typing import List, Tuple
from tqdm import tqdm
from blind_baselines import MembershipClassifier, YearKeywordClassifier

def create_dataloaders(member_path, non_member_path, batch_size=32, num_workers=4):
    member_dataset = JSONLDummyDataset(member_path)
    non_member_dataset = JSONLDummyDataset(non_member_path)
    
    member_dataloader = DataLoader(
        member_dataset, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        collate_fn=collate_fn,
        shuffle=True
    )
    non_member_dataloader = DataLoader(
        non_member_dataset, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        collate_fn=collate_fn,
        shuffle=True
    )
    
    return member_dataloader, non_member_dataloader

def evaluate_classifiers(classifiers: List[MembershipClassifier], 
                         member_dataloader: DataLoader, 
                         non_member_dataloader: DataLoader) -> List[Tuple[str, float]]:
    results = []
    
    for classifier in classifiers:
        all_scores = []
        all_labels = []
        
        # Process member data
        for batch in tqdm(member_dataloader, desc=f"Evaluating {classifier.__class__.__name__} on member data"):
            scores = classifier.predict(batch)
            all_scores.extend(scores)
            all_labels.extend([1] * len(scores))  # 1 for member
        
        # Process non-member data
        for batch in tqdm(non_member_dataloader, desc=f"Evaluating {classifier.__class__.__name__} on non-member data"):
            scores = classifier.predict(batch)
            all_scores.extend(scores)
            all_labels.extend([0] * len(scores))  # 0 for non-member
        
        # Calculate AUROC
        auroc = roc_auc_score(all_labels, all_scores)
        results.append((classifier.__class__.__name__, auroc))
    
    return results

# Usage
member_dataloader, non_member_dataloader = create_dataloaders(
    member_path="test_data/test10k-pile-train-00.jsonl",
    non_member_path="test_data/test10k-pile-val.jsonl"
)

# Usage example
classifiers = [
    YearKeywordClassifier(["2023", "2024"]),
    # Add other classifiers here
]

results = evaluate_classifiers(classifiers, member_dataloader, non_member_dataloader)

for classifier_name, auroc in results:
    print(f"{classifier_name} AUROC: {auroc:.4f}")
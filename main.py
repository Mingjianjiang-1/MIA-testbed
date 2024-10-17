import numpy as np
from sklearn.metrics import roc_auc_score
from typing import List, Tuple
from tqdm import tqdm
from blind_baselines import MembershipClassifier, YearKeywordClassifier, BagOfWordsClassifier
from torch.utils.data import Dataset, DataLoader, random_split
from dummy import JSONLDummyDataset, collate_fn
from sampler import Sampler
from large_datasets import PileDataset
import torch

def create_dataloaders(member_path, non_member_path, batch_size=32, num_workers=4, test_split=0.2, seed=42):
    member_dataset = JSONLDummyDataset(member_path)
    non_member_dataset = JSONLDummyDataset(non_member_path)
    
    # Calculate split sizes
    member_test_size = int(len(member_dataset) * test_split)
    member_train_size = len(member_dataset) - member_test_size
    non_member_test_size = int(len(non_member_dataset) * test_split)
    non_member_train_size = len(non_member_dataset) - non_member_test_size
    
    # Split datasets
    train_member_dataset, test_member_dataset = random_split(
        member_dataset, [member_train_size, member_test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    train_non_member_dataset, test_non_member_dataset = random_split(
        non_member_dataset, [non_member_train_size, non_member_test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Create DataLoaders
    train_member_dataloader = DataLoader(
        train_member_dataset, batch_size=batch_size, num_workers=num_workers, 
        collate_fn=collate_fn, shuffle=True
    )
    train_non_member_dataloader = DataLoader(
        train_non_member_dataset, batch_size=batch_size, num_workers=num_workers, 
        collate_fn=collate_fn, shuffle=True
    )
    test_member_dataloader = DataLoader(
        test_member_dataset, batch_size=batch_size, num_workers=num_workers, 
        collate_fn=collate_fn
    )
    test_non_member_dataloader = DataLoader(
        test_non_member_dataset, batch_size=batch_size, num_workers=num_workers, 
        collate_fn=collate_fn
    )
    
    return (train_member_dataloader, train_non_member_dataloader,
            test_member_dataloader, test_non_member_dataloader)

def evaluate_classifiers(classifiers: List[MembershipClassifier], 
                         member_dataloader: DataLoader, 
                         non_member_dataloader: DataLoader,
                         dataset_name: str) -> List[Tuple[str, float]]:
    results = []
    
    for classifier in classifiers:
        all_scores = []
        all_labels = []
        
        # Process member data
        for batch in tqdm(member_dataloader, desc=f"Evaluating {classifier.__class__.__name__} on {dataset_name} member data"):
            scores = classifier.predict(batch)
            all_scores.extend(scores)
            all_labels.extend([1] * len(scores))  # 1 for member
        
        # Process non-member data
        for batch in tqdm(non_member_dataloader, desc=f"Evaluating {classifier.__class__.__name__} on {dataset_name} non-member data"):
            scores = classifier.predict(batch)
            all_scores.extend(scores)
            all_labels.extend([0] * len(scores))  # 0 for non-member
        
        # Calculate AUROC
        auroc = roc_auc_score(all_labels, all_scores)
        results.append((classifier.__class__.__name__, auroc))
    
    return results

def train_classifiers(classifiers, train_member_dataloader, train_non_member_dataloader):
    for classifier in classifiers:
        print(f"{classifier.__class__.__name__} starts training.")
        classifier.train(train_member_dataloader, train_non_member_dataloader)
        print(f"{classifier.__class__.__name__} completed training.")

def sample_from_pile_data(sample_num):
    # member_dataset = PileDataset("data/pile", file_regex_filter=r'train/[^/]+\.jsonl\.zst$')
    non_member_dataloader = PileDataset("data/pile", file_regex_filter=r'^[^/]+\.jsonl\.zst$')

    member_sampler = Sampler(member_dataset)
    non_member_sampler = Sampler(non_member_dataset)

    member_sampler.random_sample_new(sample_num, saved=True, num_workers=8, seed=42, saved_name='member_samples.jsonl')
    non_member_sampler.random_sample_new(sample_num, saved=True, num_workers=8, seed=42, saved_name='non_member_samples.jsonl')

def benchmark_blind_baselines():
    train_member_dataloader, train_non_member_dataloader, \
    test_member_dataloader, test_non_member_dataloader = create_dataloaders(
        "test_data/test10k-pile-train-00.jsonl",
        "test_data/test10k-pile-val.jsonl"
    )

    # Initialize classifiers
    classifiers = [
        YearKeywordClassifier(["2023", "2024"]),
        BagOfWordsClassifier(max_features=5000)
    ]

    # Train classifiers
    train_classifiers(classifiers, train_member_dataloader, train_non_member_dataloader)

    # Evaluate on test set
    test_results = evaluate_classifiers(classifiers, test_member_dataloader, test_non_member_dataloader, "Test")

    # Print results
    print("\nTest Results:")
    for classifier_name, auroc in test_results:
        print(f"{classifier_name} AUROC: {auroc:.4f}")

if __name__ == "__main__":
    # benchmark_blind_baselines()
    sample_from_pile_data(20)
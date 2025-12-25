import numpy as np
import pandas as pd

class EraKFold:
    """
    Era-aware K-Fold cross-validation splitter.
    Ensures that eras are kept together and respects an embargo period between train and test.
    """
    def __init__(self, n_splits=5, embargo=4):
        self.n_splits = n_splits
        self.embargo = embargo

    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("Groups (eras) must be provided for EraKFold")
        
        unique_eras = np.unique(groups)
        n_eras = len(unique_eras)
        
        # Split eras into K folds
        fold_size = n_eras // self.n_splits
        
        for i in range(self.n_splits):
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i != self.n_splits - 1 else n_eras
            
            test_eras = unique_eras[test_start:test_end]
            
            # Determine train eras with embargo
            # Train must not include test_eras OR eras within 'embargo' distance from test_eras
            test_indices_in_unique = np.arange(test_start, test_end)
            embargo_indices = []
            for idx in test_indices_in_unique:
                embargo_indices.extend(range(max(0, idx - self.embargo), min(n_eras, idx + self.embargo + 1)))
            
            embargo_indices = np.unique(embargo_indices)
            train_era_indices = np.setdiff1d(np.arange(n_eras), embargo_indices)
            train_eras = unique_eras[train_era_indices]
            
            # Convert eras back to row indices
            train_idx = np.where(np.isin(groups, train_eras))[0]
            test_idx = np.where(np.isin(groups, test_eras))[0]
            
            yield train_idx, test_idx

if __name__ == "__main__":
    # Simple test
    groups = np.repeat(np.arange(20), 5) # 20 eras, 5 rows each
    ekf = EraKFold(n_splits=5, embargo=2)
    for train_idx, test_idx in ekf.split(np.zeros(len(groups)), groups=groups):
        train_eras = np.unique(groups[train_idx])
        test_eras = np.unique(groups[test_idx])
        print(f"Train Eras: {len(train_eras)}, Test Eras: {len(test_eras)}, Gap: {np.min(np.abs(np.subtract.outer(train_eras, test_eras)))}")

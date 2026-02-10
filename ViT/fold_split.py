#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset

TENSOR_PATH = '/workspace/BrainMRI-3way-classification/data/3D_tensors'
CSV_PATH = '/workspace/BrainMRI-3way-classification/data/csv_splits/all_mri_scan_list.csv'
SEED = 37
N_FOLDS = 5
LABEL_MAP = {'CN': 0, 'MCI': 1, 'AD': 2}


def _load_patient_groups():
    """Read CSV, group tensor paths by patient, assign stratification label."""
    df = pd.read_csv(CSV_PATH)

    patient_images = defaultdict(list)
    for idx, row in df.iterrows():
        tensor_path = os.path.join(TENSOR_PATH, row['label'], f'{idx}.pt')
        patient_images[row['patient_id']].append({
            'path': tensor_path,
            'label': row['label']
        })

    severity = {'CN': 0, 'MCI': 1, 'AD': 2}
    patient_ids = []
    patient_labels = []
    for pid, images in patient_images.items():
        patient_ids.append(pid)
        max_label = max(images, key=lambda x: severity[x['label']])['label']
        patient_labels.append(max_label)

    return patient_images, np.array(patient_ids), np.array(patient_labels)


def get_fold(fold_num, val_ratio=0.2):
    """
    Patient-wise stratified 5-fold split.

    Args:
        fold_num: 0-4, which fold to use as test set
        val_ratio: fraction of remaining train patients used as validation
                   (0.2 means overall ~16% val, ~64% train, ~20% test)

    Returns:
        train_files, val_files, test_files
        each is a list of (tensor_path, label_int)
    """
    patient_images, patient_ids, patient_labels = _load_patient_groups()

    outer_skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    for i, (train_val_idx, test_idx) in enumerate(outer_skf.split(patient_ids, patient_labels)):
        if i != fold_num:
            continue

        test_pids = set(patient_ids[test_idx])
        train_val_pids = patient_ids[train_val_idx]
        train_val_labels = patient_labels[train_val_idx]

        # Inner split: separate validation from training
        inner_n_splits = round(1.0 / val_ratio)
        inner_skf = StratifiedKFold(n_splits=inner_n_splits, shuffle=True, random_state=SEED)
        train_inner_idx, val_inner_idx = next(inner_skf.split(train_val_pids, train_val_labels))

        train_pids = set(train_val_pids[train_inner_idx])
        val_pids = set(train_val_pids[val_inner_idx])

        def collect_files(pids):
            files = []
            for pid in pids:
                for img in patient_images[pid]:
                    files.append((img['path'], LABEL_MAP[img['label']]))
            return files

        return collect_files(train_pids), collect_files(val_pids), collect_files(test_pids)

    raise ValueError(f"fold_num must be 0 to {N_FOLDS - 1}")


class FoldDataset(Dataset):
    """Dataset that loads pre-saved .pt tensors from a file list."""

    def __init__(self, file_list):
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path, label = self.file_list[idx]
        tensor = torch.load(path, weights_only=True)
        return tensor, label

    def label_dist(self):
        label_names = {0: 'CN', 1: 'MCI', 2: 'AD'}
        dist = {'CN': 0, 'MCI': 0, 'AD': 0}
        for _, label in self.file_list:
            dist[label_names[label]] += 1
        return dist

#!/usr/bin/env python
# coding: utf-8
"""
Spatial normalization 검증:
- 여러 샘플의 nonzero bounding box 비교 (뇌 위치가 일관되는지)
- 축별 nonzero 시작/끝 인덱스 비교
- 중앙 슬라이스의 nonzero 패턴 비교
"""

import torch
import os
import random
import numpy as np

random.seed(42)

BASE_DIR = '/workspace/BrainMRI-3way-classification/data/3D_tensors'
SPLITS = ['Train', 'Val', 'Test']
CLASSES = ['AD', 'CN', 'MCI']
SAMPLES_PER_FOLDER = 2

print("=" * 80)
print("Spatial Normalization 검증: 뇌 위치/정렬 일관성 확인")
print("=" * 80)

all_bboxes = []  # (split, cls, fname, bbox_dict)

for split in SPLITS:
    for cls in CLASSES:
        folder = os.path.join(BASE_DIR, split, cls)
        files = [f for f in os.listdir(folder) if f.endswith('.pt')]
        sampled = random.sample(files, min(SAMPLES_PER_FOLDER, len(files)))

        print(f"\n{'─' * 70}")
        print(f"[{split}/{cls}] {len(sampled)}개 샘플")
        print(f"{'─' * 70}")

        for fname in sampled:
            fpath = os.path.join(folder, fname)
            data = torch.load(fpath, weights_only=True).squeeze(0)  # (192,192,192)

            mask = (data > 0).numpy()  # 뇌 영역 마스크 (배경=0 또는 음수)

            # 각 축별로 nonzero가 존재하는 범위 (bounding box)
            bbox = {}
            axis_names = ['D (depth/axial)', 'H (height/coronal)', 'W (width/sagittal)']
            for ax, ax_name in enumerate(axis_names):
                proj = mask.any(axis=tuple(i for i in range(3) if i != ax))
                indices = np.where(proj)[0]
                if len(indices) > 0:
                    bbox[ax_name] = (int(indices[0]), int(indices[-1]), int(indices[-1] - indices[0] + 1))
                else:
                    bbox[ax_name] = (0, 0, 0)

            all_bboxes.append((split, cls, fname, bbox))

            # 중앙 슬라이스의 nonzero 비율
            mid = data.shape[0] // 2
            mid_slice = data[mid]
            mid_nz = (mid_slice > 0).sum().item()
            mid_total = mid_slice.numel()

            # center of mass (질량 중심)
            coords = np.argwhere(mask)
            if len(coords) > 0:
                com = coords.mean(axis=0)
            else:
                com = np.array([0, 0, 0])

            print(f"\n  {fname}")
            for ax_name, (start, end, span) in bbox.items():
                print(f"    {ax_name}: [{start} ~ {end}] span={span}")
            print(f"    center of mass: ({com[0]:.1f}, {com[1]:.1f}, {com[2]:.1f})")
            print(f"    mid axial slice nonzero: {mid_nz}/{mid_total} ({100*mid_nz/mid_total:.1f}%)")

# ── 전체 요약: bounding box 일관성 ──
print(f"\n{'=' * 80}")
print("Spatial Normalization 요약")
print(f"{'=' * 80}")

axis_names = ['D (depth/axial)', 'H (height/coronal)', 'W (width/sagittal)']
for ax_name in axis_names:
    starts = [b[ax_name][0] for _, _, _, b in all_bboxes]
    ends   = [b[ax_name][1] for _, _, _, b in all_bboxes]
    spans  = [b[ax_name][2] for _, _, _, b in all_bboxes]
    print(f"\n  {ax_name}:")
    print(f"    start  범위: {min(starts)} ~ {max(starts)}  (mean={np.mean(starts):.1f}, std={np.std(starts):.1f})")
    print(f"    end    범위: {min(ends)} ~ {max(ends)}  (mean={np.mean(ends):.1f}, std={np.std(ends):.1f})")
    print(f"    span   범위: {min(spans)} ~ {max(spans)}  (mean={np.mean(spans):.1f}, std={np.std(spans):.1f})")

# center of mass 일관성
print(f"\n  Center of Mass 일관성:")
all_coms = []
for split, cls, fname, bbox in all_bboxes:
    fpath = os.path.join(BASE_DIR, split, cls, fname)
    data = torch.load(fpath, weights_only=True).squeeze(0).numpy()
    coords = np.argwhere(data > 0)
    if len(coords) > 0:
        all_coms.append(coords.mean(axis=0))

all_coms = np.array(all_coms)
print(f"    D: mean={all_coms[:,0].mean():.1f}, std={all_coms[:,0].std():.1f}")
print(f"    H: mean={all_coms[:,1].mean():.1f}, std={all_coms[:,1].std():.1f}")
print(f"    W: mean={all_coms[:,2].mean():.1f}, std={all_coms[:,2].std():.1f}")

com_std_max = all_coms.std(axis=0).max()
if com_std_max < 5:
    print(f"\n  --> 뇌 위치 매우 일관됨 (CoM std max={com_std_max:.2f} voxels)")
    print(f"      MNI152 spatial normalization 잘 되어 있음")
elif com_std_max < 10:
    print(f"\n  --> 뇌 위치 대체로 일관됨 (CoM std max={com_std_max:.2f} voxels)")
    print(f"      spatial normalization 되어 있으나 약간의 변동 있음")
else:
    print(f"\n  --> 뇌 위치 불일관! (CoM std max={com_std_max:.2f} voxels)")
    print(f"      spatial normalization 안 되어 있거나 문제 있을 수 있음")

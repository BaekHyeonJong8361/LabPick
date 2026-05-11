"""
train_v7.py
===========
v7: FF++ + DFDC 통합 학습 (데이터 다양성 효과 측정)
====================================================

v6 대비 변경점 (변수 통제):
  - 학습 데이터: train_ff_only.csv → train_combined.csv (FF++ + DFDC)
  - Sampler: 2-way (REAL/FAKE) → 4-way (REAL/FAKE × FF/DFDC)
  - 그 외 augmentation, 모델, loss, 하이퍼파라미터 모두 v6과 동일

목적:
  - v6 vs v7 비교를 통해 "DFDC를 학습에 넣은 효과"를 통제 실험으로 측정
  - cross-dataset gap 0.2766이 얼마나 좁혀지는가 측정
"""

import os
import math
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, balanced_accuracy_score
import timm

from dataset import FastDeepfakeDataset

# v6에서 만든 augmentation/transform 재사용
from train_v6 import (
    BinaryFocalLoss,
    seed_worker,
    safe_auc,
    evaluate,
    warmup_cosine,
    get_v6_train_transforms,
    get_eval_transforms,
)


# =========================================================
# v7 신규: 4-way 균형 sampler (class × dataset)
# =========================================================
def make_4way_balanced_sampler(csv_path):
    """
    학습 시 매 batch에서 (FF-REAL, FF-FAKE, DFDC-REAL, DFDC-FAKE)
    4그룹이 균등하게 뽑히도록 가중치 부여.

    원리:
      - 각 (dataset, label) 조합별 샘플 수 카운트
      - sample_weight = 1 / count_in_group
      - 그룹 내 모든 샘플은 동일 가중치, 그룹 간은 크기 반비례
    """
    df = pd.read_csv(csv_path)
    if "dataset" not in df.columns:
        raise ValueError(f"{csv_path}에 'dataset' 컬럼이 없습니다 (combined CSV 필요)")

    labels = df["label"].astype(int).values
    datasets = df["dataset"].astype(str).values  # 'ff' 또는 'dfdc'

    # 4-way 그룹 키 (예: 'ff_0', 'ff_1', 'dfdc_0', 'dfdc_1')
    group_keys = np.array([f"{d}_{l}" for d, l in zip(datasets, labels)])
    unique_groups, group_counts = np.unique(group_keys, return_counts=True)

    print(f"  [4-way Sampler] 그룹별 샘플 수:")
    for g, c in zip(unique_groups, group_counts):
        print(f"    {g}: {c:,}")

    # 그룹별 가중치
    group_weight_map = {g: 1.0 / c for g, c in zip(unique_groups, group_counts)}
    sample_weights = np.array([group_weight_map[g] for g in group_keys], dtype=np.float64)

    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False  # cuDNN 심볼 충돌 우회


# =========================================================
# Main
# =========================================================
def train():
    set_seed(42)

    # GPU 1번만 사용
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        device = torch.device("cuda:1")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"

    print(f"[v7] FF++ + DFDC 통합 학습 | device={device}")

    # 하이퍼파라미터 (v6과 동일 - 변수 통제)
    batch_size = 64
    num_epochs = 30
    base_lr = 1e-4
    weight_decay = 1e-4
    patience = 6
    warmup_epochs = 2
    num_workers = min(8, os.cpu_count() or 4)

    save_dir = "saved_models"
    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, "xception_best_v7_combined.pth")

    splits_dir = "splits_v6"
    train_csv = os.path.join(splits_dir, "train_combined.csv")     # ⬅ v6과 다른 점
    val_csv = os.path.join(splits_dir, "val_combined.csv")         # ⬅ val도 통합
    test_ff_csv = os.path.join(splits_dir, "test_ff_only.csv")
    test_dfdc_csv = os.path.join(splits_dir, "test_dfdc_only.csv")

    # 데이터셋
    print("📁 데이터 로딩...")
    train_ds = FastDeepfakeDataset(train_csv, transform=get_v6_train_transforms(), label_dtype=torch.float32)
    val_ds = FastDeepfakeDataset(val_csv, transform=get_eval_transforms(), label_dtype=torch.float32)
    test_ff_ds = FastDeepfakeDataset(test_ff_csv, transform=get_eval_transforms(), label_dtype=torch.float32)
    test_dfdc_ds = FastDeepfakeDataset(test_dfdc_csv, transform=get_eval_transforms(), label_dtype=torch.float32)

    print(f"  train: {len(train_ds):,} | val: {len(val_ds):,} | test_ff: {len(test_ff_ds):,} | test_dfdc: {len(test_dfdc_ds):,}")

    # 4-way balanced sampler
    print("\n🎯 4-way balanced sampler 구성:")
    sampler = make_4way_balanced_sampler(train_csv)

    common_loader = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        persistent_workers=(num_workers > 0),
    )
    train_loader = DataLoader(train_ds, sampler=sampler, **common_loader)
    val_loader = DataLoader(val_ds, shuffle=False, **common_loader)
    test_ff_loader = DataLoader(test_ff_ds, shuffle=False, **common_loader)
    test_dfdc_loader = DataLoader(test_dfdc_ds, shuffle=False, **common_loader)

    # 모델
    print("\n🤖 Xception 로딩...")
    model = timm.create_model("xception", pretrained=True, num_classes=1).to(device)

    criterion = BinaryFocalLoss(gamma=2.0, alpha=0.25)
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)

    best_score = -1.0
    early_stop = 0

    print(f"\n{'='*60}")
    print(f"[v7] lr={base_lr}, wd={weight_decay}, bs={batch_size}, epochs={num_epochs}")
    print(f"[v7] Loss=Focal(gamma=2.0, alpha=0.25)")
    print(f"[v7] Aug=Flip+CJ+RandomResize+Blur+Noise+JPEG40-95 (v6과 동일)")
    print(f"[v7] Sampler=4-way (FF/DFDC × REAL/FAKE)")
    print(f"[v7] 데이터: FF++ + DFDC combined (변수 통제: aug/모델/loss는 v6과 동일)")
    print(f"{'='*60}\n")

    for epoch in range(num_epochs):
        model.train()
        loss_sum = 0.0
        cur_lr = warmup_cosine(optimizer, epoch, warmup_epochs, num_epochs, base_lr)

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            loss_sum += loss.item() * images.size(0)

        avg_train = loss_sum / len(train_loader.dataset)
        val_m = evaluate(model, val_loader, criterion, device, "VAL")

        print(f"[Epoch {epoch+1}/{num_epochs}] lr={cur_lr:.2e}")
        print(f"  Train Loss: {avg_train:.4f}")
        print(f"  Val   Loss: {val_m['loss']:.4f} | Acc: {val_m['acc']*100:.2f}% | F1: {val_m['f1']:.4f} | BalAcc: {val_m['bal_acc']:.4f} | AUC: {val_m['auc']:.4f}")

        score = val_m["auc"]
        if score > best_score:
            print(f"  ⭐ best 갱신! ({best_score:.4f} → {score:.4f})")
            best_score = score
            early_stop = 0
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_auc": best_score,
                "val_metrics": val_m,
            }, best_path)
        else:
            early_stop += 1
            if early_stop >= patience:
                print(f"  🛑 {patience}회 연속 개선 없음 → 조기 종료")
                break

    # =========================================================
    # 베스트 모델 평가
    # =========================================================
    print(f"\n{'='*60}\n📦 베스트 모델 로드 & 최종 평가")
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    print("\n[ TEST: FF++ (in-domain after combined training) ]")
    m_ff = evaluate(model, test_ff_loader, criterion, device, "TEST_FF")
    print(f"  AUC: {m_ff['auc']:.4f} | Acc: {m_ff['acc']*100:.2f}% | F1: {m_ff['f1']:.4f} | BalAcc: {m_ff['bal_acc']:.4f}")

    print("\n[ TEST: DFDC (in-domain after combined training) ]")
    m_dfdc = evaluate(model, test_dfdc_loader, criterion, device, "TEST_DFDC")
    print(f"  AUC: {m_dfdc['auc']:.4f} | Acc: {m_dfdc['acc']*100:.2f}% | F1: {m_dfdc['f1']:.4f} | BalAcc: {m_dfdc['bal_acc']:.4f}")

    print(f"\n[ Cross-dataset gap ]")
    print(f"  AUC delta (FF - DFDC): {m_ff['auc'] - m_dfdc['auc']:+.4f}")

    # v6과의 비교
    print(f"\n[ v6 vs v7 비교 ]")
    print(f"  v6 (FF only):     FF AUC 0.9626 | DFDC AUC 0.6860 | gap 0.2766")
    print(f"  v7 (FF + DFDC):   FF AUC {m_ff['auc']:.4f} | DFDC AUC {m_dfdc['auc']:.4f} | gap {m_ff['auc']-m_dfdc['auc']:+.4f}")
    print(f"  → DFDC AUC 변화: {m_dfdc['auc'] - 0.6860:+.4f}")

    # 결과 저장
    results = {
        "version": "v7_combined",
        "best_val_auc": float(best_score),
        "test_ff": m_ff,
        "test_dfdc": m_dfdc,
        "cross_gap_auc": float(m_ff["auc"] - m_dfdc["auc"]),
        "v6_comparison": {
            "v6_test_ff_auc": 0.9626,
            "v6_test_dfdc_auc": 0.6860,
            "v6_gap": 0.2766,
            "v7_dfdc_improvement": float(m_dfdc["auc"] - 0.6860),
            "v7_gap_reduction": float(0.2766 - (m_ff["auc"] - m_dfdc["auc"])),
        },
    }
    import json
    with open(os.path.join(save_dir, "v7_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✅ 결과 저장: {save_dir}/v7_results.json")
    print(f"✅ 베스트 모델: {best_path}\n{'='*60}")


if __name__ == "__main__":
    train()

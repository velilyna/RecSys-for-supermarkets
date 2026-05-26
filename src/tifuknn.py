import os
import sys
import time
import math
import random

import torch
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from data import load_preprocessed


def basket_to_vec(basket, item2idx, num_items):
    x = torch.zeros(num_items, dtype=torch.float32)
    for item in basket:
        idx = item2idx.get(item)
        if idx is not None:
            x[idx] = 1.0
    return x


def make_sample_refs(user_hist, seq_len=5):
    refs = []

    for user_id, seq in user_hist.items():
        if len(seq) <= seq_len:
            continue

        for target_idx in range(seq_len, len(seq)):
            refs.append((user_id, target_idx))

    return refs


def split_refs(refs, val_ratio=0.1, seed=42):
    rng = random.Random(seed)
    refs = list(refs)
    rng.shuffle(refs)

    n_total = len(refs)
    n_val = max(1, int(val_ratio * n_total))
    n_val = min(n_val, n_total - 1)

    train_refs = refs[:-n_val]
    val_refs = refs[-n_val:]

    return train_refs, val_refs


def get_hist_and_target_from_ref(ref, user_hist, item2idx, num_items, seq_len):
    user_id, target_idx = ref
    seq = user_hist[user_id]

    context = seq[target_idx - seq_len : target_idx]
    target_basket = seq[target_idx][1]

    hist = torch.stack(
        [
            basket_to_vec(basket, item2idx, num_items)
            for _, basket in context
        ],
        dim=0,
    )

    target = basket_to_vec(target_basket, item2idx, num_items)

    return hist, target


def tifu_user_vector(hist, group_size=1, decay=0.8):
    """
    hist: [L, I]
    """
    L, num_items = hist.shape

    groups = []
    for start in range(0, L, group_size):
        group = hist[start : start + group_size]
        groups.append(group.sum(dim=0))

    user_vec = torch.zeros(num_items, dtype=torch.float32)

    n_groups = len(groups)
    for g_idx, group_vec in enumerate(groups):
        age = n_groups - 1 - g_idx
        weight = decay ** age
        user_vec += weight * group_vec

    user_vec = torch.log1p(user_vec)

    norm = user_vec.norm(p=2)
    if norm > 0:
        user_vec = user_vec / norm

    return user_vec


def build_limited_train_profiles(
    train_refs,
    user_hist,
    item2idx,
    num_items,
    seq_len=5,
    group_size=1,
    decay=0.8,
    max_train_profiles=5000,
    seed=42,
):
    """
    Строим dense train profiles только для ограниченного числа refs.
    Это спасает память на Dunnhumby.
    """
    rng = random.Random(seed)
    train_refs = list(train_refs)
    rng.shuffle(train_refs)

    if max_train_profiles is not None:
        train_refs = train_refs[:max_train_profiles]

    profiles = []
    targets = []

    for idx, ref in enumerate(train_refs, start=1):
        hist, target = get_hist_and_target_from_ref(
            ref=ref,
            user_hist=user_hist,
            item2idx=item2idx,
            num_items=num_items,
            seq_len=seq_len,
        )

        profile = tifu_user_vector(
            hist,
            group_size=group_size,
            decay=decay,
        )

        profiles.append(profile)
        targets.append(target)

        if idx % 500 == 0 or idx == len(train_refs):
            print(f"    built train profile {idx}/{len(train_refs)}")

    profiles = torch.stack(profiles, dim=0)
    targets = torch.stack(targets, dim=0)

    return profiles, targets


def recommend_tifu_knn(
    query_hist,
    train_profiles,
    train_targets,
    k_neighbors=50,
    group_size=1,
    decay=0.8,
    personal_alpha=0.5,
):
    query_vec = tifu_user_vector(
        query_hist,
        group_size=group_size,
        decay=decay,
    )

    sims = torch.mv(train_profiles, query_vec)

    k_eff = min(k_neighbors, train_profiles.size(0))
    top_vals, top_idx = torch.topk(sims, k=k_eff)

    weights = torch.clamp(top_vals, min=0.0)

    if weights.sum().item() <= 0:
        weights = torch.ones_like(weights)

    neighbor_targets = train_targets[top_idx]

    scores = (neighbor_targets * weights.unsqueeze(1)).sum(dim=0)
    scores = scores / weights.sum().clamp(min=1e-8)

    personal_freq = torch.log1p(query_hist.sum(dim=0))
    scores = scores + personal_alpha * personal_freq

    return scores


@torch.no_grad()
def evaluate_one(scores, hist, target, ks=(10, 20, 100)):
    target_mask = target > 0
    seen_mask = hist.sum(dim=0) > 0

    result = {}

    max_k = min(max(ks), scores.size(0))
    topk_all = torch.topk(scores, k=max_k).indices

    target_count = target_mask.sum().float().clamp(min=1.0)

    for k in ks:
        k_eff = min(k, scores.size(0))
        pred_k = topk_all[:k_eff]

        hits = target_mask[pred_k].float()

        recall = hits.sum() / target_count

        discounts = 1.0 / torch.log2(
            torch.arange(k_eff).float() + 2.0
        )

        dcg = (hits.cpu() * discounts).sum()

        ideal_len = min(int(target_count.item()), k_eff)
        idcg = discounts[:ideal_len].sum()

        ndcg = dcg / idcg.clamp(min=1e-8)

        un = (~seen_mask[pred_k]).float().mean()

        result[f"Recall@{k}"] = recall.item()
        result[f"NDCG@{k}"] = ndcg.item()
        result[f"UN@{k}"] = un.item()

    return result


def evaluate_tifu_knn(
    val_refs,
    user_hist,
    item2idx,
    num_items,
    train_profiles,
    train_targets,
    seq_len=5,
    group_size=1,
    decay=0.8,
    k_neighbors=50,
    personal_alpha=0.5,
    max_eval_samples=1000,
    ks=(10, 20, 100),
    seed=42,
):
    rng = random.Random(seed)
    val_refs = list(val_refs)
    rng.shuffle(val_refs)

    if max_eval_samples is not None:
        val_refs = val_refs[:max_eval_samples]

    totals = {}
    for k in ks:
        totals[f"Recall@{k}"] = 0.0
        totals[f"NDCG@{k}"] = 0.0
        totals[f"UN@{k}"] = 0.0

    total = 0

    for idx, ref in enumerate(val_refs, start=1):
        hist, target = get_hist_and_target_from_ref(
            ref=ref,
            user_hist=user_hist,
            item2idx=item2idx,
            num_items=num_items,
            seq_len=seq_len,
        )

        scores = recommend_tifu_knn(
            query_hist=hist,
            train_profiles=train_profiles,
            train_targets=train_targets,
            k_neighbors=k_neighbors,
            group_size=group_size,
            decay=decay,
            personal_alpha=personal_alpha,
        )

        m = evaluate_one(
            scores=scores,
            hist=hist,
            target=target,
            ks=ks,
        )

        for key, value in m.items():
            totals[key] += value

        total += 1

        if idx % 200 == 0 or idx == len(val_refs):
            print(f"    evaluated {idx}/{len(val_refs)}")

    for key in totals:
        totals[key] /= max(total, 1)

    return totals, total


def run_one_dataset(
    dataset_name,
    seq_len=5,
    group_size=1,
    decay=0.8,
    k_neighbors=50,
    personal_alpha=0.5,
    max_train_profiles=5000,
    max_eval_samples=1000,
):
    print(f"\n=== TIFU-KNN on {dataset_name} ===")

    user_hist, item2idx, _ = load_preprocessed(dataset_name)
    num_items = len(item2idx)

    print(f"  users={len(user_hist)}, items={num_items}")
    print("  building sample refs...")

    refs = make_sample_refs(user_hist, seq_len=seq_len)

    train_refs, val_refs = split_refs(refs, val_ratio=0.1, seed=42)

    print(
        f"  refs total={len(refs)}, train={len(train_refs)}, val={len(val_refs)}"
    )

    print("  building limited train profiles...")
    train_profiles, train_targets = build_limited_train_profiles(
        train_refs=train_refs,
        user_hist=user_hist,
        item2idx=item2idx,
        num_items=num_items,
        seq_len=seq_len,
        group_size=group_size,
        decay=decay,
        max_train_profiles=max_train_profiles,
    )

    print(
        f"  train_profiles={tuple(train_profiles.shape)}, "
        f"train_targets={tuple(train_targets.shape)}"
    )

    t0 = time.time()

    metrics, n_eval = evaluate_tifu_knn(
        val_refs=val_refs,
        user_hist=user_hist,
        item2idx=item2idx,
        num_items=num_items,
        train_profiles=train_profiles,
        train_targets=train_targets,
        seq_len=seq_len,
        group_size=group_size,
        decay=decay,
        k_neighbors=k_neighbors,
        personal_alpha=personal_alpha,
        max_eval_samples=max_eval_samples,
        ks=(10, 20, 100),
    )

    elapsed = time.time() - t0

    row = {
        "Dataset": dataset_name,
        "Model": "TIFU-KNN",
        "seq_len": seq_len,
        "group_size": group_size,
        "decay": decay,
        "k_neighbors": k_neighbors,
        "personal_alpha": personal_alpha,
        "max_train_profiles": max_train_profiles,
        "max_eval_samples": max_eval_samples,
        "num_items": num_items,
        "train_refs": len(train_refs),
        "val_refs": len(val_refs),
        "n_eval": n_eval,
        "eval_time_sec": elapsed,
    }

    row.update(metrics)

    print("\n  Result:")
    print(pd.DataFrame([row]).round(4))

    return row


def main():
    rows = []

    configs = [
        {
            "dataset_name": "tafeng",
            "seq_len": 5,
            "max_train_profiles": 10000,
            "max_eval_samples": None,
        },
        {
            "dataset_name": "dunnhumby",
            "seq_len": 5,
            "max_train_profiles": 2000,
            "max_eval_samples": 1000,
        },
    ]

    for cfg in configs:
        try:
            row = run_one_dataset(
                dataset_name=cfg["dataset_name"],
                seq_len=cfg["seq_len"],
                group_size=1,
                decay=0.8,
                k_neighbors=50,
                personal_alpha=0.5,
                max_train_profiles=cfg["max_train_profiles"],
                max_eval_samples=cfg["max_eval_samples"],
            )
            rows.append(row)
        except FileNotFoundError as e:
            print(f"[SKIP] {cfg['dataset_name']}: {e}")
        except RuntimeError as e:
            print(f"[ERROR] {cfg['dataset_name']}: {e}")

    if len(rows) == 0:
        raise RuntimeError("No results produced.")

    df = pd.DataFrame(rows)

    os.makedirs("data", exist_ok=True)

    out_path = "data/results_tifu_knn_all_datasets.csv"
    df.to_csv(out_path, index=False)

    print("\n All TIFU-KNN results")
    print(df.round(4))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
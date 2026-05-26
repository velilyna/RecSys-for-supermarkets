import os
import torch
import pandas as pd
from torch.utils.data import DataLoader, random_split

from data import get_device, get_or_build_dataset, load_preprocessed


@torch.no_grad()
def evaluate_scores(scores, hist_baskets, y, K_values=(10, 100)):
    """
    Общая функция для метрик.

    scores:       [B, I]      score товара для каждого пользователя
    hist_baskets: [B, L, I]   история пользователя
    y:            [B, I]      target basket

    UN@K здесь считается как user novelty:
    доля top-K рекомендаций, которых НЕ было в истории конкретного пользователя.
    """
    device = scores.device
    y_true = y > 0
    seen_mask = hist_baskets.sum(dim=1) > 0

    max_k = max(K_values)
    max_k = min(max_k, scores.size(1))
    topk_all = torch.topk(scores, k=max_k, dim=1).indices

    batch_results = {}

    for K in K_values:
        K_eff = min(K, scores.size(1))
        topk = topk_all[:, :K_eff]

        # Recall@K
        hits = torch.gather(y_true, 1, topk).float()
        target_count = y_true.sum(dim=1).float().clamp(min=1.0)
        recall = hits.sum(dim=1) / target_count

        # NDCG@K
        discounts = 1.0 / torch.log2(
            torch.arange(K_eff, device=device).float() + 2.0
        )
        dcg = (hits * discounts).sum(dim=1)

        ideal_len = torch.minimum(
            target_count.long(),
            torch.tensor(K_eff, device=device),
        )

        idcg = torch.zeros_like(dcg)
        for i in range(scores.size(0)):
            if ideal_len[i] > 0:
                idcg[i] = discounts[: ideal_len[i]].sum()

        ndcg = dcg / idcg.clamp(min=1e-8)
        novel = torch.gather(~seen_mask, 1, topk).float()
        un = novel.mean(dim=1)

        batch_results[f"Recall@{K}"] = recall.sum().item()
        batch_results[f"NDCG@{K}"] = ndcg.sum().item()
        batch_results[f"UN@{K}"] = un.sum().item()

    return batch_results


@torch.no_grad()
def evaluate_top_popular(train_loader, val_loader, num_items, K_values=(10, 100), device="cpu"):
    """
    Top Popular:
    рекомендует всем пользователям глобально самые частые товары из train
    """
    popularity = torch.zeros(num_items, device=device)

    for hist_baskets, _ in train_loader:
        hist_baskets = hist_baskets.to(device)
        popularity += hist_baskets.sum(dim=0).sum(dim=0)

    results = {}
    for K in K_values:
        results[f"Recall@{K}"] = 0.0
        results[f"NDCG@{K}"] = 0.0
        results[f"UN@{K}"] = 0.0

    total_samples = 0

    for hist_baskets, y in val_loader:
        hist_baskets = hist_baskets.to(device)
        y = y.to(device)

        batch_size = y.size(0)
        total_samples += batch_size

        scores = popularity.unsqueeze(0).expand(batch_size, -1)
        batch_results = evaluate_scores(scores, hist_baskets, y, K_values=K_values)

        for key, value in batch_results.items():
            results[key] += value

    for key in results:
        results[key] /= max(total_samples, 1)

    return results


@torch.no_grad()
def evaluate_top_personal(val_loader, num_items, K_values=(10, 100), device="cpu"):
    """
    Top Personal:
    рекомендует товары, которые пользователь чаще всего покупал в своей истории.
    """
    results = {}
    for K in K_values:
        results[f"Recall@{K}"] = 0.0
        results[f"NDCG@{K}"] = 0.0
        results[f"UN@{K}"] = 0.0

    total_samples = 0

    for hist_baskets, y in val_loader:
        hist_baskets = hist_baskets.to(device)
        y = y.to(device)

        batch_size = y.size(0)
        total_samples += batch_size

        # [B, L, I] -> [B, I]
        # персональная частота товара в истории пользователя
        scores = hist_baskets.sum(dim=1)

        batch_results = evaluate_scores(scores, hist_baskets, y, K_values=K_values)

        for key, value in batch_results.items():
            results[key] += value

    for key in results:
        results[key] /= max(total_samples, 1)

    return results


def make_train_val_split(dataset):
    """
    Безопасный random split:
    не даёт val съесть весь датасет.
    """
    n_total = len(dataset)

    if n_total == 0:
        raise ValueError(
            "Датасет пустой. Уменьши seq_len или проверь preprocessing."
        )

    n_val = max(1, int(0.1 * n_total))
    n_val = min(n_val, n_total - 1)
    n_train = n_total - n_val

    return random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )


if __name__ == "__main__":
    device = get_device()
    print(f"Устройство: {str(device).upper()}")
    datasets = ["tafeng"]

    all_results = []

    for dataset_name in datasets:
        print(f"\n Бейзлайны для {dataset_name}")

        _, item2idx, _ = load_preprocessed(dataset_name)
        num_items = len(item2idx)

        dataset = get_or_build_dataset(
            dataset_name,
            seq_len=5,
            weighting="binary",
        )

        train_ds, val_ds = make_train_val_split(dataset)

        print(f"  train={len(train_ds)}, val={len(val_ds)}, items={num_items}")

        train_loader = DataLoader(
            train_ds,
            batch_size=256,
            shuffle=True,
            num_workers=0,
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=256,
            shuffle=False,
            num_workers=0,
        )


        pop_metrics = evaluate_top_popular(
            train_loader=train_loader,
            val_loader=val_loader,
            num_items=num_items,
            K_values=(10, 100),
            device=device,
        )

        row = {
            "Dataset": dataset_name,
            "Model": "top_popular",
        }
        row.update(pop_metrics)
        all_results.append(row)

        print(
            f"  Recall@10={pop_metrics['Recall@10']:.4f} | "
            f"Recall@100={pop_metrics['Recall@100']:.4f} | "
            f"NDCG@10={pop_metrics['NDCG@10']:.4f} | "
            f"NDCG@100={pop_metrics['NDCG@100']:.4f} | "
            f"UN@10={pop_metrics['UN@10']:.4f} | "
            f"UN@100={pop_metrics['UN@100']:.4f}"
        )

        pers_metrics = evaluate_top_personal(
            val_loader=val_loader,
            num_items=num_items,
            K_values=(10, 100),
            device=device,
        )

        row = {
            "Dataset": dataset_name,
            "Model": "top_personal",
        }
        row.update(pers_metrics)
        all_results.append(row)

        print(
            f"  Recall@10={pers_metrics['Recall@10']:.4f} | "
            f"Recall@100={pers_metrics['Recall@100']:.4f} | "
            f"NDCG@10={pers_metrics['NDCG@10']:.4f} | "
            f"NDCG@100={pers_metrics['NDCG@100']:.4f} | "
            f"UN@10={pers_metrics['UN@10']:.4f} | "
            f"UN@100={pers_metrics['UN@100']:.4f}"
        )

    df = pd.DataFrame(all_results)

    print("\nРезультаты бейзлайнов")
    print(df.round(4))

    os.makedirs("data", exist_ok=True)

    output_path = "data/results_baselines.csv"
    df.to_csv(output_path, index=False)

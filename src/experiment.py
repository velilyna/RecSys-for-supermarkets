import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pandas as pd
from data import load_preprocessed, NextBasketDataset
from model import SAFERecAutoEncoder


def bce_loss(logits, y):
    return F.binary_cross_entropy_with_logits(logits, y)


@torch.no_grad()
def evaluate_metrics(model, loader, K_values=[10, 100], device="cpu"):
    """Оценка Recall@K, NDCG@K и Coverage (UN@K)"""
    model.eval()
    results = {f"Recall@{k}": 0.0 for k in K_values}
    results.update({f"NDCG@{k}": 0.0 for k in K_values})
    results.update({f"UN@{k}": 0.0 for k in K_values})
    total_samples = 0
    all_recommended_items = {k: set() for k in K_values}

    for batch in loader:
        if len(batch) == 3:
            x, y, freq = batch
            freq = freq.to(device)
        else:
            x, y = batch
            freq = None

        x, y = x.to(device), y.to(device)
        try:
            logits = model(x, freq)
        except TypeError:
            logits = model(x)

        total_samples += y.size(0)

        for K in K_values:
            topk = torch.topk(logits, k=K, dim=1).indices
            y_true = (y > 0.5).float()

            # Recall@K
            hits = y_true.gather(1, topk).sum(dim=1)
            recall = (hits / y_true.sum(dim=1).clamp(min=1.0)).mean()
            results[f"Recall@{K}"] += recall.item() * y.size(0)

            # NDCG@K
            gains = y_true.gather(1, topk)
            discounts = 1.0 / torch.log2(torch.arange(2, K + 2, device=y.device).float())
            dcg = (gains * discounts).sum(dim=1)
            ideal_gains = torch.sort(y_true, descending=True)[0][:, :K]
            idcg = (ideal_gains * discounts[:ideal_gains.size(1)]).sum(dim=1).clamp(min=1e-6)
            ndcg = (dcg / idcg).mean()
            results[f"NDCG@{K}"] += ndcg.item() * y.size(0)

            # Coverage (UN@K)
            top_items = topk.flatten().tolist()
            all_recommended_items[K].update(top_items)

    for K in K_values:
        results[f"Recall@{K}"] /= total_samples
        results[f"NDCG@{K}"] /= total_samples
        results[f"UN@{K}"] = len(all_recommended_items[K]) / model.num_items

    return results


def train_and_evaluate(dataset_name, weighting="time_since_last", device="cuda" if torch.cuda.is_available() else "cpu"):
    """Тренировка модели + оценка на валидации"""
    user_hist, item2idx, _ = load_preprocessed(dataset_name)

    dataset = NextBasketDataset(
        user_hist,
        item2idx,
        seq_len=5,
        weighting=weighting,  # "binary", "exp_decay", "time_since_last"
        decay=0.85,
        max_days=14,
    )

    n_total = len(dataset)
    n_val = max(1000, int(0.1 * n_total))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256)

    model = SAFERecAutoEncoder(num_items=len(item2idx))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    print(f"\n=== Обучаем модель на {dataset_name} ({weighting}) ===")
    for epoch in range(1, 6):
        model.train()
        total_loss = 0
        for batch in train_loader:
            if len(batch) == 3:
                x, y, freq = batch
                freq = freq.to(device)
            else:
                x, y = batch
                freq = None

            x, y = x.to(device), y.to(device)

            try:
                logits = model(x, freq)
            except TypeError:
                logits = model(x)

            loss = bce_loss(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.size(0)

        avg_loss = total_loss / len(train_ds)
        print(f"Epoch {epoch}: loss={avg_loss:.4f}")

    print("\nОцениваем метрики...")
    metrics = evaluate_metrics(model, val_loader, device=device)
    return metrics


if __name__ == "__main__":
    results = []
    for dataset_name in ["tafeng", "dunnhumby"]:
        for weighting in ["binary", "exp_decay", "time_since_last"]:
            metrics = train_and_evaluate(dataset_name, weighting)
            row = {"Dataset": dataset_name, "Weighting": weighting}
            row.update(metrics)
            results.append(row)

    df = pd.DataFrame(results)
    print("\n=== Финальные результаты ===")
    print(df.round(4))
    df.to_csv("data/results_saferec_ae.csv", index=False)
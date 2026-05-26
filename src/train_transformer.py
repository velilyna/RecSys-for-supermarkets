import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pandas as pd
from tqdm import tqdm

from data import load_preprocessed, NextBasketDataset
from safer_transformer import SAFERTransformer


def bce_loss(logits, y):
    return F.binary_cross_entropy_with_logits(logits, y)


@torch.no_grad()
def evaluate_loss(model, loader, device="cpu"):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        if len(batch) == 3:
            x, y, freq = batch
            freq = torch.nan_to_num(freq.to(device), nan=0.0, posinf=0.0, neginf=0.0)
        else:
            x, y = batch
            freq = None

        x = torch.nan_to_num(x.to(device), nan=0.0, posinf=0.0, neginf=0.0)
        y = torch.nan_to_num(y.to(device), nan=0.0, posinf=0.0, neginf=0.0)

        logits = model(x, freq)
        loss = bce_loss(logits, y)

        total_loss += loss.item() * y.size(0)
        total_samples += y.size(0)

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate_metrics(model, loader, K_values=(10, 100), device="cpu"):
    model.eval()

    results = {f"Recall@{k}": 0.0 for k in K_values}
    results.update({f"NDCG@{k}": 0.0 for k in K_values})
    results.update({f"UN@{k}": 0.0 for k in K_values})

    total_samples = 0
    all_recommended_items = {k: set() for k in K_values}

    for batch in loader:
        if len(batch) == 3:
            x, y, freq = batch
            freq = torch.nan_to_num(freq.to(device), nan=0.0, posinf=0.0, neginf=0.0)
        else:
            x, y = batch
            freq = None

        x = torch.nan_to_num(x.to(device), nan=0.0, posinf=0.0, neginf=0.0)
        y = torch.nan_to_num(y.to(device), nan=0.0, posinf=0.0, neginf=0.0)

        logits = model(x, freq)
        y_true = (y > 0.5).float()
        total_samples += y.size(0)

        hist = (x > 0.0).float()

        for K in K_values:
            K_eff = min(K, logits.size(1))
            topk = torch.topk(logits, k=K_eff, dim=1).indices

            hits = y_true.gather(1, topk).sum(dim=1)
            denom = y_true.sum(dim=1).clamp(min=1.0)
            recall = hits / denom
            results[f"Recall@{K}"] += recall.sum().item()

            gains = y_true.gather(1, topk)
            discounts = 1.0 / torch.log2(
                torch.arange(2, K_eff + 2, device=y.device, dtype=torch.float32)
            )
            dcg = (gains * discounts.unsqueeze(0)).sum(dim=1)

            num_relevant = y_true.sum(dim=1).clamp(max=K_eff)
            idcg = torch.zeros_like(dcg)
            for i in range(K_eff):
                idcg += (i < num_relevant).float() * discounts[i]
            idcg = idcg.clamp(min=1e-6)

            ndcg = dcg / idcg
            results[f"NDCG@{K}"] += ndcg.sum().item()

            seen_flags = hist.gather(1, topk)
            un = 1.0 - seen_flags.mean(dim=1)
            results[f"UN@{K}"] += un.sum().item()

            all_recommended_items[K].update(topk.flatten().tolist())

    for K in K_values:
        results[f"Recall@{K}"] /= max(total_samples, 1)
        results[f"NDCG@{K}"] /= max(total_samples, 1)
        results[f"UN@{K}"] /= max(total_samples, 1)
        results[f"Coverage@{K}"] = len(all_recommended_items[K]) / max(model.num_items, 1)

    return results


def train_transformer(dataset_name, weighting, num_epochs=10, device=None):
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"Using device: {device}")

    user_hist, item2idx, _ = load_preprocessed(dataset_name)
    num_items = len(item2idx)

    dataset = NextBasketDataset(
        user_hist,
        item2idx,
        seq_len=5,
        weighting=weighting,
        decay=0.85,
        max_days=14,
    )

    n_total = len(dataset)
    n_val = min(max(1000, int(0.1 * n_total)), max(n_total - 1, 1))
    n_train = n_total - n_val
    if n_train <= 0:
        raise ValueError("Dataset is too small after validation split")

    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    model = SAFERTransformer(
        num_items=num_items,
        hidden_dim=128,
        d_freq=16,
        nhead=4,
        num_layers=1,
        dropout=0.1,
        max_len=128,
        use_freq=True,
        pooling="cls",
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    best_recall10 = -1.0
    best_metrics = None
    best_model_path = f"best_transformer_{dataset_name}_{weighting}.pt"
    history = []

    print(f"\n=== Обучаем TRANSFORMER на {dataset_name} ({weighting}) ===")
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        for batch in pbar:
            if len(batch) == 3:
                x, y, freq = batch
                freq = torch.nan_to_num(freq.to(device), nan=0.0, posinf=0.0, neginf=0.0)
            else:
                x, y = batch
                freq = None

            x = torch.nan_to_num(x.to(device), nan=0.0, posinf=0.0, neginf=0.0)
            y = torch.nan_to_num(y.to(device), nan=0.0, posinf=0.0, neginf=0.0)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x, freq)
            loss = bce_loss(logits, y)

            if not torch.isfinite(loss):
                print("[WARN] Пропускаю batch: loss не является конечным числом")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * y.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = total_loss / max(len(train_ds), 1)
        val_loss = evaluate_loss(model, val_loader, device=device)
        metrics = evaluate_metrics(model, val_loader, K_values=(10, 100), device=device)
        scheduler.step(metrics["Recall@10"])

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **metrics,
        }
        history.append(row)

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"Recall@10={metrics['Recall@10']:.4f}, "
            f"Recall@100={metrics['Recall@100']:.4f}, "
            f"NDCG@10={metrics['NDCG@10']:.4f}, "
            f"NDCG@100={metrics['NDCG@100']:.4f}, "
            f"UN@10={metrics['UN@10']:.4f}, "
            f"UN@100={metrics['UN@100']:.4f}, "
            f"Coverage@10={metrics['Coverage@10']:.4f}, "
            f"Coverage@100={metrics['Coverage@100']:.4f}"
        )

        if metrics["Recall@10"] > best_recall10:
            best_recall10 = metrics["Recall@10"]
            best_metrics = {"val_loss": val_loss, **metrics}
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model to {best_model_path}")

    history_df = pd.DataFrame(history)
    history_path = f"history_transformer_{dataset_name}_{weighting}.csv"
    history_df.to_csv(history_path, index=False)
    print(f"Saved training history to {history_path}")

    if best_metrics is None:
        best_metrics = evaluate_metrics(model, val_loader, K_values=(10, 100), device=device)
        best_metrics["val_loss"] = evaluate_loss(model, val_loader, device=device)

    return best_metrics


if __name__ == "__main__":
    results = []
    datasets = ["tafeng", "dunnhumby"]
    weightings = ["binary", "exp_decay", "time_since_last"]

    for dataset_name in datasets:
        for weighting in weightings:
            metrics = train_transformer(dataset_name, weighting, num_epochs=10)
            row = {
                "Dataset": dataset_name,
                "Model": "transformer",
                "Weighting": weighting,
            }
            row.update(metrics)
            results.append(row)

    df = pd.DataFrame(results)
    print(df.round(4))
    df.to_csv("results_transformer.csv", index=False)

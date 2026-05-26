import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pandas as pd

from data import get_device, get_or_build_dataset, load_preprocessed
from safer_autoencoder import SAFERecAutoEncoder


def basket_ce_loss(logits, y):
    y = y.float()
    y_norm = y / y.sum(dim=1, keepdim=True).clamp(min=1.0)

    log_probs = torch.log_softmax(logits, dim=1)
    loss = -(y_norm * log_probs).sum(dim=1).mean()

    return loss


@torch.no_grad()
def evaluate_metrics(model, loader, device, ks=(10, 20, 100)):
    model.eval()

    total_recall = {k: 0.0 for k in ks}
    total_ndcg = {k: 0.0 for k in ks}
    total_un = {k: 0.0 for k in ks}
    n_users = 0
    first_hit_positions = {k: [] for k in ks}

    for hist_baskets, y in loader:
        hist_baskets = hist_baskets.to(device)
        y = y.to(device)

        logits, _ = model(hist_baskets)
        scores = logits

        seen_mask = hist_baskets.sum(dim=1) > 0
        target_mask = y > 0

        max_k = max(ks)
        max_k = min(max_k, scores.size(1))

        topk = torch.topk(scores, k=max_k, dim=1).indices

        batch_size = y.size(0)
        n_users += batch_size

        for k in ks:
            k_eff = min(k, scores.size(1))
            pred_k = topk[:, :k_eff]
            hits = torch.gather(target_mask, 1, pred_k).float()
            for i in range(hits.size(0)):
                hit_pos = hits[i].nonzero(as_tuple=False).flatten()
                if len(hit_pos) > 0:
                    first_hit_positions[k].append(hit_pos[0].item() + 1)

            target_count = target_mask.sum(dim=1).float().clamp(min=1.0)
            recall = hits.sum(dim=1) / target_count

            discounts = 1.0 / torch.log2(
                torch.arange(k_eff, device=device).float() + 2.0
            )

            dcg = (hits * discounts).sum(dim=1)

            ideal_len = torch.minimum(
                target_count.long(),
                torch.tensor(k_eff, device=device),
            )

            idcg = torch.zeros_like(dcg)

            for i in range(batch_size):
                if ideal_len[i] > 0:
                    idcg[i] = discounts[: ideal_len[i]].sum()

            ndcg = dcg / idcg.clamp(min=1e-8)
            novel = torch.gather(~seen_mask, 1, pred_k).float()
            un = novel.mean(dim=1)

            total_recall[k] += recall.sum().item()
            total_ndcg[k] += ndcg.sum().item()
            total_un[k] += un.sum().item()

    result = {}

    for k in ks:
        result[f"Recall@{k}"] = total_recall[k] / max(n_users, 1)
        result[f"NDCG@{k}"] = total_ndcg[k] / max(n_users, 1)
        result[f"UN@{k}"] = total_un[k] / max(n_users, 1)

        if len(first_hit_positions[k]) > 0:
            result[f"avg_first_hit_pos@{k}"] = (
                sum(first_hit_positions[k]) / len(first_hit_positions[k])
            )
        else:
            result[f"avg_first_hit_pos@{k}"] = float("nan")

    return result

def train_and_evaluate(
    dataset_name,
    seq_len=5,
    batch_size=32,
    epochs=10,
    lr=1e-3,
    weight_decay=1e-4,
):
    device = get_device()
    print(f"\n  Устройство: {device.upper()}")

    dataset = get_or_build_dataset(dataset_name, seq_len=seq_len)
    _, item2idx, _ = load_preprocessed(dataset_name)

    n_total = len(dataset)

    n_val = max(1, int(0.1 * n_total))
    n_val = min(n_val, n_total - 1)
    n_train = n_total - n_val

    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )


    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    model = SAFERecAutoEncoder(
        num_items=len(item2idx),
        seq_len=seq_len,
        basket_emb_dim=128,
        ae_hidden_dim=256,
        user_dim=128,
        freq_emb_dim=64,
        freq_hidden_dim=128,
        dropout=0.2,
        max_freq_clip=47 if dataset_name == "tafeng" else 36,
        use_recon_loss=True,
        recon_loss_weight=0.1,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)



    best_recall10 = -1.0
    best_path = f"data/{dataset_name}_saferec_ae.pt"
    os.makedirs("data", exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for batch_idx, (hist_baskets, y) in enumerate(train_loader):
            hist_baskets = hist_baskets.to(device).float()
            y = y.to(device).float()

            logits, recon_loss = model(hist_baskets)
            loss = basket_ce_loss(logits, y) + model.recon_loss_weight * recon_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.size(0)

            if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(train_loader):
                elapsed = time.time() - t0
                batches_done = batch_idx + 1
                batches_total = len(train_loader)
                eta = elapsed / batches_done * (batches_total - batches_done)
                print(
                    f"  Epoch {epoch:02d} [{batches_done:3d}/{batches_total}] "
                    f"loss={total_loss / (batches_done * batch_size):.4f} "
                    f"elapsed={elapsed:.0f}s eta={eta:.0f}s",
                    end="\r",
                )

        epoch_time = time.time() - t0
        avg_loss = total_loss / len(train_ds)

        print(f"\n  Epoch {epoch:02d} done in {epoch_time:.0f}s | loss={avg_loss:.4f} | валидация...")

        val_metrics = evaluate_metrics(model, val_loader, device=device)

        print(
            f"Epoch {epoch:02d} | "
            f"Recall@10={val_metrics['Recall@10']:.4f} | "
            f"NDCG@10={val_metrics['NDCG@10']:.4f} | "
            f"UN@10={val_metrics['UN@10']:.4f} | "
            f"Recall@20={val_metrics['Recall@20']:.4f} | "
            f"NDCG@20={val_metrics['NDCG@20']:.4f} | "
            f"UN@20={val_metrics['UN@20']:.4f} | "
            f"Recall@100={val_metrics['Recall@100']:.4f} | "
            f"NDCG@100={val_metrics['NDCG@100']:.4f} | "
            f"UN@100={val_metrics['UN@100']:.4f} | "
            f"FirstHit@10={val_metrics['avg_first_hit_pos@10']:.2f} | "
            f"FirstHit@20={val_metrics['avg_first_hit_pos@20']:.2f} | "
            f"FirstHit@100={val_metrics['avg_first_hit_pos@100']:.2f}"
        )

        if val_metrics["Recall@10"] > best_recall10:
            best_recall10 = val_metrics["Recall@10"]
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "num_items": model.num_items,
                    "seq_len": model.seq_len,
                },
                best_path,
            )

    # Финальные метрики с лучшего чекпоинта
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    final_metrics = evaluate_metrics(model, val_loader, device=device)
    return final_metrics


if __name__ == "__main__":
    results = []

    for dataset_name in ["tafeng", "dunnhumby"]:
        seq_len = 5

        metrics = train_and_evaluate(
            dataset_name=dataset_name,
            seq_len=seq_len,
            batch_size=32,
            epochs=20,
            lr=1e-3,
            weight_decay=1e-4,
        )
        row = {"Dataset": dataset_name}
        row.update(metrics)
        results.append(row)

    df = pd.DataFrame(results)
    print("\n Финальные результаты")
    print(df.round(4))

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/results_saferec_ae.csv", index=False)

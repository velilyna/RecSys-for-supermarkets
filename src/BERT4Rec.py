"""
BERT4Rec adapted for Next-Basket Recommendation (v2 — fixed).

Ключевые отличия от оригинального BERT4Rec [Sun et al., 2019]:
- Вход: последовательность корзин (каждая корзина = multi-hot),
  а не отдельных item-ов. Сохраняется структура корзин.
- Masking: маскируется одна позиция (корзина) из контекста.
  Модель учится восстанавливать все item-ы маскированной корзины.
- Inference: последняя позиция маскируется → скоры по всем item-ам.
- Loss: listwise cross-entropy (аналогично SAFERec-AE).

Исправления v2:
- Двухступенчатая проекция basket: num_items -> proj_dim -> hidden_dim
  (proj_dim=256 снижает размерность до Transformer)
- Добавлен frequency residual: log(1 + sum(history)) как в SAFERec-AE
- lr warmup через OneCycleLR вместо ReduceLROnPlateau
- Увеличен num_layers до 2
"""

import os
import time
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd

from data import load_preprocessed, get_device


class BERT4RecNBRDataset(Dataset):
    def __init__(self, user_hist, item2idx, seq_len=5, mode="train"):
        self.samples = []
        self.item2idx = item2idx
        self.num_items = len(item2idx)
        self.seq_len = seq_len
        self.mode = mode

        for user_id, seq in user_hist.items():
            if len(seq) <= seq_len:
                continue
            for i in range(seq_len, len(seq)):
                context = seq[i - seq_len:i]
                target = seq[i][1]
                self.samples.append((context, target))

    def __len__(self):
        return len(self.samples)

    def _to_multihot(self, basket):
        x = torch.zeros(self.num_items, dtype=torch.float32)
        for item in basket:
            idx = self.item2idx.get(item)
            if idx is not None:
                x[idx] = 1.0
        return x

    def __getitem__(self, idx):
        context, target_items = self.samples[idx]

        baskets = torch.stack(
            [self._to_multihot(basket) for _, basket in context], dim=0
        )  # [L, I]

        if self.mode == "train":
            mask_pos = random.randint(0, self.seq_len - 1)
            masked_basket = baskets[mask_pos].clone()
            baskets[mask_pos] = 0.0
            return baskets, masked_basket, torch.tensor(mask_pos, dtype=torch.long)
        else:
            baskets_eval = baskets.clone()
            baskets_eval[-1] = 0.0
            target = self._to_multihot(target_items)
            mask_pos = torch.tensor(self.seq_len - 1, dtype=torch.long)
            return baskets_eval, target, mask_pos




class BERT4RecNBR(nn.Module):
    """
    Двухступенчатая проекция:
      [B, L, num_items] --(proj)--> [B, L, proj_dim] --(up)--> [B, L, hidden_dim]
    Это критично: прямая проекция из 14k+ измерений перегружает модель.

    Frequency residual добавляется к logits: alpha * log(1 + sum_history(i))
    Аналогично SAFERec-AE — важный сигнал для grocery данных.
    """

    def __init__(
        self,
        num_items,
        seq_len=5,
        proj_dim=256,
        hidden_dim=128,
        nhead=4,
        num_layers=2,
        dropout=0.1,
        layer_norm_eps=1e-5,
    ):
        super().__init__()
        self.num_items = num_items
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        self.basket_proj = nn.Sequential(
            nn.Linear(num_items, proj_dim),
            nn.LayerNorm(proj_dim, eps=layer_norm_eps),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, hidden_dim),
            nn.LayerNorm(hidden_dim, eps=layer_norm_eps),
        )

        self.pos_embedding = nn.Embedding(seq_len + 1, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            layer_norm_eps=layer_norm_eps,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim, eps=layer_norm_eps),
        )

        self.dropout = nn.Dropout(dropout)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim, eps=layer_norm_eps),
            nn.Linear(hidden_dim, num_items),
        )

        self.alpha_freq = nn.Parameter(torch.tensor(0.5))

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, baskets, mask_positions):
        """
        baskets:        [B, L, I]
        mask_positions: [B]
        Returns:        logits [B, I]
        """
        B, L, I = baskets.shape

        token_emb = self.basket_proj(baskets)  # [B, L, H]
        pos_ids = torch.arange(L, device=baskets.device).unsqueeze(0)
        token_emb = token_emb + self.pos_embedding(pos_ids)
        token_emb = self.dropout(token_emb)

        encoded = self.encoder(token_emb)  # [B, L, H]
        encoded = torch.nan_to_num(encoded, nan=0.0, posinf=0.0, neginf=0.0)

        mask_idx = mask_positions.view(B, 1, 1).expand(B, 1, self.hidden_dim)
        masked_hidden = encoded.gather(1, mask_idx).squeeze(1)  # [B, H]

        logits = self.decoder(masked_hidden)  # [B, I]
        freq = baskets.sum(dim=1)  # [B, I]
        freq_score = self.alpha_freq * torch.log1p(freq)
        logits = logits + freq_score

        return torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)



def listwise_ce_loss(logits, y):
    y = y.float()
    y_norm = y / y.sum(dim=1, keepdim=True).clamp(min=1.0)
    log_probs = F.log_softmax(logits, dim=1)
    return -(y_norm * log_probs).sum(dim=1).mean()



@torch.no_grad()
def evaluate_metrics(model, loader, device, ks=(10, 100)):
    model.eval()

    totals = {f"Recall@{k}": 0.0 for k in ks}
    totals.update({f"NDCG@{k}": 0.0 for k in ks})
    totals.update({f"UN@{k}": 0.0 for k in ks})
    n_users = 0

    for baskets, target, mask_pos in loader:
        baskets = baskets.to(device).float()
        target = target.to(device).float()
        mask_pos = mask_pos.to(device)

        logits = model(baskets, mask_pos)
        seen_mask = baskets.sum(dim=1) > 0  # [B, I]
        y_true = (target > 0.5).float()
        n_users += target.size(0)

        max_k = min(max(ks), logits.size(1))
        topk_all = torch.topk(logits, k=max_k, dim=1).indices

        for k in ks:
            k_eff = min(k, logits.size(1))
            topk = topk_all[:, :k_eff]

            hits = y_true.gather(1, topk).sum(dim=1)
            denom = y_true.sum(dim=1).clamp(min=1.0)
            totals[f"Recall@{k}"] += (hits / denom).sum().item()

            discounts = 1.0 / torch.log2(
                torch.arange(k_eff, device=device).float() + 2.0
            )
            dcg = (y_true.gather(1, topk) * discounts).sum(dim=1)
            ideal_len = torch.minimum(denom.long(), torch.tensor(k_eff, device=device))
            idcg = torch.zeros_like(dcg)
            for i in range(target.size(0)):
                if ideal_len[i] > 0:
                    idcg[i] = discounts[:ideal_len[i]].sum()
            totals[f"NDCG@{k}"] += (dcg / idcg.clamp(min=1e-8)).sum().item()

            novel = (~seen_mask).gather(1, topk).float().mean(dim=1)
            totals[f"UN@{k}"] += novel.sum().item()

    for key in totals:
        totals[key] /= max(n_users, 1)

    return totals



def train_bert4rec(
    dataset_name,
    seq_len=5,
    epochs=20,
    batch_size=64,
    lr=1e-3,
    weight_decay=1e-4,
    proj_dim=256,
    hidden_dim=128,
    nhead=4,
    num_layers=2,
    dropout=0.1,
):
    device = get_device()
    print(f"\n=== BERT4Rec-NBR on {dataset_name} | device={device} ===")

    user_hist, item2idx, _ = load_preprocessed(dataset_name)
    num_items = len(item2idx)

    train_ds_full = BERT4RecNBRDataset(user_hist, item2idx, seq_len=seq_len, mode="train")
    val_ds_full   = BERT4RecNBRDataset(user_hist, item2idx, seq_len=seq_len, mode="eval")

    n_total = len(train_ds_full)
    n_val = min(max(1000, int(0.1 * n_total)), n_total - 1)
    n_train = n_total - n_val
    train_idx = list(range(n_train))
    val_idx   = list(range(n_train, n_total))

    train_ds = Subset(train_ds_full, train_idx)
    val_ds   = Subset(val_ds_full,   val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"  train={len(train_ds)}, val={len(val_ds)}, items={num_items}")

    model = BERT4RecNBR(
        num_items=num_items,
        seq_len=seq_len,
        proj_dim=proj_dim,
        hidden_dim=hidden_dim,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.1,       # 10% эпох на warmup
        anneal_strategy="cos",
    )

    best_recall10 = -1.0
    best_metrics  = None
    best_path = f"data/{dataset_name}_bert4rec_nbr.pt"
    os.makedirs("data", exist_ok=True)
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for baskets, target, mask_pos in train_loader:
            baskets  = baskets.to(device).float()
            target   = target.to(device).float()
            mask_pos = mask_pos.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(baskets, mask_pos)
            loss = listwise_ce_loss(logits, target)

            if not torch.isfinite(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * target.size(0)

        avg_loss = total_loss / max(len(train_ds), 1)
        metrics  = evaluate_metrics(model, val_loader, device=device)

        row = {"epoch": epoch, "train_loss": avg_loss, **metrics}
        history.append(row)

        print(
            f"Epoch {epoch:02d} | {time.time()-t0:.0f}s | loss={avg_loss:.4f} | "
            f"R@10={metrics['Recall@10']:.4f} | R@100={metrics['Recall@100']:.4f} | "
            f"NDCG@10={metrics['NDCG@10']:.4f} | "
            f"UN@100={metrics['UN@100']:.4f} |" f"NDCG@100={metrics['NDCG@100']:.4f} "
        )

        if metrics["Recall@10"] > best_recall10:
            best_recall10 = metrics["Recall@10"]
            best_metrics  = metrics.copy()
            torch.save(
                {"state_dict": model.state_dict(),
                 "num_items": num_items, "seq_len": seq_len},
                best_path,
            )
            print(f"  ✓ Best checkpoint (Recall@10={best_recall10:.4f})")

    pd.DataFrame(history).to_csv(
        f"data/history_bert4rec_{dataset_name}.csv", index=False
    )

    if best_metrics is None:
        best_metrics = evaluate_metrics(model, val_loader, device=device)

    return best_metrics



if __name__ == "__main__":
    results = []

    for dataset_name in ["tafeng", "dunnhumby"]:
        metrics = train_bert4rec(
            dataset_name=dataset_name,
            seq_len=5,
            epochs=20,
            batch_size=64,
            lr=1e-3,
            weight_decay=1e-4,
            proj_dim=256,
            hidden_dim=128,
            nhead=4,
            num_layers=2,
            dropout=0.1,
        )
        row = {"Dataset": dataset_name, "Model": "BERT4Rec-NBR"}
        row.update(metrics)
        results.append(row)

    df = pd.DataFrame(results)
    print("\nFinal Results")
    print(df.round(4))
    df.to_csv("data/results_bert4rec_nbr.csv", index=False)
    print("Saved to data/results_bert4rec_nbr.csv")
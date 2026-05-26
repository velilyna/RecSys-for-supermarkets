import os
import math
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from data import load_preprocessed, get_device


PAD = 0
BOS = 1
EOS = 2
SEP = 3
ITEM_OFFSET = 4


class GenerativeNextBasketDataset(Dataset):

    def __init__(
        self,
        user_hist,
        item2idx,
        mode="train",
        max_history_baskets=32,
        max_basket_len=20,
        max_context_tokens=256,
        max_target_len=20,
        max_samples=None,
        seed=42,
    ):
        self.item2idx = item2idx
        self.num_items = len(item2idx)
        self.vocab_size = self.num_items + ITEM_OFFSET

        self.mode = mode
        self.max_history_baskets = max_history_baskets
        self.max_basket_len = max_basket_len
        self.max_context_tokens = max_context_tokens
        self.max_target_len = max_target_len

        samples = []

        for user_id, seq in user_hist.items():
            if len(seq) < 2:
                continue

            if mode == "train":
                target_indices = list(range(1, len(seq) - 1))
            elif mode == "val":
                target_indices = [len(seq) - 1]
            else:
                raise ValueError(f"Unknown mode: {mode}")

            for i in target_indices:
                context_seq = seq[:i]
                target_basket = seq[i][1]

                context_tokens = self.encode_context(context_seq)
                target_tokens = self.encode_basket(target_basket)

                if len(context_tokens) == 0 or len(target_tokens) == 0:
                    continue

                samples.append((context_tokens, target_tokens))

        random.Random(seed).shuffle(samples)

        if max_samples is not None:
            samples = samples[:max_samples]

        self.samples = samples

        print(
            f"GenerativeNextBasketDataset(mode={mode}): "
            f"samples={len(self.samples)}, items={self.num_items}, "
            f"vocab={self.vocab_size}, max_history_baskets={self.max_history_baskets}"
        )

    def encode_basket(self, basket):
        token_ids = []

        for item in basket:
            if item in self.item2idx:
                token_ids.append(self.item2idx[item] + ITEM_OFFSET)

        token_ids = sorted(set(token_ids))
        token_ids = token_ids[: self.max_basket_len]

        return token_ids

    def encode_context(self, context_seq):
        context_seq = context_seq[-self.max_history_baskets :]

        tokens = [BOS]

        for _, basket in context_seq:
            basket_tokens = self.encode_basket(basket)

            if len(basket_tokens) == 0:
                continue

            tokens.extend(basket_tokens)
            tokens.append(SEP)

        if len(tokens) > 1 and tokens[-1] == SEP:
            tokens[-1] = EOS
        else:
            tokens.append(EOS)

        if len(tokens) > self.max_context_tokens:
            tokens = [BOS] + tokens[-(self.max_context_tokens - 1) :]

        return tokens

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        context_tokens, target_items = self.samples[idx]

        target_items = target_items[: self.max_target_len]

        enc = context_tokens
        dec_in = [BOS] + target_items
        dec_tgt = target_items + [EOS]

        return (
            torch.tensor(enc, dtype=torch.long),
            torch.tensor(dec_in, dtype=torch.long),
            torch.tensor(dec_tgt, dtype=torch.long),
        )


def collate_batch(batch):
    enc_list, dec_in_list, dec_tgt_list = zip(*batch)

    enc = nn.utils.rnn.pad_sequence(
        enc_list,
        batch_first=True,
        padding_value=PAD,
    )

    dec_in = nn.utils.rnn.pad_sequence(
        dec_in_list,
        batch_first=True,
        padding_value=PAD,
    )

    dec_tgt = nn.utils.rnn.pad_sequence(
        dec_tgt_list,
        batch_first=True,
        padding_value=PAD,
    )

    return enc, dec_in, dec_tgt


class GenerativeBasketTransformer(nn.Module):

    def __init__(
        self,
        vocab_size,
        d_model=128,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=512,
        dropout=0.1,
        max_len=512,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len

        self.token_emb = nn.Embedding(
            vocab_size,
            d_model,
            padding_idx=PAD,
        )

        self.pos_emb = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            enable_nested_tensor=False,
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
        )

        self.output_proj = nn.Linear(d_model, vocab_size, bias=True)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)

        nn.init.xavier_uniform_(self.output_proj.weight)

        if self.output_proj.bias is not None:
            nn.init.zeros_(self.output_proj.bias)

        with torch.no_grad():
            self.token_emb.weight[PAD].zero_()

    def add_pos(self, tokens):
        batch_size, seq_len = tokens.shape

        if seq_len > self.max_len:
            raise ValueError(
                f"seq_len={seq_len} > max_len={self.max_len}. "
                f"Increase max_len or reduce max_context_tokens."
            )

        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0)

        x = self.token_emb(tokens) * math.sqrt(self.d_model)
        x = x + self.pos_emb(positions)

        return x

    def make_causal_mask(self, seq_len, device):
        return torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
            diagonal=1,
        )

    def encode(self, enc_tokens):
        src_key_padding_mask = enc_tokens.eq(PAD)
        src = self.add_pos(enc_tokens)

        memory = self.encoder(
            src,
            src_key_padding_mask=src_key_padding_mask,
        )

        return memory, src_key_padding_mask

    def decode(self, dec_in_tokens, memory, memory_key_padding_mask):
        tgt_key_padding_mask = dec_in_tokens.eq(PAD)
        tgt = self.add_pos(dec_in_tokens)

        tgt_mask = self.make_causal_mask(
            dec_in_tokens.size(1),
            dec_in_tokens.device,
        )

        out = self.decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        logits = self.output_proj(out)
        return logits

    def forward(self, enc_tokens, dec_in_tokens):
        memory, src_key_padding_mask = self.encode(enc_tokens)

        logits = self.decode(
            dec_in_tokens,
            memory,
            src_key_padding_mask,
        )

        return logits

    def build_personal_frequency_scores(self, enc_tokens):
        batch_size = enc_tokens.size(0)
        device = enc_tokens.device

        freq = torch.zeros(
            batch_size,
            self.vocab_size,
            dtype=torch.float32,
            device=device,
        )

        valid_item_mask = enc_tokens >= ITEM_OFFSET
        safe_tokens = enc_tokens.masked_fill(~valid_item_mask, 0)

        ones = valid_item_mask.float()

        freq.scatter_add_(1, safe_tokens, ones)

        # Спец-токены не должны получать boost.
        freq[:, PAD] = 0.0
        freq[:, BOS] = 0.0
        freq[:, EOS] = 0.0
        freq[:, SEP] = 0.0

        return torch.log1p(freq)

    @torch.no_grad()
    def generate_ranked_items(
        self,
        enc_tokens,
        max_k=100,
        rerank_alpha=2.0,
    ):
        self.eval()

        batch_size = enc_tokens.size(0)
        device = enc_tokens.device

        memory, src_key_padding_mask = self.encode(enc_tokens)

        generated_seq = torch.full(
            (batch_size, 1),
            BOS,
            dtype=torch.long,
            device=device,
        )

        ranked_items = []

        banned = torch.zeros(
            batch_size,
            self.vocab_size,
            dtype=torch.bool,
            device=device,
        )

        banned[:, PAD] = True
        banned[:, BOS] = True
        banned[:, EOS] = True
        banned[:, SEP] = True

        personal_freq_scores = self.build_personal_frequency_scores(enc_tokens)

        for _ in range(max_k):
            logits = self.decode(
                generated_seq,
                memory,
                src_key_padding_mask,
            )

            next_logits = logits[:, -1, :]

            if rerank_alpha != 0.0:
                next_logits = next_logits + rerank_alpha * personal_freq_scores

            next_logits = next_logits.masked_fill(banned, -1e9)

            next_token = torch.argmax(next_logits, dim=-1)

            ranked_items.append(next_token)

            banned.scatter_(1, next_token.unsqueeze(1), True)

            generated_seq = torch.cat(
                [generated_seq, next_token.unsqueeze(1)],
                dim=1,
            )

        ranked_items = torch.stack(ranked_items, dim=1)

        return ranked_items


def train_loss_fn(logits, target):
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target.reshape(-1),
        ignore_index=PAD,
        reduction="mean",
    )


def target_items_from_dec_tgt(dec_tgt):
    result = []

    for i in range(dec_tgt.size(0)):
        tokens = dec_tgt[i]
        tokens = tokens[
            (tokens != PAD)
            & (tokens != BOS)
            & (tokens != EOS)
            & (tokens != SEP)
        ]

        result.append(set(tokens.detach().cpu().tolist()))

    return result


def seen_items_from_enc(enc):
    result = []

    for i in range(enc.size(0)):
        tokens = enc[i]
        tokens = tokens[
            (tokens != PAD)
            & (tokens != BOS)
            & (tokens != EOS)
            & (tokens != SEP)
        ]

        result.append(set(tokens.detach().cpu().tolist()))

    return result


def compute_ranking_metrics(ranked_items, dec_tgt, enc, ks=(10, 20, 100)):
    batch_size = ranked_items.size(0)

    target_sets = target_items_from_dec_tgt(dec_tgt)
    seen_sets = seen_items_from_enc(enc)

    metrics = {}

    for k in ks:
        recall_sum = 0.0
        ndcg_sum = 0.0
        un_sum = 0.0
        valid_users = 0

        k_eff = min(k, ranked_items.size(1))

        discounts = [
            1.0 / math.log2(pos + 2.0)
            for pos in range(k_eff)
        ]

        for i in range(batch_size):
            target_set = target_sets[i]

            if len(target_set) == 0:
                continue

            preds = ranked_items[i, :k_eff].detach().cpu().tolist()
            seen_set = seen_sets[i]

            hits = [1 if item in target_set else 0 for item in preds]

            recall = sum(hits) / len(target_set)

            dcg = sum(h * discounts[pos] for pos, h in enumerate(hits))

            ideal_len = min(len(target_set), k_eff)
            idcg = sum(discounts[:ideal_len])

            ndcg = dcg / max(idcg, 1e-8)

            novel_count = sum(1 for item in preds if item not in seen_set)
            un = novel_count / k_eff

            recall_sum += recall
            ndcg_sum += ndcg
            un_sum += un
            valid_users += 1

        metrics[f"Recall@{k}"] = recall_sum / max(valid_users, 1)
        metrics[f"NDCG@{k}"] = ndcg_sum / max(valid_users, 1)
        metrics[f"UN@{k}"] = un_sum / max(valid_users, 1)

    return metrics


@torch.no_grad()
def evaluate(model, loader, device, ks=(10, 20, 100), rerank_alpha=1.0):
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    metric_sums = {}
    n_batches = 0

    max_k = max(ks)

    for enc, dec_in, dec_tgt in loader:
        enc = enc.to(device)
        dec_in = dec_in.to(device)
        dec_tgt = dec_tgt.to(device)

        logits = model(enc, dec_in)

        if not torch.isfinite(logits).all():
            print("[WARN] non-finite logits in eval, skip batch")
            continue

        loss_sum = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            dec_tgt.reshape(-1),
            ignore_index=PAD,
            reduction="sum",
        )

        n_tokens = dec_tgt.ne(PAD).sum().item()

        ranked_items = model.generate_ranked_items(
            enc,
            max_k=max_k,
            rerank_alpha=rerank_alpha,
        )

        batch_metrics = compute_ranking_metrics(
            ranked_items=ranked_items,
            dec_tgt=dec_tgt,
            enc=enc,
            ks=ks,
        )

        total_loss += loss_sum.item()
        total_tokens += n_tokens

        for key, value in batch_metrics.items():
            metric_sums[key] = metric_sums.get(key, 0.0) + value

        n_batches += 1

    result = {
        "Loss": total_loss / max(total_tokens, 1),
    }

    for key, value in metric_sums.items():
        result[key] = value / max(n_batches, 1)

    return result


def train_generative_nextbasket(
    dataset_name="tafeng",
    max_history_baskets=32,
    max_basket_len=20,
    max_context_tokens=256,
    max_target_len=20,
    max_train_samples=30000,
    max_val_samples=1000,
    batch_size=32,
    epochs=10,
    lr=5e-4,
    weight_decay=1e-4,
    rerank_alpha=1.0,
    force_cpu=True,
):
    if force_cpu:
        device = torch.device("cpu")
    else:
        device = get_device()

    print(f"Устройство: {str(device).upper()}")

    user_hist, item2idx, _ = load_preprocessed(dataset_name)

    train_dataset = GenerativeNextBasketDataset(
        user_hist=user_hist,
        item2idx=item2idx,
        mode="train",
        max_history_baskets=max_history_baskets,
        max_basket_len=max_basket_len,
        max_context_tokens=max_context_tokens,
        max_target_len=max_target_len,
        max_samples=max_train_samples,
    )

    val_dataset = GenerativeNextBasketDataset(
        user_hist=user_hist,
        item2idx=item2idx,
        mode="val",
        max_history_baskets=max_history_baskets,
        max_basket_len=max_basket_len,
        max_context_tokens=max_context_tokens,
        max_target_len=max_target_len,
        max_samples=max_val_samples,
    )

    if len(train_dataset) == 0:
        raise ValueError("Train dataset is empty.")

    if len(val_dataset) == 0:
        raise ValueError("Validation dataset is empty.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_batch,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_batch,
    )

    model = GenerativeBasketTransformer(
        vocab_size=train_dataset.vocab_size,
        d_model=128,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=512,
        dropout=0.1,
        max_len=max(max_context_tokens, max_target_len + 1) + 2,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    os.makedirs("checkpoints", exist_ok=True)

    ckpt_path = f"checkpoints/generative_nextbasket_{dataset_name}_alpha{rerank_alpha}.pt"

    best_recall = -1.0
    best_metrics = None

    history = []

    print(
        f"\n=== Generative Next-Basket on {dataset_name} ===\n"
        f"train={len(train_dataset)}, val={len(val_dataset)}, "
        f"vocab={train_dataset.vocab_size}, "
        f"history_baskets={max_history_baskets}, "
        f"batch_size={batch_size}, epochs={epochs}, "
        f"rerank_alpha={rerank_alpha}"
    )

    for epoch in range(1, epochs + 1):
        model.train()

        t0 = time.time()

        total_loss = 0.0
        total_batches = 0
        skipped_batches = 0

        for batch_idx, (enc, dec_in, dec_tgt) in enumerate(train_loader, start=1):
            enc = enc.to(device)
            dec_in = dec_in.to(device)
            dec_tgt = dec_tgt.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(enc, dec_in)

            if epoch == 1 and batch_idx == 1:
                print(
                    "DEBUG logits:",
                    "min=", logits.min().item(),
                    "max=", logits.max().item(),
                    "mean=", logits.mean().item(),
                    "std=", logits.std().item(),
                    "finite=", torch.isfinite(logits).all().item(),
                )

            if not torch.isfinite(logits).all():
                print("[WARN] non-finite logits, skip batch")
                skipped_batches += 1
                continue

            loss = train_loss_fn(logits, dec_tgt)

            if not torch.isfinite(loss):
                print("[WARN] non-finite loss, skip batch")
                skipped_batches += 1
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

            if batch_idx % 50 == 0 or batch_idx == len(train_loader):
                elapsed = time.time() - t0
                print(
                    f"  Epoch {epoch:02d} "
                    f"[{batch_idx:4d}/{len(train_loader)}] "
                    f"loss={loss.item():.4f} "
                    f"elapsed={elapsed:.0f}s"
                )

        train_loss = total_loss / max(total_batches, 1)

        print(f"Epoch {epoch:02d} done | TrainLoss={train_loss:.4f} | validation...")

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            ks=(10, 20, 100),
            rerank_alpha=rerank_alpha,
        )

        row = {
            "Epoch": epoch,
            "TrainLoss": train_loss,
            "SkippedBatches": skipped_batches,
            "RerankAlpha": rerank_alpha,
            **val_metrics,
        }

        history.append(row)

        print(
            f"Epoch {epoch:02d} | "
            f"TrainLoss={train_loss:.4f} | "
            f"ValLoss={val_metrics['Loss']:.4f} | "
            f"Recall@10={val_metrics['Recall@10']:.4f} | "
            f"NDCG@10={val_metrics['NDCG@10']:.4f} | "
            f"UN@10={val_metrics['UN@10']:.4f} | "
            f"Recall@20={val_metrics['Recall@20']:.4f} | "
            f"NDCG@20={val_metrics['NDCG@20']:.4f} | "
            f"UN@20={val_metrics['UN@20']:.4f} | "
            f"Recall@100={val_metrics['Recall@100']:.4f} | "
            f"NDCG@100={val_metrics['NDCG@100']:.4f} | "
            f"UN@100={val_metrics['UN@100']:.4f} | "
            f"Skipped={skipped_batches}"
        )

        if val_metrics["Recall@10"] > best_recall:
            best_recall = val_metrics["Recall@10"]
            best_metrics = val_metrics

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "dataset_name": dataset_name,
                    "vocab_size": train_dataset.vocab_size,
                    "num_items": train_dataset.num_items,
                    "item2idx": item2idx,
                    "max_history_baskets": max_history_baskets,
                    "max_basket_len": max_basket_len,
                    "max_context_tokens": max_context_tokens,
                    "max_target_len": max_target_len,
                    "rerank_alpha": rerank_alpha,
                    "best_metrics": best_metrics,
                },
                ckpt_path,
            )

    os.makedirs("data", exist_ok=True)

    hist_path = f"data/history_generative_nextbasket_{dataset_name}_alpha{rerank_alpha}.csv"
    pd.DataFrame(history).to_csv(hist_path, index=False)

    result_path = f"data/results_generative_nextbasket_{dataset_name}_alpha{rerank_alpha}.csv"
    pd.DataFrame(
        [
            {
                "Dataset": dataset_name,
                "Model": "GenerativeNextBasket+FrequencyRerank",
                "RerankAlpha": rerank_alpha,
                **best_metrics,
            }
        ]
    ).to_csv(result_path, index=False)

    return best_metrics


if __name__ == "__main__":
    metrics = train_generative_nextbasket(
        dataset_name="dunnhumby",
        max_history_baskets=32,
        max_basket_len=20,
        max_context_tokens=256,
        max_target_len=20,
        max_train_samples=30000,
        max_val_samples=None,

        batch_size=32,
        epochs=5,
        lr=5e-4,
        weight_decay=1e-4,
        rerank_alpha=0.0,

        force_cpu=True,
    )

    print("\nBest validation metrics")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
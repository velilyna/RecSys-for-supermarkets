import os
import math
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split

from data import load_preprocessed, get_device


PAD = 0
BOS = 1
EOS = 2
ITEM_OFFSET = 3


class BasketCopyDataset(Dataset):

    def __init__(
        self,
        user_hist,
        item2idx,
        max_basket_len=20,
        max_samples=None,
        seed=42,
    ):
        self.item2idx = item2idx
        self.num_items = len(item2idx)
        self.vocab_size = self.num_items + ITEM_OFFSET
        self.max_basket_len = max_basket_len

        samples = []

        for user_id, seq in user_hist.items():
            for ts, basket in seq:
                token_ids = []

                for item in basket:
                    if item in item2idx:
                        token_ids.append(item2idx[item] + ITEM_OFFSET)

                # Убираем дубликаты внутри корзины.
                token_ids = list(dict.fromkeys(token_ids))

                if len(token_ids) == 0:
                    continue

                # Фиксируем порядок, чтобы задача была стабильной.
                token_ids = sorted(token_ids)

                # Ограничиваем длину корзины.
                token_ids = token_ids[:max_basket_len]

                samples.append(token_ids)

        random.Random(seed).shuffle(samples)

        if max_samples is not None:
            samples = samples[:max_samples]

        self.samples = samples

        print(
            f"BasketCopyDataset: samples={len(self.samples)}, "
            f"items={self.num_items}, vocab_size={self.vocab_size}, "
            f"max_basket_len={self.max_basket_len}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        items = self.samples[idx]

        enc = [BOS] + items + [EOS]
        dec_in = [BOS] + items
        dec_tgt = items + [EOS]

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
        max_len=64,
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
                f"Увеличь max_len или уменьши max_basket_len."
            )

        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0)

        x = self.token_emb(tokens) * math.sqrt(self.d_model)
        x = x + self.pos_emb(positions)

        return x

    def make_causal_mask(self, seq_len, device):
        # True = запрещено смотреть в будущее.
        return torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
            diagonal=1,
        )

    def forward(self, enc_tokens, dec_in_tokens):
        src_key_padding_mask = enc_tokens.eq(PAD)
        tgt_key_padding_mask = dec_in_tokens.eq(PAD)

        src = self.add_pos(enc_tokens)
        tgt = self.add_pos(dec_in_tokens)

        tgt_mask = self.make_causal_mask(
            dec_in_tokens.size(1),
            dec_in_tokens.device,
        )

        memory = self.encoder(
            src,
            src_key_padding_mask=src_key_padding_mask,
        )

        out = self.decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )

        logits = self.output_proj(out)

        return logits

    @torch.no_grad()
    def generate(self, enc_tokens, max_new_tokens=30):
        self.eval()

        batch_size = enc_tokens.size(0)
        device = enc_tokens.device

        generated = torch.full(
            (batch_size, 1),
            BOS,
            dtype=torch.long,
            device=device,
        )

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            logits = self.forward(enc_tokens, generated)
            next_logits = logits[:, -1, :]

            next_logits[:, PAD] = -1e9
            next_logits[:, BOS] = -1e9

            next_token = torch.argmax(next_logits, dim=-1)

            next_token = torch.where(
                finished,
                torch.full_like(next_token, EOS),
                next_token,
            )

            generated = torch.cat(
                [generated, next_token.unsqueeze(1)],
                dim=1,
            )

            finished = finished | next_token.eq(EOS)

            if finished.all():
                break

        return generated


def token_accuracy(logits, target):

    pred = logits.argmax(dim=-1)
    mask = target.ne(PAD)

    correct = (pred.eq(target) & mask).sum().item()
    total = mask.sum().item()

    return correct / max(total, 1)


def reconstruction_recall_at_k(logits, target, k=10):

    k_eff = min(k, logits.size(-1))

    topk = torch.topk(logits, k=k_eff, dim=-1).indices

    mask = target.ne(PAD)
    target_expanded = target.unsqueeze(-1)

    hit = topk.eq(target_expanded).any(dim=-1)
    hit = hit & mask

    return hit.sum().item() / max(mask.sum().item(), 1)


def basket_recall_at_k(logits, target, k=10):

    batch_size, seq_len, vocab_size = logits.shape

    scores = logits.clone()
    scores[:, :, PAD] = -1e9
    scores[:, :, BOS] = -1e9
    scores[:, :, EOS] = -1e9

    item_scores = scores.max(dim=1).values  # [B, vocab_size]

    k_eff = min(k, vocab_size)
    topk_items = torch.topk(item_scores, k=k_eff, dim=1).indices

    recalls = []

    for i in range(batch_size):
        target_items = target[i]
        target_items = target_items[
            (target_items != PAD)
            & (target_items != BOS)
            & (target_items != EOS)
        ]

        target_set = set(target_items.detach().cpu().tolist())
        pred_set = set(topk_items[i].detach().cpu().tolist())

        if len(target_set) == 0:
            continue

        hit_count = len(target_set & pred_set)
        recalls.append(hit_count / len(target_set))

    if len(recalls) == 0:
        return 0.0

    return sum(recalls) / len(recalls)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    total_acc = 0.0
    total_recon_r10 = 0.0

    total_basket_recall = 0.0
    total_basket_examples = 0

    for enc, dec_in, dec_tgt in loader:
        enc = enc.to(device)
        dec_in = dec_in.to(device)
        dec_tgt = dec_tgt.to(device)

        logits = model(enc, dec_in)

        if not torch.isfinite(logits).all():
            print("[WARN] non-finite logits in eval, skip batch")
            continue

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            dec_tgt.reshape(-1),
            ignore_index=PAD,
            reduction="sum",
        )

        n_tokens = dec_tgt.ne(PAD).sum().item()

        if n_tokens == 0:
            continue

        acc = token_accuracy(logits, dec_tgt)
        recon_r10 = reconstruction_recall_at_k(logits, dec_tgt, k=10)
        basket_r10 = basket_recall_at_k(logits, dec_tgt, k=10)

        batch_size = enc.size(0)

        total_loss += loss.item()
        total_tokens += n_tokens
        total_acc += acc * n_tokens
        total_recon_r10 += recon_r10 * n_tokens

        total_basket_recall += basket_r10 * batch_size
        total_basket_examples += batch_size

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 20.0))

    return {
        "Loss": avg_loss,
        "PPL": ppl,
        "TokenAcc": total_acc / max(total_tokens, 1),
        "ReconRecall@10": total_recon_r10 / max(total_tokens, 1),
        "BasketRecall@10": total_basket_recall / max(total_basket_examples, 1),
    }


def train_generative_copy(
    dataset_name="tafeng",
    max_basket_len=20,
    max_samples=10000,
    batch_size=32,
    epochs=10,
    lr=5e-4,
    weight_decay=1e-4,
    force_cpu=True,
):
    if force_cpu:
        device = torch.device("cpu")
    else:
        device = get_device()

    print(f"Устройство: {str(device).upper()}")

    user_hist, item2idx, _ = load_preprocessed(dataset_name)

    dataset = BasketCopyDataset(
        user_hist=user_hist,
        item2idx=item2idx,
        max_basket_len=max_basket_len,
        max_samples=max_samples,
    )

    n_total = len(dataset)

    if n_total == 0:
        raise ValueError("BasketCopyDataset пустой.")

    n_val = max(1, int(0.1 * n_total))
    n_val = min(n_val, n_total - 1)
    n_train = n_total - n_val

    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_batch,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_batch,
    )

    model = GenerativeBasketTransformer(
        vocab_size=dataset.vocab_size,
        d_model=128,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=512,
        dropout=0.1,
        max_len=max_basket_len + 2,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    os.makedirs("checkpoints", exist_ok=True)

    ckpt_path = f"checkpoints/generative_copy_{dataset_name}.pt"

    best_loss = float("inf")
    best_metrics = None

    history = []

    print(
        f"\n=== Generative Copy Task on {dataset_name} ===\n"
        f"train={len(train_ds)}, val={len(val_ds)}, "
        f"vocab={dataset.vocab_size}, batch_size={batch_size}, epochs={epochs}"
    )

    for epoch in range(1, epochs + 1):
        model.train()

        t0 = time.time()

        total_loss = 0.0
        total_tokens = 0
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

            loss_sum = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                dec_tgt.reshape(-1),
                ignore_index=PAD,
                reduction="sum",
            )

            n_tokens = dec_tgt.ne(PAD).sum()

            if n_tokens.item() == 0:
                skipped_batches += 1
                continue

            loss = loss_sum / n_tokens.clamp(min=1)

            if not torch.isfinite(loss):
                print("[WARN] non-finite loss, skip batch")
                skipped_batches += 1
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss_sum.item()
            total_tokens += n_tokens.item()

            if batch_idx % 50 == 0 or batch_idx == len(train_loader):
                elapsed = time.time() - t0
                print(
                    f"  Epoch {epoch:02d} "
                    f"[{batch_idx:4d}/{len(train_loader)}] "
                    f"loss={loss.item():.4f} "
                    f"elapsed={elapsed:.0f}s"
                )

        train_loss = total_loss / max(total_tokens, 1)
        val_metrics = evaluate(model, val_loader, device)

        row = {
            "Epoch": epoch,
            "TrainLoss": train_loss,
            "SkippedBatches": skipped_batches,
            **val_metrics,
        }

        history.append(row)

        print(
            f"Epoch {epoch:02d} | "
            f"TrainLoss={train_loss:.4f} | "
            f"ValLoss={val_metrics['Loss']:.4f} | "
            f"PPL={val_metrics['PPL']:.2f} | "
            f"TokenAcc={val_metrics['TokenAcc']:.4f} | "
            f"ReconRecall@10={val_metrics['ReconRecall@10']:.4f} | "
            f"BasketRecall@10={val_metrics['BasketRecall@10']:.4f} | "
            f"Skipped={skipped_batches}"
        )

        if val_metrics["Loss"] < best_loss:
            best_loss = val_metrics["Loss"]
            best_metrics = val_metrics

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "dataset_name": dataset_name,
                    "vocab_size": dataset.vocab_size,
                    "num_items": dataset.num_items,
                    "max_basket_len": max_basket_len,
                    "item2idx": item2idx,
                    "best_metrics": best_metrics,
                },
                ckpt_path,
            )


    os.makedirs("data", exist_ok=True)

    hist_path = f"data/history_generative_copy_{dataset_name}.csv"
    pd.DataFrame(history).to_csv(hist_path, index=False)


    return best_metrics


if __name__ == "__main__":
    metrics = train_generative_copy(
        dataset_name="tafeng",
        max_basket_len=20,
        max_samples=10000,
        batch_size=32,
        epochs=10,
        lr=5e-4,
        weight_decay=1e-4,
        force_cpu=True,
    )

    print("\nBest validation metrics")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
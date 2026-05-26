import os
import sys
import math
import glob
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))

if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from data import load_preprocessed


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

        generator = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(samples), generator=generator).tolist()
        samples = [samples[i] for i in perm]

        if max_samples is not None:
            samples = samples[:max_samples]

        self.samples = samples

        print(
            f"Dataset(mode={mode}): samples={len(self.samples)}, "
            f"items={self.num_items}, vocab={self.vocab_size}"
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
    """
    Та же архитектура, что у обученной generative next-basket модели.
    """

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
        logits = self.decode(dec_in_tokens, memory, src_key_padding_mask)
        return logits


class LearnableHybridScorer(nn.Module):
    """
    Обучаемый hybrid scoring layer

    features:
        0: generative_score
        1: frequency_score
        2: recency_score
        3: popularity_score

    final_score =
        w_gen * gen
      + w_freq * freq
      + w_recency * recency
      + w_pop * pop
      + bias
    """

    def __init__(self):
        super().__init__()

        self.weights = nn.Parameter(
            torch.tensor([1.0, 1.0, 1.0, 0.1], dtype=torch.float32)
        )

        self.bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, gen_score, freq_score, recency_score, popularity_score):
        return (
            self.weights[0] * gen_score
            + self.weights[1] * freq_score
            + self.weights[2] * recency_score
            + self.weights[3] * popularity_score.unsqueeze(0)
            + self.bias
        )


def find_generative_checkpoint(dataset_name="tafeng"):
    patterns = [
        f"checkpoints/generative_nextbasket_{dataset_name}*.pt",
        f"checkpoints/*nextbasket*{dataset_name}*.pt",
        f"checkpoints/*generative*{dataset_name}*.pt",
        f"src/checkpoints/generative_nextbasket_{dataset_name}*.pt",
        f"src/checkpoints/*nextbasket*{dataset_name}*.pt",
        f"src/checkpoints/*generative*{dataset_name}*.pt",
    ]

    candidates = []

    for pattern in patterns:
        candidates.extend(glob.glob(pattern))

    candidates = [
        p for p in candidates
        if "copy" not in os.path.basename(p).lower()
    ]

    candidates = sorted(set(candidates))

    if len(candidates) == 0:
        raise FileNotFoundError(
            "\nНе найден checkpoint генеративной next-basket модели.\n"
            "Проверь:\n"
            "  ls checkpoints\n"
        )

    print("\nFound generative checkpoints:")
    for i, path in enumerate(candidates):
        print(f"  {i}: {path}")

    checkpoint_path = candidates[0]

    return checkpoint_path


def load_generative_model(
    checkpoint_path,
    device,
    max_context_tokens=256,
    max_target_len=20,
):
    ckpt = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    vocab_size = ckpt["vocab_size"]

    model = GenerativeBasketTransformer(
        vocab_size=vocab_size,
        d_model=128,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=512,
        dropout=0.1,
        max_len=max(max_context_tokens, max_target_len + 1) + 2,
    ).to(device)

    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict)
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    return model, ckpt


def build_global_popularity_train_only(user_hist, item2idx, vocab_size):
    pop = torch.zeros(vocab_size, dtype=torch.float32)

    for user_id, seq in user_hist.items():
        train_seq = seq[:-1]

        for _, basket in train_seq:
            for item in basket:
                if item in item2idx:
                    token_id = item2idx[item] + ITEM_OFFSET
                    pop[token_id] += 1.0

    pop = torch.log1p(pop)

    pop[PAD] = 0.0
    pop[BOS] = 0.0
    pop[EOS] = 0.0
    pop[SEP] = 0.0

    if pop.max().item() > 0:
        pop = pop / pop.max()

    return pop


def build_frequency_score(enc_tokens, vocab_size):
    batch_size = enc_tokens.size(0)
    device = enc_tokens.device

    freq = torch.zeros(
        batch_size,
        vocab_size,
        dtype=torch.float32,
        device=device,
    )

    valid_mask = enc_tokens >= ITEM_OFFSET
    safe_tokens = enc_tokens.masked_fill(~valid_mask, 0)

    freq.scatter_add_(1, safe_tokens, valid_mask.float())

    freq[:, PAD] = 0.0
    freq[:, BOS] = 0.0
    freq[:, EOS] = 0.0
    freq[:, SEP] = 0.0

    return torch.log1p(freq)


def build_recency_score(enc_tokens, vocab_size, decay=0.9):
    batch_size, seq_len = enc_tokens.shape
    device = enc_tokens.device

    rec = torch.zeros(
        batch_size,
        vocab_size,
        dtype=torch.float32,
        device=device,
    )

    for b in range(batch_size):
        tokens = enc_tokens[b].detach().cpu().tolist()

        baskets = []
        cur = []

        for tok in tokens:
            if tok in (PAD, BOS):
                continue

            if tok in (SEP, EOS):
                if len(cur) > 0:
                    baskets.append(cur)
                    cur = []
                continue

            if tok >= ITEM_OFFSET:
                cur.append(tok)

        if len(cur) > 0:
            baskets.append(cur)

        n_baskets = len(baskets)

        for idx, basket in enumerate(baskets):
            age = n_baskets - 1 - idx
            weight = decay ** age

            for tok in basket:
                rec[b, tok] += weight

    rec[:, PAD] = 0.0
    rec[:, BOS] = 0.0
    rec[:, EOS] = 0.0
    rec[:, SEP] = 0.0

    return rec


@torch.no_grad()
def get_generative_score(model, enc_tokens):
    """
    One-step generative score:
        decoder input = [BOS]
        output logits over all items
    """
    batch_size = enc_tokens.size(0)
    device = enc_tokens.device

    memory, src_key_padding_mask = model.encode(enc_tokens)

    dec_in = torch.full(
        (batch_size, 1),
        BOS,
        dtype=torch.long,
        device=device,
    )

    logits = model.decode(
        dec_in,
        memory,
        src_key_padding_mask,
    )

    gen_score = logits[:, -1, :]

    return gen_score


def make_target_multihot(dec_tgt, vocab_size):
    """
    dec_tgt: [B, T]
    Возвращает target matrix [B, vocab_size]
    """
    batch_size = dec_tgt.size(0)
    device = dec_tgt.device

    y = torch.zeros(
        batch_size,
        vocab_size,
        dtype=torch.float32,
        device=device,
    )

    valid_mask = dec_tgt >= ITEM_OFFSET
    safe_tokens = dec_tgt.masked_fill(~valid_mask, 0)

    y.scatter_(1, safe_tokens, valid_mask.float())

    y[:, PAD] = 0.0
    y[:, BOS] = 0.0
    y[:, EOS] = 0.0
    y[:, SEP] = 0.0

    return y


def mask_special_scores(scores):
    scores[:, PAD] = -1e9
    scores[:, BOS] = -1e9
    scores[:, EOS] = -1e9
    scores[:, SEP] = -1e9
    return scores


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


def compute_ranking_metrics_from_scores(scores, dec_tgt, enc, ks=(10, 20, 100)):
    ranked_items = torch.topk(
        mask_special_scores(scores.clone()),
        k=min(max(ks), scores.size(1)),
        dim=1,
    ).indices

    batch_size = scores.size(0)

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
def evaluate_learnable_hybrid(
    gen_model,
    hybrid_scorer,
    loader,
    device,
    global_popularity,
    recency_decay=0.9,
    ks=(10, 20, 100),
):
    gen_model.eval()
    hybrid_scorer.eval()

    metric_sums = {}
    n_batches = 0

    global_popularity = global_popularity.to(device)

    for batch_idx, (enc, dec_in, dec_tgt) in enumerate(loader, start=1):
        enc = enc.to(device)
        dec_tgt = dec_tgt.to(device)

        gen_score = get_generative_score(gen_model, enc)
        freq_score = build_frequency_score(enc, gen_model.vocab_size)
        recency_score = build_recency_score(
            enc,
            gen_model.vocab_size,
            decay=recency_decay,
        )

        final_score = hybrid_scorer(
            gen_score=gen_score,
            freq_score=freq_score,
            recency_score=recency_score,
            popularity_score=global_popularity,
        )

        batch_metrics = compute_ranking_metrics_from_scores(
            scores=final_score,
            dec_tgt=dec_tgt,
            enc=enc,
            ks=ks,
        )

        for key, value in batch_metrics.items():
            metric_sums[key] = metric_sums.get(key, 0.0) + value

        n_batches += 1

    result = {}

    for key, value in metric_sums.items():
        result[key] = value / max(n_batches, 1)

    return result


def train_learnable_hybrid(
    dataset_name="tafeng",
    max_history_baskets=32,
    max_basket_len=20,
    max_context_tokens=256,
    max_target_len=20,
    max_train_samples=30000,
    max_val_samples=None,
    batch_size=32,
    epochs=10,
    lr=5e-2,
    weight_decay=1e-4,
    recency_decay=0.9,
):
    device = torch.device("cpu")
    print(f"Device: {device}")

    os.chdir(ROOT_DIR)

    checkpoint_path = find_generative_checkpoint(dataset_name)

    print("\nLoading data...")
    user_hist, item2idx, _ = load_preprocessed(dataset_name)

    print("\nBuilding train dataset...")
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

    print("\nBuilding val dataset...")
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

    print("\nLoading frozen generative Transformer...")
    gen_model, ckpt = load_generative_model(
        checkpoint_path=checkpoint_path,
        device=device,
        max_context_tokens=max_context_tokens,
        max_target_len=max_target_len,
    )

    if gen_model.vocab_size != len(item2idx) + ITEM_OFFSET:
        raise ValueError(
            "Checkpoint vocab_size does not match current processed data."
        )

    print("\nBuilding global popularity without validation leakage...")
    global_popularity = build_global_popularity_train_only(
        user_hist=user_hist,
        item2idx=item2idx,
        vocab_size=gen_model.vocab_size,
    ).to(device)

    hybrid_scorer = LearnableHybridScorer().to(device)

    optimizer = torch.optim.AdamW(
        hybrid_scorer.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    pos_weight_value = 50.0
    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=device)

    best_recall = -1.0
    best_metrics = None
    history = []

    os.makedirs("checkpoints", exist_ok=True)

    best_path = f"checkpoints/learnable_hybrid_{dataset_name}.pt"

    print("\n=== Training Learnable Hybrid Scorer ===")
    print(f"train={len(train_dataset)}, val={len(val_dataset)}")
    print(f"initial weights={hybrid_scorer.weights.detach().cpu().tolist()}")
    print(f"initial bias={hybrid_scorer.bias.item():.4f}")

    for epoch in range(1, epochs + 1):
        hybrid_scorer.train()
        gen_model.eval()

        t0 = time.time()

        total_loss = 0.0
        total_batches = 0

        for batch_idx, (enc, dec_in, dec_tgt) in enumerate(train_loader, start=1):
            enc = enc.to(device)
            dec_tgt = dec_tgt.to(device)

            with torch.no_grad():
                gen_score = get_generative_score(gen_model, enc)
                freq_score = build_frequency_score(enc, gen_model.vocab_size)
                recency_score = build_recency_score(
                    enc,
                    gen_model.vocab_size,
                    decay=recency_decay,
                )

            y = make_target_multihot(dec_tgt, gen_model.vocab_size)

            final_score = hybrid_scorer(
                gen_score=gen_score,
                freq_score=freq_score,
                recency_score=recency_score,
                popularity_score=global_popularity,
            )

            final_score[:, PAD] = 0.0
            final_score[:, BOS] = 0.0
            final_score[:, EOS] = 0.0
            final_score[:, SEP] = 0.0

            loss = F.binary_cross_entropy_with_logits(
                final_score,
                y,
                pos_weight=pos_weight,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

            if batch_idx % 100 == 0 or batch_idx == len(train_loader):
                elapsed = time.time() - t0
                print(
                    f"  Epoch {epoch:02d} "
                    f"[{batch_idx:4d}/{len(train_loader)}] "
                    f"loss={loss.item():.4f} "
                    f"elapsed={elapsed:.0f}s"
                )

        train_loss = total_loss / max(total_batches, 1)

        val_metrics = evaluate_learnable_hybrid(
            gen_model=gen_model,
            hybrid_scorer=hybrid_scorer,
            loader=val_loader,
            device=device,
            global_popularity=global_popularity,
            recency_decay=recency_decay,
            ks=(10, 20, 100),
        )

        weights = hybrid_scorer.weights.detach().cpu().tolist()
        bias = hybrid_scorer.bias.item()

        row = {
            "Epoch": epoch,
            "TrainLoss": train_loss,
            "w_gen": weights[0],
            "w_freq": weights[1],
            "w_recency": weights[2],
            "w_pop": weights[3],
            "bias": bias,
            **val_metrics,
        }

        history.append(row)

        print(
            f"Epoch {epoch:02d} | "
            f"TrainLoss={train_loss:.4f} | "
            f"Recall@10={val_metrics['Recall@10']:.4f} | "
            f"NDCG@10={val_metrics['NDCG@10']:.4f} | "
            f"UN@10={val_metrics['UN@10']:.4f} | "
            f"Recall@100={val_metrics['Recall@100']:.4f} |"
            f"UN@100={val_metrics['UN@100']:.4f} | "
            f"NDCG@100={val_metrics['NDCG@100']:.4f} | "
        )

        print(
            f"  weights: "
            f"gen={weights[0]:.4f}, "
            f"freq={weights[1]:.4f}, "
            f"recency={weights[2]:.4f}, "
            f"pop={weights[3]:.4f}, "
            f"bias={bias:.4f}"
        )

        if val_metrics["Recall@10"] > best_recall:
            best_recall = val_metrics["Recall@10"]
            best_metrics = val_metrics

            torch.save(
                {
                    "hybrid_state_dict": hybrid_scorer.state_dict(),
                    "generative_checkpoint": checkpoint_path,
                    "dataset_name": dataset_name,
                    "vocab_size": gen_model.vocab_size,
                    "recency_decay": recency_decay,
                    "best_metrics": best_metrics,
                    "weights": weights,
                    "bias": bias,
                },
                best_path,
            )

            print(f"  ✓ Best hybrid checkpoint saved: {best_path}")

    os.makedirs("data", exist_ok=True)

    hist_path = f"data/history_learnable_hybrid_{dataset_name}.csv"
    pd.DataFrame(history).to_csv(hist_path, index=False)

    result_row = {
        "Dataset": dataset_name,
        "Model": "GenerativeTransformer+LearnableHybridScoring",
        "GenerativeCheckpoint": checkpoint_path,
        "HybridCheckpoint": best_path,
        "recency_decay": recency_decay,
        **best_metrics,
    }

    result_path = f"data/results_learnable_hybrid_{dataset_name}.csv"
    pd.DataFrame([result_row]).to_csv(result_path, index=False)

    print("\n=== Best Learnable Hybrid Results ===")
    print(pd.DataFrame([result_row]).round(4))
    print(f"\nHistory saved to: {hist_path}")
    print(f"Results saved to: {result_path}")

    return result_row


if __name__ == "__main__":
    train_learnable_hybrid(
        dataset_name="dunnhumby",
        max_history_baskets=32,
        max_basket_len=20,
        max_context_tokens=256,
        max_target_len=20,

        max_train_samples=30000,

        max_val_samples=None,

        batch_size=32,
        epochs=10,
        lr=5e-2,
        weight_decay=1e-4,
        recency_decay=0.9,
    )
from typing import Callable, Sequence, Tuple, Union

import torch


TensorOrTupleLogits = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


@torch.no_grad()
def _logits_only(model_forward: Callable[..., TensorOrTupleLogits], x_batch: torch.Tensor) -> torch.Tensor:
    out = model_forward(x_batch)
    if isinstance(out, tuple):
        return out[0]
    return out


@torch.no_grad()
def ranking_metrics_multihot(
    loader,
    device,
    model_forward: Callable[..., TensorOrTupleLogits],
    *,
    recall_ks: Sequence[int] = (1, 2, 3, 4, 5),
    ndcg_ks: Sequence[int] = (1, 2, 3, 4),
    max_batches: int | None = None,
) -> dict:
    """
    Recall@K и NDCG@K для next-basket (multi-hot target), согласовано с train_transformer.evaluate_metrics:

    Recall@K: сколько позиций из target попало в top-K / число позиций в target.
    NDCG@K: бинарный gain по релевантным позициям в top-K, idcg = сумма скидок для min(|target|, K) релевантных.
    """
    max_k_need = max(max(recall_ks), max(ndcg_ks))

    recall_sums = {k: 0.0 for k in recall_ks}
    ndcg_sums = {k: 0.0 for k in ndcg_ks}
    n_users = 0

    batches_done = 0
    for hist_baskets, y in loader:
        if max_batches is not None and batches_done >= max_batches:
            break
        batches_done += 1

        hist_baskets = torch.nan_to_num(
            hist_baskets.to(device).float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        y = torch.nan_to_num(y.to(device).float(), nan=0.0, posinf=0.0, neginf=0.0)

        logits = torch.nan_to_num(
            _logits_only(model_forward, hist_baskets),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        k_top = min(max_k_need, logits.size(1))
        topk = torch.topk(logits, k=k_top, dim=1).indices

        y_true = (y > 0.5).float()
        bsz = y.size(0)
        n_users += bsz

        for K in recall_ks:
            k_eff = min(K, logits.size(1))
            preds = topk[:, :k_eff]
            hits = y_true.gather(1, preds).sum(dim=1)
            denom = y_true.sum(dim=1).clamp(min=1.0)
            recall_sums[K] += (hits / denom).sum().item()

        discounts_full = 1.0 / torch.log2(
            torch.arange(2, k_top + 2, device=y.device, dtype=torch.float32)
        )
        for K in ndcg_ks:
            k_eff = min(K, logits.size(1))
            preds = topk[:, :k_eff]
            gains = y_true.gather(1, preds)
            discounts = discounts_full[:k_eff]
            dcg = (gains * discounts.unsqueeze(0)).sum(dim=1)

            num_relevant = y_true.sum(dim=1).clamp(max=k_eff)
            idcg = torch.zeros_like(dcg)
            for i in range(k_eff):
                idcg += (i < num_relevant).float() * discounts[i]
            idcg = idcg.clamp(min=1e-6)
            ndcg = dcg / idcg
            ndcg_sums[K] += ndcg.sum().item()

    denom_u = max(n_users, 1)
    out: dict[str, float] = {}
    for K in recall_ks:
        out[f"Recall@{K}"] = recall_sums[K] / denom_u
    for K in ndcg_ks:
        out[f"NDCG@{K}"] = ndcg_sums[K] / denom_u
    return out


@torch.no_grad()
def eval_all_metrics(model, loader, device="cpu", K_list=(10, 100)):
    model.eval()

    rec_sum = {K: 0.0 for K in K_list}
    ndcg_sum = {K: 0.0 for K in K_list}
    un_sum = {K: 0.0 for K in K_list}
    n = 0

    for hist_baskets, y in loader:
        hist_baskets = hist_baskets.to(device).float()
        y = y.to(device).float()

        logits, _ = model(hist_baskets)

        # Для NBR не маскируем уже купленные товары в логитах
        seen_mask = hist_baskets.sum(dim=1) > 0

        y_true = (y > 0.5).float()

        for K in K_list:
            k_eff = min(K, logits.size(1))
            topk = torch.topk(logits, k=k_eff, dim=1).indices

            hits = y_true.gather(1, topk).sum(dim=1)
            denom = y_true.sum(dim=1).clamp(min=1.0)
            rec = hits / denom

            gains = y_true.gather(1, topk)
            discounts = 1.0 / torch.log2(torch.arange(2, k_eff + 2, device=y.device).float())
            dcg = (gains * discounts).sum(dim=1)

            ideal_k = torch.topk(y_true, k=min(k_eff, y_true.size(1)), dim=1).values
            idcg = (ideal_k * discounts[: ideal_k.size(1)]).sum(dim=1).clamp(min=1e-6)
            ndcg = dcg / idcg

            rec_hist_flags = seen_mask.gather(1, topk).float()
            un = 1.0 - rec_hist_flags.mean(dim=1)

            rec_sum[K] += rec.sum().item()
            ndcg_sum[K] += ndcg.sum().item()
            un_sum[K] += un.sum().item()

        n += y.size(0)

    out = {}
    for K in K_list:
        out[f"Recall@{K}"] = rec_sum[K] / n
        out[f"NDCG@{K}"] = ndcg_sum[K] / n
        out[f"UN@{K}"] = un_sum[K] / n

    return out

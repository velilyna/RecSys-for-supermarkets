
import torch

@torch.no_grad()
def eval_all_metrics(model, loader, device="cpu", K_list=(10, 100)):
    model.eval()
    # агрегаторы: {K: [sum, count]}
    rec_sum = {K: 0.0 for K in K_list}
    ndcg_sum = {K: 0.0 for K in K_list}
    un_sum = {K: 0.0 for K in K_list}
    n = 0

    for x, y, freq in loader:
        x, y, freq = x.to(device), y.to(device), freq.to(device)
        logits = model(x, freq)
        y_true = (y > 0.5).float()

        for K in K_list:
            topk = torch.topk(logits, k=min(K, logits.size(1)), dim=1).indices  # (B, K)

            # Recall@K
            hits = y_true.gather(1, topk).sum(dim=1)
            denom = y_true.sum(dim=1).clamp(min=1.0)
            rec = hits / denom

            # NDCG@K
            gains = y_true.gather(1, topk)
            discounts = 1.0 / torch.log2(torch.arange(2, K + 2, device=y.device).float())
            dcg = (gains * discounts).sum(dim=1)
            # идеальный DCG для данного y (берём K наибольших)
            ideal_k = torch.topk(y_true, k=min(K, y_true.size(1)), dim=1).values
            idcg = (ideal_k * discounts[:ideal_k.size(1)]).sum(dim=1).clamp(min=1e-6)
            ndcg = dcg / idcg

            # UN@K: доля рекомендованных позиций, которых нет в недавней истории.
            # в качестве «недавней истории» используем x (объединение последних L корзин)
            hist = (x > 0.5).float()
            rec_hist_flags = hist.gather(1, topk)  # 1 если товар встречался в истории
            un = 1.0 - rec_hist_flags.mean(dim=1)  # доля «новых» среди Top-K

            rec_sum[K] += rec.sum().item()
            ndcg_sum[K] += ndcg.sum().item()
            un_sum[K] += un.sum().item()

        n += y.size(0)

    # усреднение по пользователям
    out = {}
    for K in K_list:
        out[f"Recall@{K}"] = rec_sum[K] / n
        out[f"NDCG@{K}"] = ndcg_sum[K] / n
        out[f"UN@{K}"] = un_sum[K] / n
    return out

import argparse
import torch

from data import get_device, load_preprocessed
from safer_autoencoder import SAFERecAutoEncoder


def basket_to_multihot(basket, item2idx, num_items):
    x = torch.zeros(num_items, dtype=torch.float32)
    for item in basket:
        if item in item2idx:
            x[item2idx[item]] = 1.0
    return x


def build_user_history_sample(user_hist, item2idx, user_id, seq_len=5):
    seq = user_hist[user_id]
    if len(seq) < seq_len:
        raise ValueError(f"У пользователя {user_id} слишком короткая история: {len(seq)}")

    context   = seq[-seq_len:]
    num_items = len(item2idx)

    hist_baskets = torch.stack(
        [basket_to_multihot(basket, item2idx, num_items) for _, basket in context],
        dim=0
    )  # [L, num_items]

    return hist_baskets.unsqueeze(0)   # [1, L, num_items]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",  choices=["tafeng", "dunnhumby"], default="tafeng")
    parser.add_argument("--user_id",  type=int, required=True)
    parser.add_argument("--k",        type=int, default=10)
    parser.add_argument("--seq_len",  type=int, default=5)
    args = parser.parse_args()

    device = get_device()
    print(f"Устройство: {device.upper()}")

    user_hist, item2idx, _ = load_preprocessed(args.dataset)

    if args.user_id not in user_hist:
        raise KeyError(f"Пользователь {args.user_id} не найден")

    ckpt_path = f"data/{args.dataset}_saferec_ae.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = SAFERecAutoEncoder(
        num_items=ckpt["num_items"],
        seq_len=ckpt.get("seq_len", args.seq_len),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    hist_baskets = build_user_history_sample(
        user_hist=user_hist,
        item2idx=item2idx,
        user_id=args.user_id,
        seq_len=ckpt.get("seq_len", args.seq_len),
    ).to(device)

    with torch.no_grad():
        logits, _ = model(hist_baskets)   # распаковываем (logits, recon_loss)

        seen_mask = hist_baskets.sum(dim=1) > 0
        logits = logits

        topk = torch.topk(logits, k=args.k, dim=1).indices.squeeze(0).tolist()

    inv_item2idx = {v: k for k, v in item2idx.items()}
    rec_items = [inv_item2idx[i] for i in topk]

    print(f"\nTop-{args.k} рекомендации для пользователя {args.user_id}:")
    print(rec_items)


if __name__ == "__main__":
    main()

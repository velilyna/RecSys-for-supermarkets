import argparse
import torch
from data import load_preprocessed
from model import SAFERecAutoEncoder


def build_user_sample(user_hist, item2idx, user_id, seq_len=5, freq_window=20):
    """
    Формирует (x, freq) для конкретного пользователя.
    x — объединение последних seq_len корзин.
    freq — частоты по freq_window корзинам.
    """
    seq = user_hist[user_id]
    if len(seq) <= seq_len:
        raise ValueError(f"У пользователя {user_id} слишком короткая история ({len(seq)}).")

    # последние seq_len корзин — контекст
    prev_baskets = [seq[i][1] for i in range(len(seq) - seq_len, len(seq))]
    # последние freq_window корзин — для частот
    freq_from = max(0, len(seq) - freq_window)
    freq_baskets = [seq[i][1] for i in range(freq_from, len(seq))]

    num_items = len(item2idx)
    # контекст
    x = torch.zeros(num_items)
    for b in prev_baskets:
        for item in b:
            if item in item2idx:
                x[item2idx[item]] = 1.0
    # частоты
    freq = torch.zeros(num_items)
    for b in freq_baskets:
        for item in b:
            if item in item2idx:
                freq[item2idx[item]] += 1.0
    freq = torch.log1p(freq)
    if freq.max() > 0:
        freq /= freq.max()
    return x.unsqueeze(0), freq.unsqueeze(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["tafeng", "dunnhumby"], default="tafeng")
    parser.add_argument("--user_id", type=int, required=True, help="ID пользователя (из исходного датасета)")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seq_len", type=int, default=5)
    parser.add_argument("--freq_window", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    user_hist, item2idx, _ = load_preprocessed(args.dataset)
    if args.user_id not in user_hist:
        raise KeyError(f"Пользователь {args.user_id} не найден в user_hist.")

    ckpt_path = f"data/{args.dataset}_saferec_ae.pt"
    ckpt = torch.load(ckpt_path, map_location=args.device)
    model = SAFERecAutoEncoder(num_items=ckpt["num_items"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(args.device)
    model.eval()

    # Формируем вход пользователя
    x, freq = build_user_sample(user_hist, item2idx, args.user_id,
                                seq_len=args.seq_len, freq_window=args.freq_window)
    x, freq = x.to(args.device), freq.to(args.device)

    with torch.no_grad():
        logits = model(x, freq)
        topk = torch.topk(logits, k=args.k, dim=1).indices.squeeze(0).tolist()

    # возвращаем оригинальные ID товаров
    inv_item2idx = {v: k for k, v in item2idx.items()}
    rec_items = [inv_item2idx[i] for i in topk]

    print(f"\nTop-{args.k} рекомендации для пользователя {args.user_id}:")
    print(rec_items)


if __name__ == "__main__":
    main()
import pickle
import torch
from torch.utils.data import Dataset


def load_preprocessed(dataset_name="tafeng"):
    path = f"data/processed/{dataset_name}_preprocessed.pkl"
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["user_hist"], data["item2idx"], data["user2idx"]


class NextBasketDataset(Dataset):
    """
    weighting:
        'binary' - все предыдущие корзины одинаково важны
        'exp_decay' - экспоненциальное затухание по "возрасту" корзины
        'time_since_last' - чем дольше не покупал, тем выше вес
    """

    def __init__(self, user_hist, item2idx, seq_len=5, weighting="binary", decay=0.8, max_days=14):
        self.samples = []
        self.item2idx = item2idx
        self.seq_len = seq_len
        self.weighting = weighting
        self.decay = decay
        self.max_days = max_days

        for u, seq in user_hist.items():
            if len(seq) <= seq_len:
                continue

            # для каждого шага в истории формируем (контекст, цель)
            for i in range(seq_len, len(seq)):
                context = seq[i - seq_len: i]  # [(ts, basket), ...]
                target = seq[i][1]
                current_ts = seq[i][0]

                item_weights = {}

                # мульти-хот
                if self.weighting == "binary":
                    for _, basket in context:
                        for item in basket:
                            item_weights[item] = 1.0

                # экспоненциальное затухание
                elif self.weighting == "exp_decay":
                    for pos, (_, basket) in enumerate(context):
                        # pos = 0  -старая корзина, pos = seq_len-1 -  свежая
                        w = self.decay ** (self.seq_len - 1 - pos)
                        for item in basket:
                            item_weights[item] = item_weights.get(item, 0.0) + w

                # чем дольше не покупал, тем выше вес
                elif self.weighting == "time_since_last":
                    for ts, basket in context:
                        # считаем разницу в днях
                        try:
                            delta_days = (current_ts - ts).days
                        except Exception:
                            delta_days = 0
                        w = min(1.0, max(0.0, delta_days / float(self.max_days)))
                        for item in basket:
                            item_weights[item] = max(item_weights.get(item, 0.0), w)

                else:
                    raise ValueError(f"Unknown weighting mode: {self.weighting}")

                self.samples.append((item_weights, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item_weights, target = self.samples[idx]
        I = len(self.item2idx)
        x = torch.zeros(I)
        y = torch.zeros(I)

        for it, w in item_weights.items():
            j = self.item2idx.get(it)
            if j is not None:
                x[j] = float(w)

        for it in target:
            j = self.item2idx.get(it)
            if j is not None:
                y[j] = 1.0

        return x, y
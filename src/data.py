import os
import pickle
import torch
from torch.utils.data import Dataset


def get_device():
    """M3 Mac: используем MPS если доступен, иначе CPU."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_preprocessed(dataset_name="tafeng"):
    path = os.path.join(project_root(), "data", "processed", f"{dataset_name}_preprocessed.pkl")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Нет файла препроцессинга: {path}\n"
            "Запустите prepare/preprocessing из корня проекта или положите .pkl в data/processed/."
        )
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["user_hist"], data["item2idx"], data["user2idx"]


class NextBasketDataset(Dataset):
    """
    Возвращает:
        hist_baskets: [L, num_items]   - последние L корзин (multi-hot или взвешенные)
        target_basket: [num_items]     - следующая корзина (multi-hot)
    """

    def __init__(
        self,
        user_hist,
        item2idx,
        seq_len=5,
        weighting="binary",
        decay=0.85,
        max_days=14,
    ):
        self.samples = []
        self.item2idx = item2idx
        self.seq_len = seq_len
        self.num_items = len(item2idx)
        self.weighting = weighting
        self.decay = decay
        self.max_days = max_days

        for user_id, seq in user_hist.items():
            if len(seq) <= seq_len:
                continue
            for i in range(seq_len, len(seq)):
                context = seq[i - seq_len:i]
                target = seq[i][1]
                self.samples.append((context, target))

    def __len__(self):
        return len(self.samples)

    def _basket_to_multihot(self, basket, weight=1.0):
        x = torch.zeros(self.num_items, dtype=torch.float32)
        for item in basket:
            idx = self.item2idx.get(item)
            if idx is not None:
                x[idx] = weight
        return x

    def _build_hist_baskets(self, context):
        if self.weighting == "binary":
            return torch.stack(
                [self._basket_to_multihot(basket) for _, basket in context],
                dim=0,
            )
        elif self.weighting == "frequency":
            frames = []
            for _, basket in context:
                x = torch.zeros(self.num_items, dtype=torch.float32)
                for item in basket:
                    idx = self.item2idx.get(item)
                    if idx is not None:
                        x[idx] += 1.0
                frames.append(x)
            return torch.stack(frames, dim=0)
        elif self.weighting == "recency":
            last_ts = context[-1][0]
            frames = []
            for ts, basket in context:
                try:
                    delta = (last_ts - ts).days
                    weight = self.decay ** (delta / max(self.max_days, 1))
                except Exception:
                    weight = 1.0
                frames.append(self._basket_to_multihot(basket, weight=weight))
            return torch.stack(frames, dim=0)
        else:
            raise ValueError(
                f"Неизвестный weighting='{self.weighting}'. "
                f"Допустимые: 'binary', 'frequency', 'recency'."
            )

    def __getitem__(self, idx):
        context, target = self.samples[idx]
        hist_baskets = self._build_hist_baskets(context)
        target_basket = self._basket_to_multihot(target)
        return hist_baskets, target_basket


def get_or_build_dataset(dataset_name, seq_len=5, weighting="binary",
                         decay=0.85, max_days=14, cache_dir=None):
    """
    Строит датасет и кеширует на диск.
    При повторном запуске загружает из кеша — намного быстрее.

    По умолчанию кеш в <project>/data/cache/ (не зависит от текущего cwd).
    """
    if cache_dir is None:
        cache_dir = os.path.join(project_root(), "data", "cache")

    os.makedirs(cache_dir, exist_ok=True)
    cache_path = f"{cache_dir}/{dataset_name}_seq{seq_len}_{weighting}.pt"

    if os.path.exists(cache_path):
        print(f"  Загружаем датасет из кеша: {cache_path}")
        return torch.load(cache_path, weights_only=False)

    print(f"  Строим датасет (первый раз — займёт пару минут, потом мгновенно)...")
    user_hist, item2idx, user2idx = load_preprocessed(dataset_name)
    dataset = NextBasketDataset(
        user_hist, item2idx,
        seq_len=seq_len,
        weighting=weighting,
        decay=decay,
        max_days=max_days,
    )
    torch.save(dataset, cache_path)
    print(f"  Сохранили в кеш: {cache_path}")
    return dataset

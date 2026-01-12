import torch
from torch import nn
import torch.nn.functional as F


class SAFERecAutoEncoder(nn.Module):
    def __init__(self, num_items, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.num_items = num_items
        self.input_dim = num_items
        self.hidden_dim = hidden_dim

        # энкодер и декодер
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_items),
        )

        self.freq_proj = nn.Linear(1, hidden_dim // 4)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, freq=None):
        """
        если freq=None - работает как обычный автоэнкодер.
        если freq не None - добавляет частотную информацию к x.
        """
        if freq is not None:
            # приведение размеров: freq [B, I] => [B, I, 1] => freq_emb [B, I, hidden_dim//4]
            if freq.dim() == 2:
                freq = freq.unsqueeze(-1)
            freq_emb = self.freq_proj(freq).mean(dim=1)
            out = torch.cat([x, freq_emb], dim=1)
        else:
            out = x

        encoded = self.encoder(out)
        decoded = self.decoder(encoded)
        return decoded
import torch
from torch import nn
from typing import Optional, Tuple


class SAFERTransformer(nn.Module):
    PAD_TOKEN_ID = 0
    CLS_TOKEN_ID = 1

    def __init__(
        self,
        num_items: int,
        hidden_dim: int = 256,
        d_freq: int = 16,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.2,
        max_len: int = 512,
        use_freq: bool = True,
        pooling: str = "cls",
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        if num_items <= 0:
            raise ValueError("num_items must be > 0")
        if pooling not in {"cls", "mean"}:
            raise ValueError("pooling must be either 'cls' or 'mean'")
        if hidden_dim % nhead != 0:
            raise ValueError("hidden_dim must be divisible by nhead")

        self.num_items = num_items
        self.hidden_dim = hidden_dim
        self.d_freq = d_freq
        self.use_freq = use_freq
        self.pooling = pooling
        self.max_len = max_len
        self.vocab_size = num_items + 2
        self.item_offset = 2

        self.item_embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=hidden_dim,
            padding_idx=self.PAD_TOKEN_ID,
        )
        self.pos_embedding = nn.Embedding(max_len + 1, hidden_dim)

        if use_freq:
            self.freq_proj = nn.Sequential(
                nn.Linear(1, d_freq),
                nn.ReLU(),
                nn.Linear(d_freq, d_freq),
            )
            self.token_proj = nn.Sequential(
                nn.Linear(hidden_dim + d_freq, hidden_dim),
                nn.LayerNorm(hidden_dim, eps=layer_norm_eps),
                nn.ReLU(),
            )
        else:
            self.freq_proj = None
            self.token_proj = None

        ff_dim = dim_feedforward or hidden_dim * 4
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            layer_norm_eps=layer_norm_eps,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim, eps=layer_norm_eps),
        )

        self.pre_encoder_norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Linear(hidden_dim, num_items)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

        with torch.no_grad():
            self.item_embedding.weight[self.PAD_TOKEN_ID].zero_()

    def _extract_item_indices(self, x_row: torch.Tensor) -> torch.Tensor:
        return torch.nonzero(x_row > 0, as_tuple=False).flatten()

    def _build_user_sequence(
        self,
        x_row: torch.Tensor,
        freq_row: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = x_row.device
        item_idx = self._extract_item_indices(x_row)
        token_ids = item_idx + self.item_offset
        cls_token = torch.tensor([self.CLS_TOKEN_ID], device=device, dtype=torch.long)
        token_ids = torch.cat([cls_token, token_ids], dim=0)

        token_emb = self.item_embedding(token_ids)

        if self.use_freq and freq_row is not None:
            if item_idx.numel() > 0:
                freq_vals = freq_row[item_idx].unsqueeze(-1)
                cls_freq = torch.zeros(1, 1, device=device, dtype=freq_vals.dtype)
                freq_vals = torch.cat([cls_freq, freq_vals], dim=0)
            else:
                freq_vals = torch.zeros(1, 1, device=device, dtype=x_row.dtype)

            freq_emb = self.freq_proj(freq_vals)
            token_emb = self.token_proj(torch.cat([token_emb, freq_emb], dim=-1))

        return token_emb

    def _build_batch_sequences(
        self,
        x: torch.Tensor,
        freq: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_tokens = []
        lengths = []

        for i in range(x.size(0)):
            freq_row = freq[i] if freq is not None else None
            tokens = self._build_user_sequence(x[i], freq_row)

            if tokens.size(0) > self.max_len:
                tokens = torch.cat([tokens[:1], tokens[-(self.max_len - 1):]], dim=0)

            batch_tokens.append(tokens)
            lengths.append(tokens.size(0))

        padded = nn.utils.rnn.pad_sequence(
            batch_tokens,
            batch_first=True,
            padding_value=0.0,
        )
        lengths = torch.tensor(lengths, device=x.device, dtype=torch.long)
        max_t = padded.size(1)
        positions = torch.arange(max_t, device=x.device).unsqueeze(0)
        padding_mask = positions >= lengths.unsqueeze(1)
        return padded, padding_mask, lengths

    def _add_positional_encoding(self, tokens: torch.Tensor) -> torch.Tensor:
        t = tokens.size(1)
        pos_ids = torch.arange(t, device=tokens.device).unsqueeze(0)
        return tokens + self.pos_embedding(pos_ids)

    def encode(
        self,
        x: torch.Tensor,
        freq: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Преобразует вход в скрытые представления TransformerEncoder."""
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if freq is not None:
            freq = torch.nan_to_num(freq, nan=0.0, posinf=0.0, neginf=0.0)

        tokens, padding_mask, lengths = self._build_batch_sequences(x, freq)
        tokens = self._add_positional_encoding(tokens)
        tokens = self.pre_encoder_norm(tokens)
        tokens = self.dropout(tokens)

        encoded = self.encoder(tokens, src_key_padding_mask=padding_mask)
        encoded = torch.nan_to_num(encoded, nan=0.0, posinf=0.0, neginf=0.0)
        return encoded, padding_mask, lengths

    def pool(
        self,
        encoded: torch.Tensor,
        padding_mask: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Агрегирует последовательность в user embedding."""
        if self.pooling == "cls":
            user_emb = encoded[:, 0, :]
        elif self.pooling == "mean":
            valid_mask = (~padding_mask).unsqueeze(-1).to(encoded.dtype)
            summed = (encoded * valid_mask).sum(dim=1)
            denom = valid_mask.sum(dim=1).clamp(min=1.0)
            user_emb = summed / denom
        else:
            raise RuntimeError(f"Unknown pooling: {self.pooling}")

        return torch.nan_to_num(user_emb, nan=0.0, posinf=0.0, neginf=0.0)

    def decode(self, user_emb: torch.Tensor) -> torch.Tensor:
        """Декодирует user embedding в логиты по всем товарам."""
        logits = self.decoder(user_emb)
        return torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)

    def forward(
        self,
        x: torch.Tensor,
        freq: Optional[torch.Tensor] = None,
        return_user_emb: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        encoded, padding_mask, lengths = self.encode(x, freq)
        user_emb = self.pool(encoded, padding_mask, lengths)
        logits = self.decode(user_emb)

        if return_user_emb:
            return logits, user_emb
        return logits

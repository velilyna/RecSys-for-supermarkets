import torch
from torch import nn
import torch.nn.functional as F


class SAFERecAutoEncoder(nn.Module):

    def __init__(
        self,
        num_items,
        seq_len=5,
        basket_emb_dim=128,
        ae_hidden_dim=256,
        user_dim=128,
        freq_emb_dim=64,
        freq_hidden_dim=128,
        dropout=0.2,
        max_freq_clip=20,
        use_recon_loss=True,
        recon_loss_weight=0.1,
    ):
        super().__init__()

        self.num_items = num_items
        self.seq_len = seq_len
        self.basket_emb_dim = basket_emb_dim
        self.user_dim = user_dim
        self.max_freq_clip = max_freq_clip
        self.use_recon_loss = use_recon_loss
        self.recon_loss_weight = recon_loss_weight

        self.basket_encoder = nn.Sequential(
            nn.Linear(num_items, basket_emb_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(basket_emb_dim, basket_emb_dim),
            nn.Tanh(),
        )

        self.pos_emb = nn.Embedding(seq_len, basket_emb_dim)
        self.user_encoder = nn.Sequential(
            nn.Linear(seq_len * basket_emb_dim, ae_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ae_hidden_dim, user_dim),
            nn.ReLU(),
        )

        self.ae_decoder = nn.Sequential(
            nn.Linear(user_dim, ae_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ae_hidden_dim, seq_len * basket_emb_dim),
        )

        self.item_emb_user = nn.Embedding(num_items, user_dim)

        self.item_emb_freq = nn.Embedding(num_items, freq_emb_dim)
        self.freq_embedding = nn.Embedding(max_freq_clip + 1, freq_emb_dim)

        self.freq_mlp = nn.Sequential(
            nn.Linear(freq_emb_dim + seq_len, freq_hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(freq_hidden_dim, 1),
        )
        self.alpha_personal = nn.Parameter(torch.tensor(1.0))
        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)

    def encode(self, hist_baskets):
        B, L, _ = hist_baskets.shape

        # [B, L, I] -> [B, L, d]
        basket_latent = self.basket_encoder(hist_baskets)

        positions = torch.arange(L, device=hist_baskets.device)
        pos = self.pos_emb(positions).unsqueeze(0)   # [1, L, d]
        basket_latent = basket_latent + pos           # [B, L, d]

        z_flat = basket_latent.reshape(B, L * self.basket_emb_dim)
        user_vec = self.user_encoder(z_flat)

        return user_vec, z_flat

    def forward(self, hist_baskets):
        """
        hist_baskets: [B, L, num_items], float multi-hot
        returns:
            logits:     [B, num_items]
            recon_loss: scalar (0.0 если use_recon_loss=False)
        """
        B, L, I = hist_baskets.shape
        assert I == self.num_items, f"Expected {self.num_items} items, got {I}"
        assert L == self.seq_len,   f"Expected seq_len={self.seq_len}, got {L}"

        user_vec, z_flat = self.encode(hist_baskets)

        if self.use_recon_loss:
            z_reconstructed = self.ae_decoder(user_vec)
            recon_loss = F.mse_loss(z_reconstructed, z_flat.detach())
        else:
            recon_loss = torch.tensor(0.0, device=hist_baskets.device)

        logits_user = user_vec @ self.item_emb_user.weight.t()

        item_hist = hist_baskets.transpose(1, 2).float()                         # [B, I, L]
        freq_counts = item_hist.sum(dim=-1).long().clamp(0, self.max_freq_clip)  # [B, I]

        item_ids = torch.arange(self.num_items, device=hist_baskets.device)
        item_emb  = self.item_emb_freq(item_ids).unsqueeze(0).expand(B, -1, -1) # [B, I, freq_emb_dim]
        freq_emb  = self.freq_embedding(freq_counts)                             # [B, I, freq_emb_dim]
        item_plus_freq = item_emb + freq_emb                                     # [B, I, freq_emb_dim]

        freq_input  = torch.cat([item_plus_freq, item_hist], dim=-1)             # [B, I, freq_emb_dim+L]
        logits_freq = self.freq_mlp(freq_input).squeeze(-1)                      # [B, I]

        personal_freq = hist_baskets.sum(dim=1)  # [B, I]
        logits_personal = self.alpha_personal * torch.log1p(personal_freq)

        logits = logits_user + logits_freq + logits_personal
        return logits, recon_loss

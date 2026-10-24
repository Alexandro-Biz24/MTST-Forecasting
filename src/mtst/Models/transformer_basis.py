import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RelativePositionalEncoder(nn.Module):
    
    def __init__(self, dim, max_len=500):
        super().__init__()
        self.dim = dim
        self.max_len = max_len

        # Pré-calcul des fréquences
        pos = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer('pe', pe)  # pas de gradient, mais stocké sur device

    def forward(self, positions: torch.Tensor):
        
        positions = positions.long().clamp(0, self.max_len - 1)
        return self.pe[positions]

class RelativeMultiHeadAttention(nn.Module):
    """
    Implémente l'équation :
        α = Softmax( (W_QY)ᵀ(W_KY)/√D + W_posᵀP )
        Z̃ = W_VY α

    - Ajoute une composante de position relative (sinusoïdale) entre tokens.
    - Suit la formulation du papier que tu m’as montré.
    """

    def __init__(self, embed_dim, num_heads, max_len=500):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.max_len = max_len

        # Projections linéaires Q, K, V, O
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)

        # Projection positionnelle apprise (correspond à W_pos)
        self.W_pos = nn.Linear(self.head_dim, self.head_dim, bias=False)

        # Encodeur sinusoïdal des distances relatives
        self.pos_encoder = RelativePositionalEncoder(self.head_dim, max_len=max_len)

    def forward(self, x):
        B, T, E = x.shape
        H, D = self.num_heads, self.head_dim

        # 1) Linear projections
        Q = self.W_q(x).view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
        K = self.W_k(x).view(B, T, H, D).transpose(1, 2)
        V = self.W_v(x).view(B, T, H, D).transpose(1, 2)

        # 2) Classic similarity scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D)  # (B, H, T, T)

        # 3) Matrix of relative shifts between tokens
        rel_pos = torch.arange(T, device=x.device)
        rel_pos = rel_pos[None, :] - rel_pos[:, None]  # (T, T)
        rel_sign = torch.sign(rel_pos).unsqueeze(-1)   # (T, T, 1)
        rel_dist = rel_pos.abs().clamp(0, self.max_len - 1)  

        # 4) Sinusoidal encoding of relative distance |i-j|
        P = self.pos_encoder(rel_dist) * rel_sign  # (T, T, D)

        # 5) Linear projection on positions (W_posᵀ P)
        P_proj = self.W_pos(P) 

        # 6) Reduction to dimension D → scalar bias for each pair (i,j)
        pos_bias = P_proj.mean(-1)  # (T, T)

        # 7) Addition of relative bias to attention scores
        attn_scores = attn_scores + pos_bias.unsqueeze(0).unsqueeze(0) 

        # 8) Weighting and combination
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)  # (B, H, T, D)

        # 9) Head fusion and final projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, E)
        out = self.W_o(attn_output)

        return out

class CustomTransformerBlock(nn.Module):
    """Un bloc Transformer avec attention relative + feed-forward."""
    def __init__(self, embed_dim, num_heads, ff_hidden_dim=2048, dropout=0.1, max_len=500):
        super().__init__()
        self.attn = RelativeMultiHeadAttention(embed_dim, num_heads, max_len=max_len)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Bloc attention + residual
        attn_out = self.attn(x)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # Bloc feed-forward + residual
        ff_out = self.ff(x)
        x = x + self.dropout(ff_out)
        x = self.norm2(x)

        return x

class CustomTransformerEncoder(nn.Module):
    """
    Remplace directement ton ancien TransformerBlock.
    Même interface :
        - input:  (B, T, E)
        - output: (B, T, E)
    """
    def __init__(self, embed_size, heads=10, dropout=0.1, n_layers=5, ff_hidden_dim=2048, max_len=500):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.dropout = dropout
        self.n_layers = n_layers

        self.layers = nn.ModuleList([
            CustomTransformerBlock(embed_size, heads, ff_hidden_dim, dropout, max_len)
            for _ in range(n_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads=10, dropout=0.1, n_layers=5):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.dropout = dropout
        self.n_layers = n_layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
    
    def forward(self, x):
        return self.transformer(x)

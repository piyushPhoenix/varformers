import math
import torch
from torch import nn
from einops import rearrange

class LinformerAttention(nn.Module):
    def __init__(self, dim, heads, device,
                 k_dim, dropout=0.1, qkv_bias=True, attn_out_bias=True):
        super(LinformerAttention, self).__init__()
        
        assert dim % heads == 0, 'dimension must be divisible by number of heads'
        dim_heads = dim // heads
        inner_dim = dim_heads * heads
        
        self.heads = heads
        self.dim_heads = dim_heads
        self.k_dim = k_dim
        
        # Projection matrices for Q, K, V and the low-rank projection for K and V
        self.to_q = nn.Linear(dim, inner_dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, inner_dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, inner_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim_heads, k_dim , bias=False)  # Linear projection for key
        self.v_proj = nn.Linear(dim_heads, k_dim, bias=False)  # Linear projection for value
        self.to_out = nn.Linear(inner_dim, dim, bias=attn_out_bias)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        b, n, _ = x.shape
        
        # Generate Q, K, V matrices
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        
        # Reshape to separate heads
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)

        print(k.shape)
        
        # Apply low-rank projections on K and V
        k = self.k_proj(k)  # Shape: (b, h, n, k_dim)
        v = self.v_proj(v)  # Shape: (b, h, n, k_dim)
        
        # Perform scaled dot-product attention with low-rank K and V
        scale = math.sqrt(self.dim_heads)
        attn_scores = torch.einsum('bhqd,bhkd->bhqk', q, k) / scale
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Weighted sum of V with attention probabilities
        out = torch.einsum('bhqk,bhkd->bhqd', attn_probs, v)
        
        # Reshape and project out
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
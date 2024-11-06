import math
import torch
from torch import nn
from einops import rearrange

class LinformerAttention(nn.Module):
    def __init__(self, dim, heads, k_dim, seq_len, dropout=0.1, qkv_bias=True, attn_out_bias=True):
        super(LinformerAttention, self).__init__()
        
        assert dim % heads == 0, 'dimension must be divisible by number of heads'
        dim_heads = dim // heads
        inner_dim = dim_heads * heads
        
        self.heads = heads
        self.dim_heads = dim_heads
        self.k_dim = k_dim
        
        # Projection matrices for Q, K, V
        self.to_q = nn.Linear(dim, inner_dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, inner_dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, inner_dim, bias=qkv_bias)
        
        # Low-rank projection for the sequence dimension (N -> k)
        self.proj_k = nn.Linear(seq_len, k_dim, bias=False)  # Projection for keys across sequence length
        self.proj_v = nn.Linear(seq_len, k_dim, bias=False)  # Projection for values across sequence length
        
        # Final output projection
        self.to_out = nn.Linear(inner_dim, dim, bias=attn_out_bias)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        b, n, _ = x.shape
        
        # Generate Q, K, V matrices
        q = self.to_q(x)  # Shape: (b, n, inner_dim)
        k = self.to_k(x)  # Shape: (b, n, inner_dim)
        v = self.to_v(x)  # Shape: (b, n, inner_dim)
        
        # Reshape to separate heads
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)

        print(k.shape, q.shape, v.shape) 

        # Apply sequence dimension projection to reduce N -> k 
        k = rearrange(k, 'b h n d -> b h d n')  # Shape: (b, h, dim_heads, n)
        v = rearrange(v, 'b h n d -> b h d n')  # Shape: (b, h, dim_heads, n)

      
        k = self.proj_k(k)  # Shape: (b, h, dim_heads, k_dim)
        v = self.proj_v(v)  # Shape: (b, h, dim_heads, k_dim)
        
        # Rearrange back for attention calculation
        k = rearrange(k, 'b h d k -> b h k d')  # Shape: (b, h, k_dim, dim_heads)
        v = rearrange(v, 'b h d k -> b h k d')  # Shape: (b, h, k_dim, dim_heads)
        
        # Perform scaled dot-product attention with low-rank K and V
        scale = math.sqrt(self.dim_heads)
        attn_scores = torch.einsum('bhqd,bhkd->bhqk', q, k) / scale

        print(attn_scores.shape)
        
        # Apply mask if available
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Weighted sum of V with attention probabilities
        out = torch.einsum('bhqk,bhkd->bhqd', attn_probs, v)
        
        # Reshape and project out
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
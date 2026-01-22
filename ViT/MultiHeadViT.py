import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from einops import rearrange
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
  def __init__(self, embed_dim=768, num_heads=12, dropout=0.3):
    super().__init__()
    self.num_heads = num_heads
    self.head_dim = int(embed_dim/num_heads)
    self.linear1 = nn.Linear(embed_dim, 3*embed_dim)
    self.softmax = nn.Softmax(dim=-1)
    self.dropout = nn.Dropout(dropout)
    self.norm = nn.LayerNorm(embed_dim)
    self.linear2 = nn.Sequential(
        nn.Linear(embed_dim,embed_dim),
        nn.Dropout(dropout)
    )
    self.rearrange = Rearrange('b n (h d) -> b h n d', h=num_heads)

  def forward(self,x):
    # to remember original shape of input
    B, L ,D = x.shape
    residual = x
    x = self.norm(x) # (B L D)
    x= self.linear1(x) # B L 3D
    q,k,v = torch.chunk(x,3,dim=-1) # B L D
    q = self.rearrange(q) # B H L d (H * d = embed_dim)
    k = self.rearrange(k)
    v = self.rearrange(v)
    x= self.softmax((q @ k.transpose(-2,-1)) / self.head_dim**(0.5)) # B H L d @ B H d L = B H L L
    x= self.dropout(x)
    x = x @ v # B H L L @ B H L d = B H L d
    x= x.transpose(1,2).reshape(B,L,D) # B L H d -> B L D
    x = self.linear2(x)
    return residual +x


msa = MultiHeadAttention(embed_dim=768, num_heads=12)
input_ = torch.randn((8,196,768)) # INPUT shape: [batch_size, patch_num(14x14), embedding_dim]
output = msa(input_)

print("OUTPUT shape: ", output.shape)

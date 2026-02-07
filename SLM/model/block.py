from .attention import CasualSelfAttention
from .layernorm import LayerNorm
from .linearlayer import MLP


import torch.nn as nn

class Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.ln1 = LayerNorm(config.n_embed, config.bias)
        self.attn = CasualSelfAttention(config)
        self.ln2 = LayerNorm(config.n_embed, config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x+self.attn(self.ln1(x))
        x = x+self.mlp(self.ln2(x))
        return x
    
    
import torch
import torch.nn as nn
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask=None):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        scores = torch.exp(scores)
        if attn_mask is not None:
            scores = scores * attn_mask
        attn = scores / (torch.sum(scores, dim=-1, keepdim=True) + 1e-8)

        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=0, d_v=0):
        super(MultiHeadSelfAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        if d_k == 0 or d_v == 0:
            d_k = d_model // n_heads
            d_v = d_model // n_heads
        self.d_k = d_k
        self.d_v = d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, Q, K=None, V=None, length=None):
        if K is None:
            K = Q
        if V is None:
            V = Q
        batch_size = Q.size(0)

        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads,
                               self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads,
                               self.d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads,
                               self.d_v).transpose(1, 2)

        if length is not None:
            maxlen = Q.size(1)
            attn_mask = torch.arange(maxlen).to(device).expand(
                batch_size, maxlen) < length.to(device).view(-1, 1)
            attn_mask = attn_mask.unsqueeze(1).expand(batch_size, maxlen,
                                                      maxlen)
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        else:
            attn_mask = None

        context, attn = ScaledDotProductAttention(self.d_k)(q_s, k_s, v_s,
                                                            attn_mask)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.n_heads * self.d_v)
        return context
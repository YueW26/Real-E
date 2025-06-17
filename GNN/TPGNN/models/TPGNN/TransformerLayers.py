import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionwiseFeedForward(nn.Module):
    '''
    A two-feed-forward-layer module
    '''

    def __init__(self, d_in, d_hid, drop_prob=0):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        residual = x
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x


class ScaledDotProductAttention(nn.Module):
    ''' 
    Scaled Dot-Product Attention
    '''

    def __init__(self, drop_prob=0):
        super().__init__()
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, q, k, v, mask=None):
        d_emb = q.shape[-1]
        attn = torch.matmul(q / d_emb**0.5, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))

        # 确保 v 的形状
        if v.dim() == 2:  # 如果 v 是 [batch_size, n_attr]
            v = v.unsqueeze(1)  # 添加维度

        output = torch.matmul(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    ''' 
    Multi-Head Attention module 
    '''

    def __init__(self, n_head, d_attribute, d_k, d_v, drop_prob=0):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        # q, k, v
        self.w_qs = nn.Linear(d_attribute, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_attribute, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_attribute, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_attribute, bias=False)
        self.attention = ScaledDotProductAttention(drop_prob=drop_prob)
        self.dropout = nn.Dropout(drop_prob)
        self.norm = nn.LayerNorm(d_attribute, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        b_q, n_q, t_q, k_q = q.size()
        q = q.contiguous().view(-1, q.shape[2], q.shape[3])
        k = k.contiguous().view(-1, k.shape[2], k.shape[3])
        v = v.contiguous().view(-1, v.shape[2], v.shape[3])

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = q
        
        # Pass through the pre-attention projection
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        q, _ = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual
        q = self.norm(q)
        q = q.contiguous().view(b_q, n_q, t_q, k_q)
        return q

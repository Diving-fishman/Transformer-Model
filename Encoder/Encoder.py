#负责组装Encoder模型

"""参数解释:
    x: 输入 [batch_size, seq_len, d_model]
    mask: 注意力掩码
    """

import torch.nn as nn
from Multi_Head import MultiHead_Self
from add_norm import AddNorm
from FeedForward import FeedForward

#创建Encoder类
class Transformer_Encoder(nn.Module):

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(Transformer_Encoder, self).__init__()

        #自注意力子层
        self.self_attention = MultiHead_Self(d_model, num_heads, dropout)
        self.add_norm1 = AddNorm(d_model, dropout)

        #前馈神经网络子层
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.add_norm2 = AddNorm(d_model, dropout)

    def forward(self, x, mask=None):

        #自注意力部分
        attn_output, attn_weights = self.self_attention(x, mask)
        x = self.add_norm1(x, attn_output)

        #前馈网络部分
        ff_output = self.feed_forward(x)
        x = self.add_norm2(x, ff_output)

        return x, attn_weights
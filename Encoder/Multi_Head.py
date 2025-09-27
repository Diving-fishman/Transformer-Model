#用于实现多头注意力层

"""参数解释:
    d_model: 模型总维度
    num_heads: 头的数量
    dropout: dropout比率
    query, key, value: 形状都是 [batch_size, seq_len, d_model]
    mask: 可选的掩码张量
    Returns(of forward):
    输出: [batch_size, seq_len, d_model]
    注意力权重: [batch_size, num_heads, seq_len, seq_len]"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#创建多头类
class Multi_Head(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(Multi_Head, self).__init__()

        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度

        # 4个线性变换层：Q, K, V 和 最终输出
        self.W_q = nn.Linear(d_model, d_model)  # 生成Q
        self.W_k = nn.Linear(d_model, d_model)  # 生成K
        self.W_v = nn.Linear(d_model, d_model)  # 生成V
        self.W_o = nn.Linear(d_model, d_model)  # 输出投影

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, query, key, value, mask=None):

        batch_size, seq_len, d_model = query.shape

        #线性投影并分割成多头
        Q = self.W_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        #计算缩放点积注意力，每个头独立计算
        # Q: [batch_size, num_heads, seq_len, d_k]
        # K: [batch_size, num_heads, seq_len, d_k] -> 转置后: [batch_size, num_heads, d_k, seq_len]
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        #形状: [batch_size, num_heads, seq_len, seq_len]

        #计算注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        #应用注意力权重到V
        #attention_weights: [batch_size, num_heads, seq_len, seq_len]
        #V: [batch_size, num_heads, seq_len, d_k]
        output = torch.matmul(attention_weights, V)
        #输出形状: [batch_size, num_heads, seq_len, d_k]

        #合并多头
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        #现在形状: [batch_size, seq_len, d_model]

        #最终输出投影
        output = self.W_o(output)

        #返回模型参数
        return output, attention_weights


class MultiHead_Self(Multi_Head):
    def forward(self, x, mask=None):
        return super().forward(x, x, x, mask)
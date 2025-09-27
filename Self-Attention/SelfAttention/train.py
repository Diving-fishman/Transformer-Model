"""
这一文件中，我实现了一个简单的Self-Attention层，为实现完整的Encoder模型奠定基础
参数解释:
    d_model: 输入向量的维度（也是输出向量的维度）
    d_k: Query和Key的维度（默认为d_model）
    d_v: Value的维度（默认为d_model）
    x: 输入张量，形状为 [batch_size, seq_len, d_model]
    mask: 可选掩码，形状为 [batch_size, seq_len] 或 [batch_size, seq_len, seq_len]
    Returns(of forward):输出张量，形状为 [batch_size, seq_len, d_v]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#定义自注意类
class Self_Attention(nn.Module):
    def __init__(self, d_model, d_k=None, d_v=None):
        super(Self_Attention, self).__init__()

        self.d_model = d_model
        self.d_k = d_k if d_k is not None else d_model
        self.d_v = d_v if d_v is not None else d_model

        #定义线性变换矩阵：将输入映射到Q, K, V空间
        self.W_q = nn.Linear(d_model, self.d_k)  # Query权重矩阵
        self.W_k = nn.Linear(d_model, self.d_k)  # Key权重矩阵
        self.W_v = nn.Linear(d_model, self.d_v)  # Value权重矩阵

        #缩放因子，防止点积过大
        self.scale = math.sqrt(self.d_k)

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape

        # 线性变换：将输入x映射到Q, K, V空间
        Q = self.W_q(x)  # 形状为[batch_size, seq_len, d_k]
        K = self.W_k(x)  # 形状为[batch_size, seq_len, d_k]
        V = self.W_v(x)  # 形状为[batch_size, seq_len, d_v]

        #计算注意力分数 ：做Q和K的点积
        attention_scores = torch.bmm(Q, K.transpose(1, 2))

        #缩放注意力分数
        attention_scores = attention_scores / self.scale

        #在最后一个维度应用softmax得到注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)
        # attention_weights形状: [batch_size, seq_len, seq_len]

        #注意力权重与Value加权求和
        output = torch.bmm(attention_weights, V)  # [batch_size, seq_len, d_v]

        #返回输出和注意力权重
        return output, attention_weights

#-------测试部分-------#

    #测试实现的Self-Attention层
    def test(self):
        #设置参数
        batch_size = 2
        seq_len = 5
        d_model = 8
        d_k = 6
        d_v = 10

        #创建Self-Attention层实例
        self_attn = Self_Attention(d_model, d_k, d_v)

        #创建随机输入数据：模拟batch_size=2，每个序列5个词，每个词8维向量
        x = torch.randn(batch_size, seq_len, d_model)
        print(f"输入形状: {x.shape}")

        #前向传播
        output, attn_weights = self_attn(x)

        #输出随机测试结果
        print(f"输出形状: {output.shape}")
        print(f"注意力权重形状: {attn_weights.shape}")
        print(f"注意力权重矩阵（第一个样本）:\n{attn_weights[0].detach().numpy().round(3)}")

        print("--------------------------------------")

        #验证自注意力特性：每个位置的输出都是所有位置的加权平均
        print("\n验证自注意力机制:")
        print(f"第一个样本，第一个位置的输出是其他位置的加权平均:")
        for i in range(seq_len):
            weight = attn_weights[0, 0, i].item()
            print(f"位置{i}的权重: {weight:.3f}")

if __name__ == "__main__":
    model = Self_Attention(d_model=512, d_k=64, d_v=64)
    model.test()
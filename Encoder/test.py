#用于测试参数性能并可视化结果

import torch
import matplotlib.pyplot as plt
from Encoder import Transformer_Encoder

#测试性能
def implementation():
    #参数设置
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8
    d_ff = 2048

    #创建Encoder层
    encoder_layer = Transformer_Encoder(d_model, num_heads, d_ff)

    #创建随机输入
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"输入形状: {x.shape}")

    #前向传播
    output, attn_weights = encoder_layer(x)

    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attn_weights.shape}")  # [batch_size, num_heads, seq_len, seq_len]

    #验证输入输出维度一致
    assert x.shape == output.shape, "输入输出维度必须一致!"
    print("维度一致性验证通过!")

    print("----------------------------------")

    #查看多头注意力的效果
    print(f"\n多头注意力可视化:")
    print(f"第一个样本，第一个头的注意力矩阵:")
    print(attn_weights[0, 0].detach().round(decimals=3))

#可视化测试结果
def visualize():

    #创建一个小型模型
    d_model = 64
    num_heads = 4
    encoder_layer = Transformer_Encoder(d_model, num_heads, d_ff=128)

    #创建有意义的测试数据
    seq_len = 6
    x = torch.randn(1, seq_len, d_model)

    #获取注意力权重
    receive , attn_weights = encoder_layer(x)         #只关心权重

    #绘制第一个头的注意力热力图
    plt.figure(figsize=(10, 8))
    for i in range(num_heads):
        plt.subplot(2, 2, i + 1)
        attention_matrix = attn_weights[0, i].detach().numpy()
        plt.imshow(attention_matrix, cmap='hot', interpolation='nearest')
        plt.title(f'Head {i + 1}')
        plt.colorbar()

    plt.tight_layout()
    plt.show()

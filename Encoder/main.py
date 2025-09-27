#实现Transformer Encoder的主函数

import torch
import torch.nn as nn
import math
from Encoder import Transformer_Encoder
from test import implementation , visualize

class Embed_Encoder(nn.Module):

    def __init__(self, vocab_size, max_seq_len, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super(Embed_Encoder, self).__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        #词嵌入层
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        #位置编码
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        #Dropout
        self.dropout = nn.Dropout(dropout)

        #堆叠多个Encoder层，也是经典论文中的操作
        self.layers = nn.ModuleList([
            Transformer_Encoder(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, input_ids, mask=None):
        batch_size, seq_len = input_ids.shape

        #词嵌入+位置编码
        token_embeddings = self.token_embedding(input_ids)
        positions = torch.arange(0, seq_len).expand(batch_size, seq_len).to(input_ids.device)
        position_embeddings = self.position_embedding(positions)

        #组合嵌入
        x = token_embeddings * math.sqrt(self.d_model) + position_embeddings
        x = self.dropout(x)

        #通过所有Encoder层
        all_attention_weights = []
        for layer in self.layers:
            x, attention_weights = layer(x, mask)
            all_attention_weights.append(attention_weights)

        return x, all_attention_weights


def main():

    #模型参数设置
    vocab_size = 1000
    max_seq_len = 20
    d_model = 128
    num_heads = 8
    d_ff = 512
    num_layers = 3
    batch_size = 2

    #创建完整模型
    model = Embed_Encoder(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers
    )
    print(f"模型参数: d_model={d_model}, heads={num_heads}, layers={num_layers}")

    # 运行测试
    implementation()

    # 可视化
    visualize()

#运行主函数
if __name__ == "__main__":
    main()
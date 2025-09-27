#这里用于实现第3题的Vision Transformer模型，由于借用了第2题的Encoder模型故放在同一文件夹中

"""参数解释：
x: [batch_size, channels, height, width]"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
#导入已有的Encoder
from Encoder import Transformer_Encoder

#图像块嵌入层：将图像分割成块并线性投影
class PatchEmbedding(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_channels=3, d_model=512):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        #卷积层实现图像块分割和投影类似于论文中的线性投影
        self.projection = nn.Conv2d(in_channels, d_model,
                                    kernel_size=patch_size,
                                    stride=patch_size)

        #可学习的位置嵌入
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, d_model))

        #分类令牌
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        #层归一化
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        batch_size = x.shape[0]

        #投影并展平图像块 : [batch_size, d_model, num_patches_h, num_patches_w]
        x = self.projection(x)

        #展平空间维度 : [batch_size, d_model, num_patches]
        x = x.flatten(2)

        #转置为 : [batch_size, num_patches, d_model]
        x = x.transpose(1, 2)

        #添加分类令牌
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        #添加位置嵌入
        x = x + self.position_embedding

        #层归一化
        x = self.norm(x)

        return x

#Vision Transformer架构
class VisionTransformer(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 d_model=512, num_heads=8, d_ff=2048, num_layers=12,
                 num_classes=1000, pretrained_encoder=None):
        super().__init__()

        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, d_model)
        self.num_patches = self.patch_embedding.num_patches

        #使用预训练的Encoder
        self.encoder_layers = nn.ModuleList([
            Transformer_Encoder(d_model, num_heads, d_ff)
            for i in range(num_layers)
        ])

        #分类头
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

        #初始化参数
        self.apply(self._init_weights)

    #权重初始化
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x, return_attention=False):
        #图像块嵌入
        x = self.patch_embedding(x)

        #通过Encoder层
        attention_weights = []
        for encoder_layer in self.encoder_layers:
            if return_attention:
                x, attn_weights = encoder_layer(x)
                attention_weights.append(attn_weights)
            else:
                x, receive = encoder_layer(x)

        #取分类令牌对应的输出
        cls_output = x[:, 0]

        #分类
        logits = self.mlp_head(cls_output)

        if return_attention:
            return logits, attention_weights
        return logits

    #类似PCA分析，可视化图像块嵌入的滤波器
    def v_patch_embedding(self, num_filters=16):
        weights = self.patch_embedding.projection.weight.data
        weights = weights.permute(0, 2, 3, 1).cpu().numpy()

        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        for i, ax in enumerate(axes.flat):
            if i < min(num_filters, weights.shape[0]):
                # 归一化显示
                filter_img = (weights[i] - weights[i].min()) / (weights[i].max() - weights[i].min())
                ax.imshow(filter_img)
                ax.set_title(f'Filter {i}')
                ax.axis('off')
        plt.tight_layout()
        plt.show()

    #可视化位置嵌入的相似性
    def v_position_embedding(self):
        pos_emb = self.patch_embedding.position_embedding.data[0, 1:]  # 排除CLS token
        similarity = F.cosine_similarity(pos_emb.unsqueeze(1), pos_emb.unsqueeze(0), dim=-1)

        plt.figure(figsize=(10, 8))
        plt.imshow(similarity.cpu().numpy(), cmap='hot')
        plt.title('Position Embedding Similarity')
        plt.colorbar()
        plt.show()

#测试
def test_ViT():
    #参数设置
    batch_size = 2
    img_size = 224
    patch_size = 16
    in_channels = 3
    d_model = 512
    num_heads = 8
    d_ff = 2048
    num_classes = 10

    #创建随机输入图像
    x = torch.randn(batch_size, in_channels, img_size, img_size)
    print(f"输入图像形状: {x.shape}")

    #创建ViT
    vit = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_classes=num_classes
    )

    #前向传播
    logits, attention_weights = vit(x, return_attention=True)

    print(f"输出logits形状: {logits.shape}")
    print(f"注意力权重数量: {len(attention_weights)}")
    print(f"每个注意力权重形状: {attention_weights[0].shape}")

    #测试维度一致性
    expected_patches = (img_size // patch_size) ** 2 + 1  # +1 for CLS token
    print(f"预期序列长度: {expected_patches}")

    #返回模型和测试参数
    return vit, x, attention_weights

#分析注意力模式
def analyze_attention(attention_weights, patch_size=16, img_size=224):
    num_layers = len(attention_weights)
    num_heads = attention_weights[0].shape[1]

    #计算平均注意力距离
    grid_size = img_size // patch_size
    positions = torch.arange(grid_size * grid_size).view(grid_size, grid_size)

    print("=== 注意力距离分析 ===")
    for layer_num, attention_value in enumerate(attention_weights):
        #取第一个样本的注意力权重
        layer_attn = attention_value[0]  # [num_heads, seq_len, seq_len]

        #分析每个头的注意力距离
        avg_distances = []
        for head_num in range(num_heads):
            #取分类令牌对其他所有patch的注意力
            cls_attention = layer_attn[head_num, 0, 1:]  # 排除分类令牌自身

            # 计算注意力加权的位置距离
            patch_positions = positions.flatten()
            weighted_positions = cls_attention * patch_positions.float()
            avg_distance = weighted_positions.sum() / cls_attention.sum()

            avg_distances.append(avg_distance.item())

        print(f"层 {layer_num}: 平均注意力距离 = {np.mean(avg_distances):.2f}")

#执行主函数
if __name__ == "__main__":
    #测试完整实现
    vit, test_input, attn_weights = test_ViT()

    #分析注意力模式
    analyze_attention(attn_weights)

    # 可视化
    vit.v_patch_embedding()
    vit.v_position_embedding()

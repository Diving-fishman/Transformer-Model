#这一部分负责实现add-norm层，即残差连接+层归一化处理
'''参数解释:
    x: 子层的输入 [batch_size, seq_len, d_model]
    sublayer_output: 子层的输出 [batch_size, seq_len, d_model]
Returns:
    归一化后的结果 [batch_size, seq_len, d_model]'''

import torch.nn as nn
class AddNorm(nn.Module):

    #初始化
    def __init__(self, d_model, dropout=0.1):
        super(AddNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer_output):
        return self.layer_norm(x + self.dropout(sublayer_output))
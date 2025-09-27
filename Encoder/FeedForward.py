#负责实现前馈神经网络

"""参数解释:
    x: [batch_size, seq_len, d_model]
Returns:
    [batch_size, seq_len, d_model]
"""

import torch.nn as nn

class FeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):

        return self.linear2(self.dropout(self.activation(self.linear1(x))))
"""
IMUTransformerEncoder model
"""

import torch
from torch import nn
from models.layers import cnn_block


class IMU_CNNLSTM(nn.Module):

    def __init__(self, config):
        """
        config: (dict) configuration of the model
        """
        super(IMU_CNNLSTM, self).__init__()

        self.hidden = config.get("transformer_dim")
        cnn_blocks = config.get("cnn_blocks")

        self.input_proj = cnn_block(config)

        lstm_hidden = (config.get("sample_size") // (2 ** (cnn_blocks - 1))) + 1
        self.lstm = nn.LSTM(input_size=self.hidden, hidden_size=lstm_hidden, num_layers=3, batch_first=True)

        num_classes=config.get("num_classes")
        self.imu_head = nn.Sequential(
            nn.Linear(lstm_hidden+1,  lstm_hidden//4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(lstm_hidden//4,  num_classes)
        )
        self.log_softmax = nn.LogSoftmax(dim=1)

        # init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, data):
         # (seq, batch, feature)
        inp1 = data["acc"]
        inp2 = data["clin"]

        # Embed in a high dimensional space and reshape to Transformer's expected shape
        src = self.input_proj(inp1).permute(2, 0, 1)

        src = self.lstm(src)[0].permute(1, 0, 2)
        src = torch.mean(src, dim=1)
        src = torch.cat([src, inp2], dim=1)
        out = self.imu_head(src)

        # Class probability
        target = self.log_softmax(out)
        return target

def get_activation(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return nn.ReLU(inplace=True)
    if activation == "gelu":
        return nn.GELU()
    raise RuntimeError("Activation {} not supported".format(activation))
import torch.nn as nn
import torch
from models.layers import cnn_block


class IMU_CNN(nn.Module):
    def __init__(self, config, num_classes):

        super(IMU_CNN, self).__init__()

        self.input_proj = cnn_block(config)
        cnn_blocks = config.get("cnn_blocks")

        dim = (config.get("sample_size") // (2 ** (cnn_blocks - 1))) + 1
        print(f"Dim: {dim}")
        self.imu_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim,  dim//2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim//2,  num_classes)
        )
        self.log_softmax = nn.LogSoftmax(dim=1)

        # init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, data):
        inp1 = data["acc"]
        inp2 = data["clin"]
        x = self.input_proj(inp1)
        x = torch.mean(x, dim=1)
        x = torch.cat([x, inp2], dim=1)
        x = self.imu_head(x)
        x = self.log_softmax(x)
        return x
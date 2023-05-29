import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from models.layers import cnn_block


class IMU_CNNTransformers(nn.Module):

    def __init__(self, config, num_classes):
        """
        config: (dict) configuration of the model
        """
        super(IMU_CNNTransformers).__init__()

        self.transformer_dim = config.get("transformer_dim")
        self.clinical_data_dim = config.get("clinical_data_dim")

        self.input_proj = cnn_block(config)
        cnn_blocks = config.get("cnn_blocks")

        self.dim = (config.get("sample_size") // (2 ** (cnn_blocks - 1))) + 1
        self.transformer_dim = 512

        self.window_size = self.dim
        self.encode_position = config.get("encode_position")
        encoder_layer = TransformerEncoderLayer(d_model=self.transformer_dim,
                                                nhead=config.get("nhead"),
                                                dim_feedforward=config.get("dim_feedforward"),
                                                dropout=config.get("transformer_dropout"),
                                                activation=config.get("transformer_activation"))

        self.transformer_encoder = TransformerEncoder(encoder_layer,
                                                      num_layers=config.get("num_encoder_layers"),
                                                      norm=nn.LayerNorm(self.transformer_dim))
        self.cls_token = nn.Parameter(torch.zeros((1, self.transformer_dim)), requires_grad=True)
        self.clin_embbeding = nn.Sequential(
                    nn.Linear(self.clinical_data_dim, self.transformer_dim),
        )

        if self.encode_position:
            self.position_embed = nn.Parameter(torch.randn(self.window_size + 1, 1, self.transformer_dim))

        self.imu_head = nn.Sequential(
            nn.LayerNorm(self.transformer_dim),
            nn.Linear(self.transformer_dim,  self.transformer_dim//4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.transformer_dim//4,  num_classes)
        )
        self.add_info = nn.Sequential(
            nn.Linear(23, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

        self.late_fusion = nn.Sequential(
            nn.Linear(4, 32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, num_classes)
        )
        self.log_softmax = nn.LogSoftmax(dim=1)

        # init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, data):
        # Shape N x S x C with S = sequence length, N = batch size, C = channels
        inp1 = data["acc"]
        inp2 = data["clin"]

        # Embed in a high dimensional space and reshape to Transformer's expected shape
        src = self.input_proj(inp1).permute(2, 0, 1)
        clin = self.clin_embbeding(inp2).unsqueeze(0)
        src = torch.cat([src, clin])

        # Prepend class token
        cls_token = self.cls_token.unsqueeze(1).repeat(1, src.shape[1], 1)
        src = torch.cat([cls_token, src])

        # Add the position embedding
        if self.encode_position:
            src += self.position_embed

        # Transformer Encoder pass
        target = self.transformer_encoder(src)[0]
        out = self.imu_head(target)

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
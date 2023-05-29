import torch.nn as nn

def cnn_block(config):
    cnn_blocks = config.get("cnn_blocks")

    input_proj = nn.Sequential()
    dim = 16
    for i in range(1, cnn_blocks):
        input_dim = config.get("input_dim") if i == 1 else dim
        input_proj.add_module(f"conv_1_{i}", nn.Conv1d(input_dim, dim, (1,)))
        input_proj.add_module(f"conv_2_{i}", nn.Conv1d(dim, dim * 2, (1,)))
        input_proj.add_module(f"maxpool_{i}", nn.MaxPool1d(2))
        input_proj.add_module(f"gelu_{i}", nn.GELU())
        dim = dim * 2

    return input_proj
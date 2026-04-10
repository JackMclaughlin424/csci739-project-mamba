import torch
import torch.nn as nn
import torch.nn.functional as F

class MambaBLock(nn.Module):
    # Sources:
    # - https://arxiv.org/abs/2312.00752
    # - https://www.ibm.com/think/topics/mamba-model

    # Mamba Flow:
    # 1. Linear projections on input -> x and res
    # 2. Depthwise convolution on x
    # 3. Activation
    # 4. SSM
    # 5. Multiply SSM output with activated res
    # 6. Linear projection on output
    # 7. Residual connection from input

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Linear projection on input
        self.input_proj = nn.Linear(config.d_input, config.d_model, bias=config.bias)
        self.res_proj = nn.Linear(config.d_input, config.d_model, bias=config.bias)

        # Depthwise convolution on x
        self.conv1d = nn.Conv1d(
            in_channels=config.d_model,
            out_channels=config.d_model,
            kernel_size=config.kernel_size,
            padding=config.kernel_size - 1,
            bias=config.conv_bias,
            groups=config.d_model # Depthwise convolution
        )

    def SSM(self, x):
        # TODO: Implement SSM
        return x

    def forward(self, x_in):

        # Linear projection on input
        x = self.input_proj(x_in)
        res = self.res_proj(x_in)

        # Depthwise convolution on x
        x = self.conv1d(x)

        # Activation
        x = F.silu(x)

        # SSM
        y = self.ssm(x)

        # Multiply SSM output with activated res
        y = y * F.silu(res)

        # Linear projection on output
        out = self.output_proj(y)

        # Residual connection from input
        out = out + x_in

        return out
        



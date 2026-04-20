import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from .fused_scan import fused_ssm

class MambaBlock(nn.Module):
    # Sources:
    # - https://arxiv.org/abs/2312.00752
    # - https://www.ibm.com/think/topics/mamba-model
    # - https://arxiv.org/pdf/2111.00396
    # - https://arxiv.org/pdf/2008.07669

    # Mamba Flow:
    # 1. Linear projections on input -> x and res
    # 2. Depthwise convolution on x
    # 3. Activation
    # 4. SSM
    # 5. Multiply SSM output with activated res
    # 6. Linear projection on output

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Linear projection on input
        self.input_proj = nn.Linear(config.d_input, config.d_model, bias=config.bias)
        self.res_proj = nn.Linear(config.d_input, config.d_model, bias=config.bias)

        # Linear projection on output
        self.output_proj = nn.Linear(config.d_model, config.d_input, bias=config.bias)

        # Depthwise convolution on x
        self.conv1d = nn.Conv1d(
            in_channels=config.d_model,
            out_channels=config.d_model,
            kernel_size=config.kernel_size,
            padding=config.kernel_size - 1,
            bias=config.conv_bias,
            groups=config.d_model # Depthwise convolution
        )

        # Input independent SSM parameters

        # Matrix A: State transition matrix
        # Initialize A with HiPPO matrix for long range dependencies
        # Simplified to diagonal eigenvalues
        # Each state dimension has a separate decay rate
        A = repeat(torch.arange(1, config.d_state + 1), 'n -> d n', d=config.d_model)
        # Logarithmic representation of A for stability and gradient flow
        self.A_log = nn.Parameter(torch.log(A))

        # Matrix D: Feedthrough matrix
        self.D = nn.Parameter(torch.ones(config.d_model))

        # Input dependent SSM parameters

        # Matrix B: Input 
        self.x_B_proj = nn.Linear(config.d_model, config.d_state, bias=False)

        # Matrix C: Output
        self.x_C_proj = nn.Linear(config.d_model, config.d_state, bias=False)

        # Projects x to dt_rank
        self.x_dt_proj = nn.Linear(config.d_model, config.dt_rank, bias=False)
        # Projects dt to d_model
        self.dt_proj = nn.Linear(config.dt_rank, config.d_model, bias=True)




    def ssm(self, x):
        # x: (B, L, D)
        A = -torch.exp(self.A_log.float())                        # (D, N)
        B_proj = self.x_B_proj(x)                                 # (B, L, N)
        C_proj = self.x_C_proj(x)                                 # (B, L, N)
        delta  = F.softplus(self.dt_proj(self.x_dt_proj(x)))      # (B, L, D)

        return fused_ssm(delta, A, B_proj, x, C_proj, self.D.float())

    def forward(self, x_in):

        # Linear projection on input
        x = self.input_proj(x_in)
        res = self.res_proj(x_in)

        L = x.shape[1]

        # Depthwise convolution
        # Reshape to (B D L) for convolution, then back to (B L D)
        x = rearrange(x, 'b l d -> b d l')
        x = self.conv1d(x)[:, :, :L]  # trim causal padding to original length
        x = rearrange(x, 'b d l -> b l d')

        # Activation
        x = F.silu(x)

        # SSM
        y = self.ssm(x)

        # Multiply SSM output with activated res
        y = y * F.silu(res)

        # Linear projection on output
        out = self.output_proj(y)

        return out
        
class RMSNorm(nn.Module):
    def __init__(self, d_input, eps=1e-5):
        super().__init__()
        self.d_model = d_input
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_input))

    def forward(self, x):
        y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return y

class ResidualBlock(nn.Module):
    # Wraps MambaBlock with residual connection from input and RMSNorm
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mamba_block = MambaBlock(config)
        self.norm = RMSNorm(config.d_input)

    def forward(self, x):
        output = self.mamba_block(self.norm(x)) + x
        return output

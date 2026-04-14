"""
Author: Jack McLaughlin

Adapted and Simplified from models/mixer_seq_simple.py in the Mamba github: https://github.com/state-spaces/mamba/tree/main?tab=readme-ov-file

Implemented with help from Sonnet 4.6.
"""

import math
import json
import os
from collections import namedtuple
from dataclasses import dataclass
from functools import partial

import torch
import torch.nn as nn

from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf

from mamba_block import RMSNorm, ResidualBlock


@dataclass
class MambaLMConfig:
    d_input: int = 768         # residual stream / embedding dimension
    d_model: int = 1536        # expanded inner dimension inside MambaBlock (typically 2x d_input)
    n_layer: int = 24
    vocab_size: int = 50277
    kernel_size: int = 4
    bias: bool = False
    conv_bias: bool = True
    pad_vocab_size_multiple: int = 8
    tie_embeddings: bool = True


def _init_weights(module, n_layer, initializer_range=0.02):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)
    for name, p in module.named_parameters():
        if name in ["out_proj.weight", "fc2.weight"]:
            # GPT-2 scheme: scale residual branch weights by 1/sqrt(n_layer)
            nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            with torch.no_grad():
                p /= math.sqrt(n_layer)


class MixerModel(nn.Module):
    def __init__(self, config: MambaLMConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.d_input)
        self.layers = nn.ModuleList([ResidualBlock(config) for _ in range(config.n_layer)])
        self.norm_f = RMSNorm(config.d_input)
        self.apply(partial(_init_weights, n_layer=config.n_layer))

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.norm_f(x)


class MambaLMHeadModel(nn.Module, GenerationMixin):
    def __init__(self, config: MambaLMConfig) -> None:
        super().__init__()
        self.config = config
        if config.vocab_size % config.pad_vocab_size_multiple != 0:
            config.vocab_size += config.pad_vocab_size_multiple - (config.vocab_size % config.pad_vocab_size_multiple)
        self.backbone = MixerModel(config)
        self.lm_head = nn.Linear(config.d_input, config.vocab_size, bias=False)
        self.apply(partial(_init_weights, n_layer=config.n_layer))
        self.tie_weights()

    def tie_weights(self):
        if self.config.tie_embeddings:
            self.lm_head.weight = self.backbone.embedding.weight

    def forward(self, input_ids, position_ids=None, inference_params=None, num_last_tokens=0):
        hidden_states = self.backbone(input_ids)
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        lm_logits = self.lm_head(hidden_states)
        CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
        return CausalLMOutput(logits=lm_logits)

    @classmethod
    def from_pretrained(cls, pretrained_model_name, device=None, dtype=None, **kwargs):
        config_data = load_config_hf(pretrained_model_name)
        config = MambaLMConfig(**config_data)
        model = cls(config, **kwargs)
        model.load_state_dict(load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype))
        return model

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_directory, 'pytorch_model.bin'))
        with open(os.path.join(save_directory, 'config.json'), 'w') as f:
            json.dump(self.config.__dict__, f, indent=4)

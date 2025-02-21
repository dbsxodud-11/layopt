# Utilities for diffusion.
from typing import Optional, List, Union

import gin
import numpy as np
import torch
from torch import nn

# GIN-required Imports.
from diffusion.denoiser_network import ResidualMLPDenoiser, ResidualGNNDenoiser, ResidualMLPDenoiserDiverse, ResidualGNNDenoiserDiverse
from diffusion.elucidated_diffusion import ElucidatedDiffusion, GraphElucidatedDiffusion
from diffusion.norm import normalizer_factory


def construct_diffusion_model(
    inputs: torch.Tensor,
    model_type: str,
    normalizer_type: str,
    cond_dim: Optional[int] = None,
    add_cond_dim: Optional[int] = None,
) -> ElucidatedDiffusion:
    
    if model_type == "mlp":
        inputs = inputs.flatten(1)
        event_dim = inputs.shape[1]
        if add_cond_dim is not None:
            model = ResidualMLPDenoiserDiverse(d_in=event_dim, cond_dim=cond_dim)
        else:
            model = ResidualMLPDenoiser(d_in=event_dim, cond_dim=cond_dim)
    
        normalizer = normalizer_factory(normalizer_type, inputs)
        
        return ElucidatedDiffusion(
        net=model,
        cond_type="value",
        normalizer=normalizer,
        event_shape=[event_dim],
    )
        
    elif model_type == "gnn":
        inputs = inputs.reshape(-1, inputs.shape[-1])
        event_dim = inputs.shape[-1]
        if add_cond_dim is not None:
            model = ResidualGNNDenoiserDiverse(d_in=event_dim, cond_dim=cond_dim)
        else:
            model = ResidualGNNDenoiser(d_in=event_dim, cond_dim=cond_dim)

        normalizer = normalizer_factory(normalizer_type, inputs)

        return GraphElucidatedDiffusion(
            net=model,
            cond_type="value",
            normalizer=normalizer,
            event_shape=[event_dim],
        )
        
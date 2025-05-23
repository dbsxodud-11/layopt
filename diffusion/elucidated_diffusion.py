"""
Main diffusion code.
Code was adapted from https://github.com/lucidrains/denoising-diffusion-pytorch
"""
import math
import pathlib
from multiprocessing import cpu_count
from typing import Optional, Sequence, Tuple

import gin
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from einops import rearrange, reduce, repeat
from ema_pytorch import EMA
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchdiffeq import odeint
from tqdm import tqdm

from data.dataset import WindFarmLayoutDataset, GraphWindFarmLayoutDataset
from diffusion.norm import BaseNormalizer

import torch_geometric
from torch_geometric.data import DataLoader as GraphDataLoader


# helpers
def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def cycle(dl):
    while True:
        for data in dl:
            yield data


# tensor helpers
def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


# main class
@gin.configurable
class ElucidatedDiffusion(nn.Module):
    def __init__(
            self,
            net,
            cond_type,
            normalizer: BaseNormalizer,
            event_shape: Sequence[int],  # shape of the input and output
            num_sample_steps: int = 128,  # number of sampling steps
            sigma_min: float = 0.002,  # min noise level
            sigma_max: float = 80,  # max noise level
            sigma_data: float = 1.0,  # standard deviation of data distribution
            rho: float = 7,  # controls the sampling schedule
            P_mean: float = -1.2,  # mean of log-normal distribution from which noise is drawn for training
            P_std: float = 1.2,  # standard deviation of log-normal distribution from which noise is drawn for training
            S_churn: float = 80,  # parameters for stochastic sampling - depends on dataset, Table 5 in paper
            S_tmin: float = 0.05,
            S_tmax: float = 50,
            S_noise: float = 1.003,
    ):
        super().__init__()
        assert net.random_or_learned_sinusoidal_cond
        self.net = net
        self.cond_type = cond_type
        self.normalizer = normalizer

        # input dimensions
        self.event_shape = event_shape

        # parameters
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.P_mean = P_mean
        self.P_std = P_std
        self.num_sample_steps = num_sample_steps  # otherwise known as N in the paper
        self.S_churn = S_churn
        self.S_tmin = S_tmin
        self.S_tmax = S_tmax
        self.S_noise = S_noise
        
        # class embeddings (if necessary)
        if self.cond_type in ["const", "joint", "composition"]:
            self.classes_emb = nn.Embedding(2, 10)
            self.null_classes_emb = nn.Parameter(torch.randn(10))

    @property
    def device(self):
        return next(self.net.parameters()).device

    # derived preconditioning params - Table 1
    def c_skip(self, sigma):
        return (self.sigma_data ** 2) / (sigma ** 2 + self.sigma_data ** 2)

    def c_out(self, sigma):
        return sigma * self.sigma_data * (self.sigma_data ** 2 + sigma ** 2) ** -0.5

    def c_in(self, sigma):
        return 1 * (sigma ** 2 + self.sigma_data ** 2) ** -0.5

    def c_noise(self, sigma):
        return log(sigma) * 0.25

    # preconditioned network output, equation (7) in the paper
    def preconditioned_network_forward(self, noised_inputs, sigma, clamp=False, cond=None, add_cond=None):
        batch, device = noised_inputs.shape[0], noised_inputs.device

        if isinstance(sigma, float):
            sigma = torch.full((batch,), sigma, device=device)

        padded_sigma = sigma.view(batch, *([1] * len(self.event_shape)))

        if add_cond is not None:
            net_out = self.net(
                self.c_in(padded_sigma) * noised_inputs,
                self.c_noise(sigma),
                cond=cond,
                add_cond=add_cond,
            )
        else:
            net_out = self.net(
                self.c_in(padded_sigma) * noised_inputs,
                self.c_noise(sigma),
                cond=cond,
            )

        out = self.c_skip(padded_sigma) * noised_inputs + self.c_out(padded_sigma) * net_out

        if clamp:
            out = out.clamp(-1., 1.)

        return out

    # sample schedule, equation (5) in the paper
    def sample_schedule(self, num_sample_steps=None):
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)

        N = num_sample_steps
        inv_rho = 1 / self.rho

        steps = torch.arange(num_sample_steps, device=self.device, dtype=torch.float32)
        sigmas = (self.sigma_max ** inv_rho + steps / (N - 1) * (
                self.sigma_min ** inv_rho - self.sigma_max ** inv_rho)) ** self.rho

        sigmas = F.pad(sigmas, (0, 1), value=0.)  # last step is sigma value of 0.
        return sigmas
    
    def prob_mask_like(self, shape, prob, device):
        if prob == 1:
            return torch.ones(shape, device = device, dtype = torch.bool)
        elif prob == 0:
            return torch.zeros(shape, device = device, dtype = torch.bool)
        else:
            return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

    def process_cond(self, cond=None, cond_drop_prob=0.0):
        if self.cond_type == "value":
            uncond = torch.zeros_like(cond, device=self.device)
        elif self.cond_type == "const":
            cond = self.classes_emb(cond.long().flatten())
            uncond = repeat(self.null_classes_emb, 'd -> b d', b=cond.shape[0])
            
            if cond_drop_prob > 0.0:
                keep_mask = self.prob_mask_like(cond.shape, 1 - cond_drop_prob, device=self.device)
                cond = torch.where(keep_mask, cond, uncond)
            
        elif self.cond_type == "joint" or self.cond_type == "composition":
            cond1 = cond[:, 0].unsqueeze(-1)
            uncond1 = torch.zeros_like(cond1, device=self.device)
            
            cond2 = cond[:, 1].unsqueeze(-1)
            cond2_emb = self.classes_emb(cond2.long().flatten())
            uncond2_emb = repeat(self.null_classes_emb, 'd -> b d', b=cond2.shape[0])
            
            cond = torch.cat([cond1, cond2_emb], dim=-1)
            uncond = torch.cat([uncond1, uncond2_emb], dim=-1)
        else:
            raise NotImplementedError
            
        if cond_drop_prob > 0.0:
            keep_mask = self.prob_mask_like(cond.shape, 1 - cond_drop_prob, device=self.device)
            cond = torch.where(keep_mask, cond, uncond)
                
        return uncond, cond

    @torch.no_grad()
    def sample(
            self,
            x,
            batch_size: int = 16,
            num_sample_steps: Optional[int] = None,
            clamp: bool = True,
            cond=None,
            add_cond=None,
            temperature: float = 1.2,
            disable_tqdm: bool = False,
    ):
        if cond is not None:
            uncond, cond = self.process_cond(cond)
        
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)
        shape = (batch_size, *self.event_shape)

        # get the schedule, which is returned as (sigma, gamma) tuple, and pair up with the next sigma and gamma
        sigmas = self.sample_schedule(num_sample_steps)
        gammas = torch.where(
            (sigmas >= self.S_tmin) & (sigmas <= self.S_tmax),
            min(self.S_churn / num_sample_steps, math.sqrt(2) - 1),
            0.
        )

        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[:-1]))

        # inputs are noise at the beginning
        # init_sigma = sigmas[0]
        # inputs = init_sigma * torch.randn(shape, device=self.device)
        
        # inputs are noise at the beginning
        init_sigma = sigmas[0]
        inputs = init_sigma * torch.randn(shape, device=self.device)

        # gradually denoise
        for sigma, sigma_next, gamma in tqdm(sigmas_and_gammas, desc='sampling time step', mininterval=1,
                                             disable=disable_tqdm):
            sigma, sigma_next, gamma = map(lambda t: t.item(), (sigma, sigma_next, gamma))

            eps = self.S_noise * torch.randn(shape, device=self.device)  # stochastic sampling

            sigma_hat = sigma + gamma * sigma
            inputs_hat = inputs + math.sqrt(sigma_hat ** 2 - sigma ** 2) * eps

            if cond is not None:
                if self.cond_type == "composition":
                    uncond_denoised_over_sigma = self.score_fn(inputs_hat, sigma_hat, clamp=clamp, cond=uncond)
                    
                    cond1 = cond.clone()
                    cond1[:, 1:] = repeat(self.null_classes_emb, 'd -> b d', b=cond1.shape[0])
                    cond1_denoised_over_sigma = self.score_fn(inputs_hat, sigma_hat, clamp=clamp, cond=cond1)
                    
                    cond2 = cond.clone()
                    cond2[:, :1] = 0.
                    cond2_denoised_over_sigma = self.score_fn(inputs_hat, sigma_hat, clamp=clamp, cond=cond2)
                    
                    denoised_over_sigma = uncond_denoised_over_sigma
                    denoised_over_sigma += temperature * (cond1_denoised_over_sigma - uncond_denoised_over_sigma)
                    denoised_over_sigma += temperature * (cond2_denoised_over_sigma - uncond_denoised_over_sigma)
                else:
                    uncond_denoised_over_sigma = self.score_fn(inputs_hat, sigma_hat, clamp=clamp, cond=uncond, add_cond=add_cond)
                    cond_denoised_over_sigma = self.score_fn(inputs_hat, sigma_hat, clamp=clamp, cond=cond, add_cond=add_cond)
                    denoised_over_sigma = uncond_denoised_over_sigma + temperature * (cond_denoised_over_sigma - uncond_denoised_over_sigma)
            else:
                denoised_over_sigma = self.score_fn(inputs_hat, sigma_hat, clamp=clamp, cond=cond)
            inputs_next = inputs_hat + (sigma_next - sigma_hat) * denoised_over_sigma

            # second order correction, if not the last timestep
            if sigma_next != 0:
                denoised_prime_over_sigma = self.score_fn(inputs_next, sigma_next, clamp=clamp, cond=cond, add_cond=add_cond)
                inputs_next = inputs_hat + 0.5 * (sigma_next - sigma_hat) * (
                        denoised_over_sigma + denoised_prime_over_sigma)

            inputs = inputs_next

        if clamp:
            inputs = inputs.clamp(-1., 1.)
        return self.normalizer.unnormalize(inputs)

    # This is known as 'denoised_over_sigma' in the lucidrains repo.
    def score_fn(
            self,
            x,
            sigma,
            clamp: bool = False,
            cond=None,
            add_cond=None,
    ):
        denoised = self.preconditioned_network_forward(x, sigma, clamp=clamp, cond=cond, add_cond=add_cond)
        denoised_over_sigma = (x - denoised) / sigma

        return denoised_over_sigma

    # Adapted from https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/sampling.py
    @torch.no_grad()
    def log_likelihood(self, x, atol=1e-4, rtol=1e-4, clamp=False, normalize=True, cond=None):
        # Input to the ODE solver must be in normalized space.
        if normalize:
            x = self.normalizer.normalize(x)
        v = torch.randint_like(x, 2) * 2 - 1
        s_in = x.new_ones([x.shape[0]])
        fevals = 0

        def ode_fn(sigma, x):
            nonlocal fevals
            with torch.enable_grad():
                x = x[0].detach().requires_grad_()
                sigma = sigma * s_in
                padded_sigma = sigma.view(x.shape[0], *([1] * len(self.event_shape)))
                denoised = self.preconditioned_network_forward(x, sigma, clamp=clamp, cond=cond)
                denoised_over_sigma = (x - denoised) / padded_sigma
                fevals += 1
                grad = torch.autograd.grad((denoised_over_sigma * v).sum(), x)[0]
                d_ll = (v * grad).flatten(1).sum(1)
            return denoised_over_sigma.detach(), d_ll

        x_min = x, x.new_zeros([x.shape[0]])
        t = x.new_tensor([self.sigma_min, self.sigma_max])
        sol = odeint(ode_fn, x_min, t, atol=atol, rtol=rtol, method='dopri5')
        latent, delta_ll = sol[0][-1], sol[1][-1]
        ll_prior = torch.distributions.Normal(0, self.sigma_max).log_prob(latent).flatten(1).sum(1)

        return ll_prior + delta_ll, {'fevals': fevals}

    # training
    def loss_weight(self, sigma):
        return (sigma ** 2 + self.sigma_data ** 2) * (sigma * self.sigma_data) ** -2

    def noise_distribution(self, batch_size):
        return (self.P_mean + self.P_std * torch.randn((batch_size,), device=self.device)).exp()

    def forward(self, inputs, cond=None, cond_drop_prob=0.0, add_cond=None):
        inputs = self.normalizer.normalize(inputs)

        batch_size, *event_shape = inputs.shape
        assert event_shape == self.event_shape, f'mismatch of event shape, ' \
                                                f'expected {self.event_shape}, got {event_shape}'

        sigmas = self.noise_distribution(batch_size)
        padded_sigmas = sigmas.view(batch_size, *([1] * len(self.event_shape)))

        noise = torch.randn_like(inputs)
        noised_inputs = inputs + padded_sigmas * noise  # alphas are 1. in the paper
        
        if cond is not None:
            _, cond = self.process_cond(cond, cond_drop_prob)

        denoised = self.preconditioned_network_forward(noised_inputs, sigmas, cond=cond, add_cond=add_cond)
        losses = F.mse_loss(denoised, inputs, reduction='none')
        losses = reduce(losses, 'b ... -> b', 'mean')
        losses = losses * self.loss_weight(sigmas)
        return losses.mean()
    
    
class GraphElucidatedDiffusion(nn.Module):
    def __init__(
            self,
            net,
            cond_type,
            normalizer: BaseNormalizer,
            event_shape: Sequence[int],  # shape of the input and output
            num_sample_steps: int = 128,  # number of sampling steps
            sigma_min: float = 0.002,  # min noise level
            sigma_max: float = 80,  # max noise level
            sigma_data: float = 1.0,  # standard deviation of data distribution
            rho: float = 7,  # controls the sampling schedule
            P_mean: float = -1.2,  # mean of log-normal distribution from which noise is drawn for training
            P_std: float = 1.2,  # standard deviation of log-normal distribution from which noise is drawn for training
            S_churn: float = 80,  # parameters for stochastic sampling - depends on dataset, Table 5 in paper
            S_tmin: float = 0.05,
            S_tmax: float = 50,
            S_noise: float = 1.003,
    ):
        super().__init__()
        assert net.random_or_learned_sinusoidal_cond
        self.net = net
        self.cond_type = cond_type
        self.normalizer = normalizer

        # input dimensions
        self.event_shape = event_shape

        # parameters
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.P_mean = P_mean
        self.P_std = P_std
        self.num_sample_steps = num_sample_steps
        self.S_churn = S_churn
        self.S_tmin = S_tmin
        self.S_tmax = S_tmax
        self.S_noise = S_noise
        
    @property
    def device(self):
        return next(self.net.parameters()).device
    
    # derived preconditioning params - Table 1
    def c_skip(self, sigma):
        return (self.sigma_data ** 2) / (sigma ** 2 + self.sigma_data ** 2)

    def c_out(self, sigma):
        return sigma * self.sigma_data * (self.sigma_data ** 2 + sigma ** 2) ** -0.5

    def c_in(self, sigma):
        return 1 * (sigma ** 2 + self.sigma_data ** 2) ** -0.5

    def c_noise(self, sigma):
        return log(sigma) * 0.25
    
    # preconditioned network output, equation (7) in the paper
    def preconditioned_network_forward(self, noised_inputs, sigma, adj, ef, clamp=False, cond=None, add_cond=None):
        batch, device = noised_inputs.shape[0], noised_inputs.device

        if isinstance(sigma, float):
            sigma = torch.full((batch,), sigma, device=device)

        padded_sigma = sigma.view(batch, *([1] * len(self.event_shape)))

        if add_cond is not None:
            net_out = self.net(
                self.c_in(padded_sigma) * noised_inputs,
                adj,
                ef,
                self.c_noise(sigma),
                cond=cond,
                add_cond=add_cond,
            )
        else:
            net_out = self.net(
                self.c_in(padded_sigma) * noised_inputs,
                adj,
                ef,
                self.c_noise(sigma),
                cond=cond,
            )

        out = self.c_skip(padded_sigma) * noised_inputs + self.c_out(padded_sigma) * net_out

        if clamp:
            out = out.clamp(-1., 1.)

        return out
    
    # sample schedule, equation (5) in the paper
    def sample_schedule(self, num_sample_steps=None):
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)

        N = num_sample_steps
        inv_rho = 1 / self.rho

        steps = torch.arange(num_sample_steps, device=self.device, dtype=torch.float32)
        sigmas = (self.sigma_max ** inv_rho + steps / (N - 1) * (
                self.sigma_min ** inv_rho - self.sigma_max ** inv_rho)) ** self.rho

        sigmas = F.pad(sigmas, (0, 1), value=0.)  # last step is sigma value of 0.
        return sigmas
    
    def prob_mask_like(self, shape, prob, device):
        if prob == 1:
            return torch.ones(shape, device = device, dtype = torch.bool)
        elif prob == 0:
            return torch.zeros(shape, device = device, dtype = torch.bool)
        else:
            return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob
    
    def process_cond(self, cond=None, cond_drop_prob=0.0):
        if self.cond_type == "value":
            uncond = torch.zeros_like(cond, device=self.device)
        elif self.cond_type == "const":
            cond = self.classes_emb(cond.long().flatten())
            uncond = repeat(self.null_classes_emb, 'd -> b d', b=cond.shape[0])
            
            if cond_drop_prob > 0.0:
                keep_mask = self.prob_mask_like(cond.shape, 1 - cond_drop_prob, device=self.device)
                cond = torch.where(keep_mask, cond, uncond)
            
        elif self.cond_type == "joint" or self.cond_type == "composition":
            cond1 = cond[:, 0].unsqueeze(-1)
            uncond1 = torch.zeros_like(cond1, device=self.device)
            
            cond2 = cond[:, 1].unsqueeze(-1)
            cond2_emb = self.classes_emb(cond2.long().flatten())
            uncond2_emb = repeat(self.null_classes_emb, 'd -> b d', b=cond2.shape[0])
            
            cond = torch.cat([cond1, cond2_emb], dim=-1)
            uncond = torch.cat([uncond1, uncond2_emb], dim=-1)
        else:
            raise NotImplementedError
            
        if cond_drop_prob > 0.0:
            keep_mask = self.prob_mask_like(cond.shape, 1 - cond_drop_prob, device=self.device)
            cond = torch.where(keep_mask, cond, uncond)
                
        return uncond, cond
    
    @torch.no_grad()
    def sample(
            self,
            x,
            adj,
            ef,
            num_turbins: int = 10,
            batch_size: int = 16,
            num_sample_steps: Optional[int] = None,
            clamp: bool = True,
            cond=None,
            temperature: float = 1.2,
            disable_tqdm: bool = False,
            add_cond=None,
    ):
        x = x.to(device=self.device)
        x = self.normalizer.normalize(x)
        
        if cond is not None:
            uncond, cond = self.process_cond(cond)
            # uncond = uncond.repeat_interleave(num_turbins, dim=0)
            # cond = cond.repeat_interleave(num_turbins, dim=0)
        
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)
        shape = (batch_size * num_turbins, *self.event_shape)

        # get the schedule, which is returned as (sigma, gamma) tuple, and pair up with the next sigma and gamma
        sigmas = self.sample_schedule(num_sample_steps)
        gammas = torch.where(
            (sigmas >= self.S_tmin) & (sigmas <= self.S_tmax),
            min(self.S_churn / num_sample_steps, math.sqrt(2) - 1),
            0.
        )

        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[:-1]))

        # inputs are noise at the beginning
        init_sigma = sigmas[0]
        inputs = init_sigma * torch.randn(shape, device=self.device)

        # gradually denoise
        for sigma, sigma_next, gamma in tqdm(sigmas_and_gammas, desc='sampling time step', mininterval=1,
                                             disable=disable_tqdm):
            sigma, sigma_next, gamma = map(lambda t: t.item(), (sigma, sigma_next, gamma))

            eps = self.S_noise * torch.randn(shape, device=self.device)  # stochastic sampling

            sigma_hat = sigma + gamma * sigma
            inputs_hat = inputs + math.sqrt(sigma_hat ** 2 - sigma ** 2) * eps

            if cond is not None:
                # if self.cond_type == "composition":
                #     uncond_denoised_over_sigma = self.score_fn(inputs_hat, sigma_hat, clamp=clamp, cond=uncond)
                    
                #     cond1 = cond.clone()
                #     cond1[:, 1:] = repeat(self.null_classes_emb, 'd -> b d', b=cond1.shape[0])
                #     cond1_denoised_over_sigma = self.score_fn(inputs_hat, sigma_hat, clamp=clamp, cond=cond1)
                    
                #     cond2 = cond.clone()
                #     cond2[:, :1] = 0.
                #     cond2_denoised_over_sigma = self.score_fn(inputs_hat, sigma_hat, clamp=clamp, cond=cond2)
                    
                #     denoised_over_sigma = uncond_denoised_over_sigma
                #     denoised_over_sigma += temperature * (cond1_denoised_over_sigma - uncond_denoised_over_sigma)
                #     denoised_over_sigma += temperature * (cond2_denoised_over_sigma - uncond_denoised_over_sigma)
                # else:
                uncond_denoised_over_sigma = self.score_fn(inputs_hat, sigma_hat, adj, ef, clamp=clamp, cond=uncond, add_cond=add_cond)
                cond_denoised_over_sigma = self.score_fn(inputs_hat, sigma_hat, adj, ef, clamp=clamp, cond=cond, add_cond=add_cond)
                denoised_over_sigma = uncond_denoised_over_sigma + temperature * (cond_denoised_over_sigma - uncond_denoised_over_sigma)
            else:
                denoised_over_sigma = self.score_fn(inputs_hat, sigma_hat, adj, ef, clamp=clamp, cond=cond)
            inputs_next = inputs_hat + (sigma_next - sigma_hat) * denoised_over_sigma

            # second order correction, if not the last timestep
            if sigma_next != 0:
                denoised_prime_over_sigma = self.score_fn(inputs_next, sigma_next, adj, ef, clamp=clamp, cond=cond, add_cond=add_cond)
                inputs_next = inputs_hat + 0.5 * (sigma_next - sigma_hat) * (
                        denoised_over_sigma + denoised_prime_over_sigma)

            inputs = inputs_next

        if clamp:
            inputs = inputs.clamp(-1., 1.)
        return self.normalizer.unnormalize(inputs)
    
    # This is known as 'denoised_over_sigma' in the lucidrains repo.
    def score_fn(
            self,
            x,
            sigma,
            adj,
            ef,
            clamp: bool = False,
            cond=None,
            add_cond=None,
    ):
        denoised = self.preconditioned_network_forward(x, sigma, adj, ef, clamp=clamp, cond=cond, add_cond=add_cond)
        denoised_over_sigma = (x - denoised) / sigma

        return denoised_over_sigma

    # Adapted from https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/sampling.py
    @torch.no_grad()
    def log_likelihood(self, x, atol=1e-4, rtol=1e-4, clamp=False, normalize=True, cond=None):
        # Input to the ODE solver must be in normalized space.
        if normalize:
            x = self.normalizer.normalize(x)
        v = torch.randint_like(x, 2) * 2 - 1
        s_in = x.new_ones([x.shape[0]])
        fevals = 0

        def ode_fn(sigma, x):
            nonlocal fevals
            with torch.enable_grad():
                x = x[0].detach().requires_grad_()
                sigma = sigma * s_in
                padded_sigma = sigma.view(x.shape[0], *([1] * len(self.event_shape)))
                denoised = self.preconditioned_network_forward(x, sigma, clamp=clamp, cond=cond)
                denoised_over_sigma = (x - denoised) / padded_sigma
                fevals += 1
                grad = torch.autograd.grad((denoised_over_sigma * v).sum(), x)[0]
                d_ll = (v * grad).flatten(1).sum(1)
            return denoised_over_sigma.detach(), d_ll

        x_min = x, x.new_zeros([x.shape[0]])
        t = x.new_tensor([self.sigma_min, self.sigma_max])
        sol = odeint(ode_fn, x_min, t, atol=atol, rtol=rtol, method='dopri5')
        latent, delta_ll = sol[0][-1], sol[1][-1]
        ll_prior = torch.distributions.Normal(0, self.sigma_max).log_prob(latent).flatten(1).sum(1)

        return ll_prior + delta_ll, {'fevals': fevals}
    
    # training
    def loss_weight(self, sigma):
        return (sigma ** 2 + self.sigma_data ** 2) * (sigma * self.sigma_data) ** -2

    def noise_distribution(self, batch_size):
        return (self.P_mean + self.P_std * torch.randn((batch_size,), device=self.device)).exp()
    
    # def forward(self, inputs, cond=None, cond_drop_prob=0.0):
    def forward(self, x, adj, ef, cond=None, cond_drop_prob=0.0, add_cond=None):
        inputs = self.normalizer.normalize(x)

        batch_size, _ = cond.shape
        input_size, *event_shape = inputs.shape
        assert event_shape == self.event_shape, f'mismatch of event shape, ' \
                                                f'expected {self.event_shape}, got {event_shape}'

        sigmas = self.noise_distribution(batch_size).repeat_interleave(input_size // batch_size)
        padded_sigmas = sigmas.view(input_size, *([1] * len(self.event_shape)))

        noise = torch.randn_like(inputs)
        noised_inputs = inputs + padded_sigmas * noise  # alphas are 1. in the paper

        if cond is not None:
            _, cond = self.process_cond(cond, cond_drop_prob)
            # cond = cond.repeat_interleave(input_size // batch_size, dim=0)
        denoised = self.preconditioned_network_forward(noised_inputs, sigmas, adj, ef, cond=cond, add_cond=add_cond)
        losses = F.mse_loss(denoised, inputs, reduction='none')
        losses = reduce(losses, 'b ... -> b', 'mean')
        losses = losses * self.loss_weight(sigmas)
        return losses.mean()


@gin.configurable
class Trainer(object):
    def __init__(
            self,
            diffusion_model,
            dataset: Optional[torch.utils.data.Dataset] = None,
            train_batch_size: int = 16,
            small_batch_size: int = 16,
            gradient_accumulate_every: int = 1,
            train_lr: float = 1e-4,
            lr_scheduler: Optional[str] = "cosine",
            train_num_steps: int = 100000,
            ema_update_every: int = 10,
            ema_decay: float = 0.995,
            adam_betas: Tuple[float, float] = (0.9, 0.99),
            save_and_sample_every: int = 1000,
            weight_decay: float = 0.,
            results_folder: str = './results',
            amp: bool = False,
            fp16: bool = False,
            split_batches: bool = True,
            cond_type: str = "none",
            cond_drop_prob: float = 0.1,
            add_cond: bool = False,
    ):
        super().__init__()
        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision='fp16' if fp16 else 'no'
        )
        self.accelerator.native_amp = amp
        self.model = diffusion_model

        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'Number of trainable parameters: {num_params}.')

        self.save_and_sample_every = save_and_sample_every
        self.train_num_steps = train_num_steps
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset
        if dataset is not None:
            # If dataset size is less than 800K use the small batch size
            if len(dataset) < int(8e5):
                self.batch_size = small_batch_size
            else:
                self.batch_size = train_batch_size
            print(f'Using batch size: {self.batch_size}')
            # dataset and dataloader
            if isinstance(dataset, WindFarmLayoutDataset):
                # scores = dataset.scores
                # weights = torch.exp(scores).flatten()
                # sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
                # dl = DataLoader(dataset, batch_size=self.batch_size, sampler=sampler, pin_memory=True, num_workers=1)
                dl = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=1)
            elif isinstance(dataset, GraphWindFarmLayoutDataset):
                # scores = dataset.scores
                # weights = torch.exp(scores).flatten()
                # sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
                dl = GraphDataLoader(dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=1)
            dl = self.accelerator.prepare(dl)
            self.dl = cycle(dl)
        else:
            # No dataloader, train batch by batch
            self.batch_size = train_batch_size
            self.dl = None

        # optimizer, make sure that the bias and layer-norm weights are not decayed
        no_decay = ['bias', 'LayerNorm.weight', 'norm.weight', '.g']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay,
            },
            {
                'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]
        self.opt = torch.optim.AdamW(optimizer_grouped_parameters, lr=train_lr, betas=adam_betas)

        # for logging results in a folder periodically
        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)
            self.results_folder = pathlib.Path(results_folder)
            # self.results_folder.mkdir(exist_ok=True)

        # step counter state
        self.step = 0

        # prepare model, dataloader, optimizer with accelerator
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        if lr_scheduler == 'linear':
            print('using linear learning rate scheduler')
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.opt,
                lambda step: max(0, 1 - step / train_num_steps)
            )
        elif lr_scheduler == 'cosine':
            print('using cosine learning rate scheduler')
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.opt,
                train_num_steps
            )
        else:
            self.lr_scheduler = None

        self.model.normalizer.to(self.accelerator.device)
        self.ema.ema_model.normalizer.to(self.accelerator.device)
        self.cond_type = cond_type
        self.cond_drop_prob = cond_drop_prob
        self.add_cond = add_cond

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
        }

        # torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))
        torch.save(data, f"{self.results_folder}_epoch{milestone}.pt")

    def load(self, milestone: int):
        accelerator = self.accelerator
        device = accelerator.device

        # data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)
        data = torch.load(f"{self.results_folder}_epoch{milestone}.pt", map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        # self.opt.load_state_dict(data['opt'])
        self.ema.load_state_dict(data['ema'])

        # if exists(self.accelerator.scaler) and exists(data['scaler']):
        #     self.accelerator.scaler.load_state_dict(data['scaler'])

    # Train for the full number of steps.
    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:
            while self.step < self.train_num_steps:
                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    if self.cond_type == "none":
                        data, _ = next(self.dl)
                        data = data.to(device)
                        cond = None
                    else:
                        if isinstance(self.dataset, WindFarmLayoutDataset):
                            if self.add_cond:
                                data, cond, add_cond = next(self.dl)
                                data = data.to(device)
                                cond = cond.to(device)
                                add_cond = add_cond.to(device)
                            else:
                                data, cond = next(self.dl)
                                data = data.to(device)
                                cond = cond.to(device)
                                add_cond = None
                            
                            with self.accelerator.autocast():
                                loss = self.model(data, 
                                                  cond=cond,
                                                  cond_drop_prob=self.cond_drop_prob,
                                                  add_cond=add_cond)
                        else:
                            graph = next(self.dl)
                            if self.add_cond:
                                x, adj, ef, y, add_cond = graph.x, graph.edge_index, graph.edge_attr, graph.y, graph.add_cond
                                add_cond = add_cond.to(device)
                            else:
                                x, adj, ef, y = graph.x, graph.edge_index, graph.edge_attr, graph.y
                                add_cond = None
                            x = x.to(device)
                            adj = adj.to(device)
                            ef = ef.to(device)
                            y = y.to(device)

                            with self.accelerator.autocast():
                                loss = self.model(x, 
                                                  adj,
                                                  ef,
                                                  cond=y,
                                                  cond_drop_prob=self.cond_drop_prob,
                                                  add_cond=add_cond)
                                
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                pbar.set_description(f'loss: {total_loss:.4f}')
                wandb.log({
                    'step': self.step,
                    'loss': total_loss,
                    'lr': self.opt.param_groups[0]['lr']
                })

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                if accelerator.is_main_process:
                    self.ema.to(device)
                    self.ema.update()

                    if self.step % self.save_and_sample_every == 0:
                        self.save(self.step)
                self.step += 1
                pbar.update(1)

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

        accelerator.print('training complete')
        self.save(self.step)

    # Allow user to pass in external data.
    def train_on_batch(
            self,
            data: torch.Tensor,
            use_wandb=True,
            splits=1,  # number of splits to split the batch into
            **kwargs,
    ):
        accelerator = self.accelerator
        device = accelerator.device
        data = data.to(device)

        total_loss = 0.
        if splits == 1:
            with self.accelerator.autocast():
                loss = self.model(data, **kwargs)
                total_loss += loss.item()
            self.accelerator.backward(loss)
        else:
            assert splits > 1 and data.shape[0] % splits == 0
            split_data = torch.split(data, data.shape[0] // splits)

            for idx, d in enumerate(split_data):
                with self.accelerator.autocast():
                    # Split condition as well
                    new_kwargs = {}
                    for k, v in kwargs.items():
                        if isinstance(v, torch.Tensor):
                            new_kwargs[k] = torch.split(v, v.shape[0] // splits)[idx]
                        else:
                            new_kwargs[k] = v

                    loss = self.model(d, **new_kwargs)
                    loss = loss / splits
                    total_loss += loss.item()
                self.accelerator.backward(loss)

        accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
        if use_wandb:
            wandb.log({
                'step': self.step,
                'loss': total_loss,
                'lr': self.opt.param_groups[0]['lr'],
            })

        accelerator.wait_for_everyone()

        self.opt.step()
        self.opt.zero_grad()

        accelerator.wait_for_everyone()

        self.step += 1
        if accelerator.is_main_process:
            self.ema.to(device)
            self.ema.update()

            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                self.save(self.step)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return total_loss

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def safe_tensor(x: torch.Tensor,
                nan: float = 0.0,
                posinf: float = 1e6,
                neginf: float = -1e6) -> torch.Tensor:
    x = torch.where(torch.isnan(x), torch.full_like(x, nan), x)
    x = torch.where(x == float('inf'), torch.full_like(x, posinf), x)
    x = torch.where(x == float('-inf'), torch.full_like(x, neginf), x)
    return x


class SimpleAutoencoder(nn.Module):
    def __init__(self, activation_function: nn.Module, hidden_layers: List[int], verbose: bool = False) -> None:
        super(SimpleAutoencoder, self).__init__()
        self.seq = nn.Sequential()
        n_neurons_in = 115

        n_in = n_neurons_in
        for i, n_out in enumerate(hidden_layers):
            self.seq.add_module('fc' + str(i), nn.Linear(n_in, n_out, bias=True))
            self.seq.add_module('act_fn' + str(i), activation_function())
            n_in = n_out

        self.seq.add_module('final_fc', nn.Linear(n_in, n_neurons_in, bias=True))

        if verbose:
            print(self.seq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.seq(x)
        return safe_tensor(out)


class VAE(nn.Module):
    def __init__(self, activation_function: nn.Module, hidden_layers: List[int], latent_dim: int = 10) -> None:
        super(VAE, self).__init__()

        self.encoder_layers = nn.ModuleList()
        n_in = 115

        for n_out in hidden_layers:
            self.encoder_layers.append(nn.Linear(n_in, n_out))
            self.encoder_layers.append(activation_function())
            n_in = n_out

        self.fc_mu = nn.Linear(n_in, latent_dim)
        self.fc_logvar = nn.Linear(n_in, latent_dim)

        self.decoder_layers = nn.ModuleList()
        n_in = latent_dim
        reversed_hidden = hidden_layers[::-1]

        for n_out in reversed_hidden:
            self.decoder_layers.append(nn.Linear(n_in, n_out))
            self.decoder_layers.append(activation_function())
            n_in = n_out

        self.final_layer = nn.Linear(n_in, 115)

    def encode(self, x):
        h = x
        for layer in self.encoder_layers:
            h = layer(h)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        mu = safe_tensor(mu, nan=0.0, posinf=1e6, neginf=-1e6)
        logvar = safe_tensor(logvar, nan=0.0, posinf=10.0, neginf=-10.0)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        std = torch.exp(0.5 * logvar)
        std = torch.clamp(std, min=1e-6, max=1e2)

        eps = torch.randn_like(std)
        z = mu + eps * std
        return safe_tensor(z, nan=0.0, posinf=1e6, neginf=-1e6)

    def decode(self, z):
        h = z
        for layer in self.decoder_layers:
            h = layer(h)
        out = self.final_layer(h)
        return safe_tensor(out, nan=0.0, posinf=1e6, neginf=-1e6)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def generate_synthetic_data(self, num_samples, device='cpu'):
        z = torch.randn(num_samples, self.fc_mu.out_features).to(device)
        out = self.decode(z)
        return safe_tensor(out, nan=0.0, posinf=1e6, neginf=-1e6)


class Threshold(nn.Module):
    def __init__(self, threshold: torch.Tensor):
        super(Threshold, self).__init__()
        self.threshold = nn.Parameter(threshold, requires_grad=False)


class BinaryClassifier(nn.Module):
    def __init__(self, activation_function: nn.Module, hidden_layers: List[int], verbose: bool = False) -> None:
        super(BinaryClassifier, self).__init__()
        self.seq = nn.Sequential()
        n_neurons_in = 115

        n_in = n_neurons_in
        for i, n_out in enumerate(hidden_layers):
            self.seq.add_module('fc' + str(i), nn.Linear(n_in, n_out, bias=True))
            self.seq.add_module('act_fn' + str(i), activation_function())
            n_in = n_out

        self.seq.add_module('final_fc', nn.Linear(n_in, 1, bias=True))
        self.seq.add_module('sigmoid', nn.Sigmoid())

        if verbose:
            print(self.seq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.seq(x)
        return safe_tensor(out, nan=0.0, posinf=1.0, neginf=0.0)


class NormalizingModel(nn.Module):
    def __init__(self, model: torch.nn.Module, sub: torch.Tensor, div: torch.Tensor) -> None:
        super(NormalizingModel, self).__init__()
        self.sub = nn.Parameter(sub, requires_grad=False)
        self.div = nn.Parameter(div, requires_grad=False)
        self.model = model

    def set_sub_div(self, sub: torch.Tensor, div: torch.Tensor) -> None:
        self.sub = nn.Parameter(sub, requires_grad=False)
        self.div = nn.Parameter(div, requires_grad=False)

    def forward(self, x: torch.Tensor):
        return self.model(self.normalize(x))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        safe_div = torch.where(self.div == 0, torch.ones_like(self.div), self.div)
        out = (x - self.sub) / safe_div
        return safe_tensor(out, nan=0.0, posinf=1e6, neginf=-1e6)
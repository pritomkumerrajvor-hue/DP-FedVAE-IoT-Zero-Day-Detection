from types import SimpleNamespace
from typing import List, Dict, Tuple, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from context_printer import Color
from context_printer import ContextPrinter as Ctp
from torch.utils.data import DataLoader

from architectures import Threshold, VAE
from federated_util import model_poisoning, model_aggregation
from metrics import BinaryClassificationResult
from print_util import print_autoencoder_loss_stats, print_rates, print_autoencoder_loss_header
from data import device_names


# ========================================================
# Standard Autoencoder Optimization
# ========================================================
def optimize(model: nn.Module,
             data: torch.Tensor,
             optimizer: torch.optim.Optimizer,
             criterion: torch.nn.Module) -> torch.Tensor:
    output = model(data)
    loss = criterion(output, model.normalize(data))
    loss_mean = loss.mean()

    if not torch.isfinite(loss_mean):
        print("Non-finite AE loss detected, skipping batch")
        optimizer.zero_grad()
        return torch.full(
            (data.size(0),),
            1e6,
            device=data.device,
            dtype=data.dtype
        )

    optimizer.zero_grad()
    loss_mean.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return loss


# ========================================================
# VAE LOSS & OPTIMIZATION (For Zero-Day Detection)
# ========================================================
def vae_loss_function(recon_x, x, mu, logvar):
    mse = F.mse_loss(recon_x, x, reduction='mean')
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return mse + 0.0001 * kld


def optimize_vae(model: nn.Module,
                 data: torch.Tensor,
                 optimizer: torch.optim.Optimizer) -> torch.Tensor:
    normalized_data = model.normalize(data)

    recon_batch, mu, logvar = model.model(normalized_data)
    loss = vae_loss_function(recon_batch, normalized_data, mu, logvar)

    if not torch.isfinite(loss):
        print("Non-finite VAE loss detected, skipping batch")
        optimizer.zero_grad()
        return torch.tensor(1e6, device=data.device)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return loss.detach()


# ========================================================
# Train Autoencoder (Centralized / Local)
# ========================================================
def train_autoencoder(model: nn.Module,
                      params: SimpleNamespace,
                      train_loader,
                      lr_factor: float = 1.0) -> None:
    criterion = nn.MSELoss(reduction='none')
    optimizer = params.optimizer(model.parameters(), **params.optimizer_params)

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * lr_factor

    scheduler = params.lr_scheduler(optimizer, **params.lr_scheduler_params)
    model.train()

    num_elements = len(train_loader.dataset)
    num_batches = len(train_loader)
    batch_size = train_loader.batch_size

    print_autoencoder_loss_header(first_column='Epoch', print_lr=True)

    is_vae = isinstance(model.model, VAE)

    for epoch in range(params.epochs):
        losses = torch.zeros(num_elements)

        for i, (data,) in enumerate(train_loader):
            if params.cuda:
                data = data.cuda()

            start = i * batch_size
            end = start + batch_size
            if i == num_batches - 1:
                end = num_elements

            if is_vae:
                loss = optimize_vae(model, data, optimizer)
                losses[start:end] = loss.detach().item()
            else:
                loss = optimize(model, data, optimizer, criterion)
                losses[start:end] = loss.mean(dim=1).detach().cpu()

        print_autoencoder_loss_stats(
            '[{}/{}]'.format(epoch + 1, params.epochs),
            losses,
            lr=optimizer.param_groups[0]['lr']
        )

        scheduler.step()


# ========================================================
# Federated Training (FedSGD Helper)
# ========================================================
def train_autoencoders_fedsgd(global_model: nn.Module,
                              models: List[nn.Module],
                              dls: List[DataLoader],
                              params: SimpleNamespace,
                              lr_factor: float = 1.0,
                              mimicked_client_id: Optional[int] = None):
    criterion = nn.MSELoss(reduction='none')
    lr = params.optimizer_params['lr'] * lr_factor
    is_vae = isinstance(global_model.model, VAE)

    for model in models:
        model.train()

    for data_tuple in zip(*dls):
        for model, (data,) in zip(models, data_tuple):
            if params.cuda:
                data = data.cuda()

            optimizer = params.optimizer(
                model.parameters(),
                lr=lr,
                weight_decay=params.optimizer_params['weight_decay']
            )

            if is_vae:
                optimize_vae(model, data, optimizer)
            else:
                optimize(model, data, optimizer, criterion)

        models = model_poisoning(
            global_model,
            models,
            params,
            mimicked_client_id=mimicked_client_id,
            verbose=False
        )

        global_model, models = model_aggregation(
            global_model,
            models,
            params,
            verbose=False
        )

    return global_model, models


# ========================================================
# Compute Reconstruction Losses
# ========================================================
def compute_reconstruction_losses(model: nn.Module, dataloader) -> torch.Tensor:
    with torch.no_grad():
        criterion = nn.MSELoss(reduction='none')
        model.eval()

        num_elements = len(dataloader.dataset)
        num_batches = len(dataloader)
        batch_size = dataloader.batch_size

        losses = torch.zeros(num_elements)
        is_vae = isinstance(model.model, VAE)

        for i, (x,) in enumerate(dataloader):
            if next(model.parameters()).is_cuda:
                x = x.cuda()

            normalized_x = model.normalize(x)

            if is_vae:
                output, _, _ = model.model(normalized_x)
            else:
                output = model(x)

            loss = criterion(output, normalized_x)

            start = i * batch_size
            end = start + batch_size
            if i == num_batches - 1:
                end = num_elements

            losses[start:end] = loss.mean(dim=1).cpu()

        return losses


# ========================================================
# Testing Autoencoder
# ========================================================
def test_autoencoder(model: nn.Module,
                     threshold: nn.Module,
                     dataloaders: Dict[str, DataLoader]):
    print_autoencoder_loss_header(print_positives=True)
    result = BinaryClassificationResult()

    for key, dataloader in dataloaders.items():
        losses = compute_reconstruction_losses(model, dataloader)
        predictions = torch.gt(losses, threshold.threshold.cpu()).int()

        current_results = count_scores(predictions, is_attack=(key != 'benign'))
        title = ' '.join(key.split('_')).title()

        print_autoencoder_loss_stats(
            title,
            losses,
            positives=current_results.tp + current_results.fp,
            n_samples=current_results.n_samples()
        )

        result += current_results

    return result


# ========================================================
# Multi Training Wrapper
# ========================================================
def multitrain_autoencoders(trains: List[Tuple[str, DataLoader, nn.Module]],
                            params: SimpleNamespace,
                            lr_factor: float = 1.0,
                            main_title: str = 'Multitrain autoencoders',
                            color: Union[str, Color] = Color.NONE):
    Ctp.enter_section(main_title, color)

    for i, (title, dataloader, model) in enumerate(trains):
        Ctp.enter_section(
            '[{}/{}] '.format(i + 1, len(trains)) + title,
            color=Color.NONE,
            header='      '
        )
        train_autoencoder(model, params, dataloader, lr_factor)
        Ctp.exit_section()

    Ctp.exit_section()


# ========================================================
# Threshold Utilities
# ========================================================
def compute_threshold_value(losses: torch.Tensor, quantile: Optional[float] = None):
    if quantile is None:
        return losses.mean() + losses.std()
    else:
        return losses.quantile(quantile)


def compute_thresholds(opts: List[Tuple[str, DataLoader, nn.Module]],
                       quantile: Optional[float] = None):
    thresholds = []

    for title, dataloader, model in opts:
        losses = compute_reconstruction_losses(model, dataloader)
        threshold_value = compute_threshold_value(losses, quantile)
        threshold = Threshold(threshold_value)
        thresholds.append(threshold)

    return thresholds


# ========================================================
# Evaluation Utilities
# ========================================================
def count_scores(predictions: torch.Tensor, is_attack: bool):
    positive_predictions = predictions.sum().item()
    negative_predictions = len(predictions) - positive_predictions

    results = BinaryClassificationResult()

    if is_attack:
        results.add_tp(positive_predictions)
        results.add_fn(negative_predictions)
    else:
        results.add_fp(positive_predictions)
        results.add_tn(negative_predictions)

    return results


def multitest_autoencoders(tests: List[Tuple[str, Dict[str, DataLoader], nn.Module, nn.Module]]):
    result = BinaryClassificationResult()

    for title, dataloaders, model, threshold in tests:
        Ctp.print(title, bold=True)
        current_result = test_autoencoder(model, threshold, dataloaders)
        result += current_result
        print_rates(current_result)

    print_rates(result)
    return result


# ========================================================
# Federated Thresholds
# ========================================================
def federated_thresholds(models: List[nn.Module],
                         threshold_dls: List[DataLoader],
                         global_threshold: nn.Module,
                         params: SimpleNamespace,
                         global_thresholds: List[float]):
    Ctp.enter_section("Computing Federated Thresholds", Color.DARK_PURPLE)

    thresholds = compute_thresholds(
        opts=list(zip(
            [f'Client {i}' for i in range(len(models))],
            threshold_dls,
            models
        )),
        quantile=params.quantile
    )

    all_thresholds_tensor = torch.stack([t.threshold for t in thresholds])
    avg_threshold_value = all_thresholds_tensor.mean()

    global_threshold.threshold = nn.Parameter(avg_threshold_value)
    global_thresholds.append(avg_threshold_value.item())

    Ctp.print(f"Global Threshold set to: {avg_threshold_value.item():.4f}")
    Ctp.exit_section()


# ========================================================
# Federated Testing
# ========================================================
def federated_testing(global_model: nn.Module,
                      global_threshold: nn.Module,
                      local_test_dls_dicts: List[Dict[str, DataLoader]],
                      new_test_dls_dict: Dict[str, DataLoader],
                      params: SimpleNamespace,
                      local_results: List[BinaryClassificationResult],
                      new_devices_results: List[BinaryClassificationResult]):
    Ctp.enter_section("Federated Testing", Color.BLUE)

    tests_local = []
    for i, client_devices in enumerate(params.clients_devices):
        tests_local.append((
            f"Testing Global Model on Client {i} ({device_names(client_devices)})",
            local_test_dls_dicts[i],
            global_model,
            global_threshold
        ))

    res_local = multitest_autoencoders(tests_local)
    local_results.append(res_local)

    tests_new = []
    for i in range(len(params.clients_devices)):
        tests_new.append((
            f"Testing Global Model on New Devices (View {i})",
            new_test_dls_dict,
            global_model,
            global_threshold
        ))

    res_new = multitest_autoencoders(tests_new)
    new_devices_results.append(res_new)

    Ctp.exit_section()
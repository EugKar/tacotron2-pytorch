import os
import time
import argparse
import math
from numpy import finfo

import torch
from torch.utils.data import DataLoader
from torch._six import inf

import torch_xla
import torch_xla_py.data_parallel as dp
import torch_xla_py.utils as xu
import torch_xla_py.xla_model as xm
import torch_xla.distributed.parallel_loader as pl

from model import VAE
from data_utils import TextMelLoader, TextMelCollate
from loss_function import Tacotron2Loss, VAELoss
from logger import Tacotron2Logger
from hparams import create_hparams

def clip_grad_norm_xla_(parameters, max_norm, norm_type=2):
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        device = parameters[0].device
        total_norm = torch.zeros([], device=device if parameters else None)
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type) ** norm_type
            total_norm.add_(param_norm)
        total_norm = total_norm ** (1. / norm_type)
    clip_coef = torch.tensor(max_norm, device=device) / (total_norm + 1e-6)
    for p in parameters:
        p.grad.data.mul_(torch.where(clip_coef < 1, clip_coef, torch.tensor(1., device=device)))
    return total_norm

class VarianceClipper(object):

    def __init__(self, hparams):
        self.latent_sigma_min = hparams.latent_sigma_min
        self.observed_sigma_min = hparams.observed_sigma_min

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'latent_prior_sigma'):
            param = module.latent_prior_sigma
            param = param.clamp(min=self.latent_sigma_min)
            module.latent_prior_sigma.data = param
        if hasattr(module, 'observed_prior_sigma'):
            param = module.observed_prior_sigma
            param = param.clamp(min=self.observed_sigma_min)
            module.observed_prior_sigma.data = param

def prepare_dataloaders(hparams):
    # Get data, data loaders and collate function ready
    trainset = TextMelLoader(hparams.training_files, hparams)
    valset = TextMelLoader(hparams.validation_files, hparams)
    collate_fn = TextMelCollate(hparams.n_frames_per_step,
        max_input_len=hparams.max_input_len,
        max_target_len=hparams.max_frames)

    train_sampler = None
    val_sampler = None
    shuffle = True

    if xm.xrt_world_size() > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            trainset,
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            valset,
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=False)
        shuffle = False

    train_loader = DataLoader(trainset, num_workers=hparams.num_workers,
        shuffle=shuffle,
        sampler=train_sampler,
        batch_size=hparams.batch_size,
        collate_fn=collate_fn)
    val_loader = DataLoader(valset, num_workers=hparams.num_workers,
        shuffle=False,
        sampler=val_sampler,
        batch_size=hparams.batch_size,
        collate_fn=collate_fn)
    return train_loader, val_loader, collate_fn

# def prepare_directories_and_logger(output_directory, log_directory):
#     if not os.path.isdir(output_directory):
#         os.makedirs(output_directory)
#         os.chmod(output_directory, 0o775)
#     logger = Tacotron2Logger(os.path.join(output_directory, log_directory))
#     return logger


def load_checkpoint(checkpoint_path, model):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint '{}' from iteration {}" .format(
        checkpoint_path, iteration))
    return model, checkpoint_dict['optimizer'], learning_rate, iteration


def save_checkpoint(model, optimizer_dict, learning_rate, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
        iteration, filepath))
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer_dict,
                'learning_rate': learning_rate}, filepath)

def train(output_directory, log_directory, checkpoint_path, hparams):
    """Training and validation logging results to tensorboard and stdout

    Params
    ------
    output_directory (string): directory to save checkpoints
    log_directory (string) directory to save tensorboard logs
    checkpoint_path(string): checkpoint path
    hparams (object): comma separated list of "name=value" pairs.
    """
    torch.manual_seed(hparams.seed)

    num_cores = None if hparams.num_cores == -1 else hparams.num_cores
    device = xm.xla_device()
    learning_rate = hparams.learning_rate * xm.xrt_world_size()

    val_criterion = Tacotron2Loss()

    clipper = VarianceClipper(hparams)

    # logger = prepare_directories_and_logger(
    #     output_directory, log_directory)

    train_loader, val_loader, _ = prepare_dataloaders(hparams)

    # Load checkpoint if one exists
    epoch_offset = 0
    optimizer_dict = None
    model = VAE(hparams)
    if checkpoint_path is not None:
        model, optimizer_dict, _learning_rate, epoch_offset = load_checkpoint(
            checkpoint_path, model)
        if hparams.use_saved_learning_rate:
            learning_rate = _learning_rate
    model = model.to(device=device)

    epoch_offset += 1  # next iteration is iteration + 1

    if hparams.autograd_detect_anomalies:
        torch.autograd.set_detect_anomaly(True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                                     weight_decay=hparams.weight_decay)

    criterion = VAELoss(enable_numerical_check=False)
    def train_loop_fn(loader):
        tracker = xm.RateTracker()
        model.train()

        for iteration, batch in enumerate(loader):
            start = time.perf_counter()

            optimizer.zero_grad()

            (x, y), speaker_ids = model.parse_batch(batch,
                                                    max_input_length=hparams.max_input_len,
                                                    max_output_length=hparams.max_frames,
                                                    device=device)

            (y_pred, latent_params, observed_params,
                latent_prior_params, observed_prior_params) = model(x)

#            elbo, mel_loss, gate_loss = criterion(y_pred, latent_params,
#                                                  observed_params,
#                                                  latent_prior_params,
#                                                  observed_prior_params,
#                                                  speaker_ids, y)
            print('Calculating loss')
            mel_loss, gate_loss = val_criterion(y_pred, y)
            loss = mel_loss
#            loss = elbo + gate_loss
            print('Backward step')
            loss.backward()

            grad_norm = clip_grad_norm_xla_(model.parameters(), hparams.grad_clip_thresh)
            print('Optimizer step')
            xm.optimizer_step(optimizer)
            #print(torch_xla._XLAC._xla_metrics_report())

            model.apply(clipper)
            tracker.add(hparams.batch_size)
            duration = time.perf_counter() - start
            print('Materializing calculations')
            print("Device {} iteration {} epoch {}: train loss {:.6f} Grad Norm {:.6f} {:.2f}s/it".format(
                device, iteration, epoch, loss.item(), grad_norm, duration))
            if hparams.metrics_debug:
                print(torch_xla._XLAC._xla_metrics_report())
        return optimizer.state_dict()

    def val_loop_fn(loader):
        model.eval()
        val_loss = 0.0
        total_samples = 0
        for batch in loader:
            (x, y), speaker_ids = model.parse_batch(batch,
                                                    max_input_length=hparams.max_input_len,
                                                    max_output_length=hparams.max_frames,
                                                    device=device)
            (y_pred, latent_params, observed_params,
                latent_prior_params, observed_prior_params) = model(x)
            mel_loss, gate_loss = val_criterion(y_pred, y)
            loss = mel_loss + gate_loss
            val_loss += loss
            total_samples += batch.size()[0]
        val_loss = val_loss / total_samples
        return val_loss

    # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in range(epoch_offset, hparams.epochs):
        para_loader = pl.ParallelLoader(train_loader, [device])
        optimizer_dict_save = train_loop_fn(para_loader.per_device_loader(device))
        para_loader = pl.ParallelLoader(val_loader, [device])
        val_loss = val_loop_fn(para_loader.per_device_loader(device))
        # logger.log_training(
        #     reduced_loss, reduced_mel_loss, reduced_elbo, grad_norm,
        #     learning_rate, duration, iteration)

        print("Validation loss {}: {:9f}  ".format(epoch, val_loss))
        # logger.log_validation(val_loss, model, y, y_pred, iteration)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str,
                        help='directory to save checkpoints')
    parser.add_argument('-l', '--log_directory', type=str,
                        help='directory to save tensorboard logs')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=False, help='checkpoint path')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')

    args = parser.parse_args()
    hparams = create_hparams(args.hparams)
    hparams.set_hparam('enable_pack_padded_sequence', False)

    print("Dynamic Loss Scaling:", hparams.dynamic_loss_scaling)

    train(args.output_directory, args.log_directory, args.checkpoint_path,
          hparams)

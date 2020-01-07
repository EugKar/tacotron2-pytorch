import os
import time
import argparse
import math
from numpy import finfo

import torch
from distributed import apply_gradient_allreduce
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
# from pytorch_memlab import MemReporter

from model import VAE
from data_utils import TextMelLoader, TextMelCollate
from loss_function import Tacotron2Loss, VAELoss
from logger import Tacotron2Logger
from hparams import create_hparams

import multiprocessing
multiprocessing.set_start_method('spawn', True)

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

def reduce_tensor(tensor, n_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= n_gpus
    return rt


def init_distributed(hparams, n_gpus, rank, group_name):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    print("Initializing Distributed")

    # Set cuda device so everything is done on the right GPU.
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Initialize distributed communication
    dist.init_process_group(
        backend=hparams.dist_backend, init_method=hparams.dist_url,
        world_size=n_gpus, rank=rank, group_name=group_name)

    print("Done initializing distributed")


def prepare_dataloaders(hparams):
    # Get data, data loaders and collate function ready
    trainset = TextMelLoader(hparams.training_files, hparams)
    valset = TextMelLoader(hparams.validation_files, hparams)
    collate_fn = TextMelCollate(hparams.n_frames_per_step)

    if hparams.distributed_run:
        train_sampler = DistributedSampler(trainset)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_loader = DataLoader(trainset, num_workers=1, shuffle=shuffle,
                              sampler=train_sampler,
                              batch_size=hparams.batch_size, pin_memory=False,
                              drop_last=True, collate_fn=collate_fn)
    return train_loader, valset, collate_fn


def prepare_directories_and_logger(output_directory, log_directory, rank):
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        logger = Tacotron2Logger(os.path.join(output_directory, log_directory))
    else:
        logger = None
    return logger


def load_model(hparams):
    model = VAE(hparams).cuda()
    if hparams.fp16_run:
        model.synthesizer.decoder.attention_layer.score_mask_value = finfo('float16').min

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    return model


def warm_start_model(checkpoint_path, model, ignore_layers):
    assert os.path.isfile(checkpoint_path)
    print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model_dict = checkpoint_dict['state_dict']
    prefix = 'synthesizer'
    if len(ignore_layers) > 0:
        model_dict = {f'{prefix}.{k}': v for k, v in model_dict.items()
                      if k not in ignore_layers}
        dummy_dict = model.state_dict()
        dummy_dict.update(model_dict)
        model_dict = dummy_dict
    model.load_state_dict(model_dict)
    return model


def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint '{}' from iteration {}" .format(
        checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
        iteration, filepath))
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)


def validate(model, criterion, valset, iteration, batch_size, n_gpus,
             collate_fn, logger, distributed_run, rank):
    """Handles all the validation scoring and printing"""
    model.eval()
    with torch.no_grad():
        val_sampler = DistributedSampler(valset) if distributed_run else None
        val_loader = DataLoader(valset, sampler=val_sampler, num_workers=1,
                                shuffle=False, batch_size=batch_size,
                                pin_memory=False, collate_fn=collate_fn)

        val_loss = 0.0
        for i, batch in enumerate(val_loader):
            (x, y), speaker_ids = model.parse_batch(batch)
            (y_pred, latent_params, observed_params,
             latent_prior_params, observed_prior_params) = model(x)
            mel_loss, gate_loss = criterion(y_pred, y)
            loss = mel_loss + gate_loss

            if distributed_run:
                reduced_val_loss = reduce_tensor(loss.data, n_gpus).item()
            else:
                reduced_val_loss = loss.item()
            val_loss += reduced_val_loss
        val_loss = val_loss / (i + 1)

    model.train()
    if rank == 0:
        print("Validation loss {}: {:9f}  ".format(iteration, reduced_val_loss))
        logger.log_validation(reduced_val_loss, model, y, y_pred, iteration)


def train(output_directory, log_directory, checkpoint_path, warm_start, n_gpus,
          rank, group_name, hparams):
    """Training and validation logging results to tensorboard and stdout

    Params
    ------
    output_directory (string): directory to save checkpoints
    log_directory (string) directory to save tensorboard logs
    checkpoint_path(string): checkpoint path
    n_gpus (int): number of gpus
    rank (int): rank of current gpu
    hparams (object): comma separated list of "name=value" pairs.
    """
    if hparams.distributed_run:
        init_distributed(hparams, n_gpus, rank, group_name)

    # torch.manual_seed(hparams.seed)
    # torch.cuda.manual_seed(hparams.seed)

    model = load_model(hparams)
    learning_rate = hparams.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=hparams.weight_decay)

    if hparams.fp16_run:
        from apex import amp
        model, optimizer = amp.initialize(
            model, optimizer, opt_level='O2')

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    criterion = VAELoss()
    val_criterion = Tacotron2Loss()

    clipper = VarianceClipper(hparams)

    logger = prepare_directories_and_logger(
        output_directory, log_directory, rank)

    train_loader, valset, collate_fn = prepare_dataloaders(hparams)

    # Load checkpoint if one exists
    iteration = 0
    epoch_offset = 0
    batch_offset = 0
    if checkpoint_path is not None:
        if warm_start:
            model = warm_start_model(
                checkpoint_path, model, hparams.ignore_layers)
        else:
            model, optimizer, _learning_rate, iteration = load_checkpoint(
                checkpoint_path, model, optimizer)
            if hparams.use_saved_learning_rate:
                learning_rate = _learning_rate
            iteration += 1  # next iteration is iteration + 1
            epoch_offset = max(0, int(iteration / len(train_loader)))
            iters_per_epoch = math.ceil(len(train_loader.dataset) / train_loader.batch_size)
            batch_offset = iteration % iters_per_epoch
    if hparams.autograd_detect_anomalies:
        torch.autograd.set_detect_anomaly(True)

    # reporter = MemReporter(model)

    smaller_batch_size = hparams.batch_size // hparams.smaller_batch_count

    model.train()
    is_overflow = False
    # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in range(epoch_offset, hparams.epochs):
        print("Epoch: {}".format(epoch))
        for i, big_batch in enumerate(train_loader):
            # if i < batch_offset:
            #     continue
            batch_offset = 0
            start = time.perf_counter()
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

            model.zero_grad()

            max_length = int(big_batch[1].max())
            max_output_length = int(big_batch[-2].max())
            batch_start = 0
            step_elbo, step_mel_loss, step_loss = 0, 0, 0
            for _ in range(hparams.smaller_batch_count):
                batch = [t[batch_start:batch_start+smaller_batch_size, ...] for t in big_batch]
                batch_start += smaller_batch_size
                (x, y), speaker_ids = model.parse_batch(batch, max_input_length=max_length,
                                                        max_output_length=max_output_length)
                (y_pred, latent_params, observed_params,
                latent_prior_params, observed_prior_params) = model(x)

                try:
                    elbo, mel_loss, gate_loss = criterion(y_pred, latent_params, observed_params,
                        latent_prior_params, observed_prior_params, speaker_ids, y)
                    elbo /= hparams.smaller_batch_count
                    mel_loss /= hparams.smaller_batch_count
                    gate_loss /= hparams.smaller_batch_count
                except ValueError:
                    if rank == 0:
                        checkpoint_path = os.path.join(
                            output_directory, "checkpoint_{}".format(iteration))
                        save_checkpoint(model, optimizer, learning_rate, iteration,
                                        checkpoint_path)
                    raise

                loss = elbo + gate_loss
                # reporter.report(verbose=True)
                if hparams.distributed_run:
                    reduced_loss = reduce_tensor(loss.data, n_gpus).item()
                    reduced_mel_loss = reduce_tensor(mel_loss.data, n_gpus).item()
                    reduced_elbo = reduce_tensor(elbo.data, n_gpus).item()
                else:
                    reduced_loss = loss.item()
                    reduced_mel_loss = mel_loss.item()
                    reduced_elbo = elbo.item()
                if hparams.fp16_run:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                # reporter.report(verbose=True)
                step_elbo += float(reduced_elbo)
                step_mel_loss += float(reduced_mel_loss)
                step_loss += float(reduced_loss)


            if hparams.fp16_run:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    amp.master_params(optimizer), hparams.grad_clip_thresh)
                is_overflow = math.isnan(grad_norm)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), hparams.grad_clip_thresh)

            optimizer.step()

            model.apply(clipper)

            if not is_overflow and rank == 0:
                duration = time.perf_counter() - start
                print("Train loss {} {:.6f} mel loss {:.6f} ELBO {:.6f} Grad Norm {:.6f} {:.2f}s/it".format(
                    iteration, step_loss, step_mel_loss, step_elbo, grad_norm, duration))
                logger.log_training(
                    reduced_loss, reduced_mel_loss, reduced_elbo, grad_norm, learning_rate, duration, iteration)

            if not is_overflow and (iteration % hparams.iters_per_checkpoint == 0):
                validate(model, val_criterion, valset, iteration,
                         hparams.batch_size, n_gpus, collate_fn, logger,
                         hparams.distributed_run, rank)
                if rank == 0:
                    checkpoint_path = os.path.join(
                        output_directory, "checkpoint_{}".format(iteration))
                    save_checkpoint(model, optimizer, learning_rate, iteration,
                                    checkpoint_path)

            iteration += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str,
                        help='directory to save checkpoints')
    parser.add_argument('-l', '--log_directory', type=str,
                        help='directory to save tensorboard logs')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=False, help='checkpoint path')
    parser.add_argument('--warm_start', action='store_true',
                        help='load model weights only, ignore specified layers')
    parser.add_argument('--n_gpus', type=int, default=1,
                        required=False, help='number of gpus')
    parser.add_argument('--rank', type=int, default=0,
                        required=False, help='rank of current gpu')
    parser.add_argument('--group_name', type=str, default='group_name',
                        required=False, help='Distributed group name')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')

    args = parser.parse_args()
    hparams = create_hparams(args.hparams)

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    print("FP16 Run:", hparams.fp16_run)
    print("Dynamic Loss Scaling:", hparams.dynamic_loss_scaling)
    print("Distributed Run:", hparams.distributed_run)
    print("cuDNN Enabled:", hparams.cudnn_enabled)
    print("cuDNN Benchmark:", hparams.cudnn_benchmark)

    train(args.output_directory, args.log_directory, args.checkpoint_path,
          args.warm_start, args.n_gpus, args.rank, args.group_name, hparams)

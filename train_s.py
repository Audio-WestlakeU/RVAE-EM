import os
import sys
import argparse
import json
import warnings
import random
import numpy as np
import torch
import torch.multiprocessing as mp
import logging
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader, RandomSampler
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from torchinfo import summary
import time
from datetime import datetime
from pesq import pesq
from pystoi import stoi

from utils.env import build_env, dict_to_namespace
from utils.utils import (
    scan_checkpoint,
    load_checkpoint,
    set_optimizer,
    set_scheduler,
    save_checkpoint,
    plot_spectrogram,
    wav_path2txt,
)
from model.RVAE import RVAE as MODEL
from model.lossF import LossF
from dataset.mydataset import MyDataset as dataset
from dataset.io import aud2sptm_woDC as data_preprocess
from dataset.io import sptm_woDC2aud as data_postprocess

# global settings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.simplefilter(action="ignore", category=UserWarning)

# CUDNN
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

logger = logging.getLogger("mylogger")
logger.setLevel(logging.DEBUG)

# training
def train(rank, a, c):

    # create namespace of args and configs
    a = dict_to_namespace(a)
    c = dict_to_namespace(c)
    
    # logger
    if rank == 0:
        fh = logging.FileHandler(os.path.join(a.save_path, "log.txt"), mode="a")
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
        
    # seed
    seed = c.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    seed_this_gpu = c.seed + rank
    torch.cuda.manual_seed(seed_this_gpu)

    # create paths
    os.makedirs(a.save_path, exist_ok=True)
    a.ckpt_save_path = os.path.join(a.save_path, "ckpt")
    os.makedirs(a.ckpt_save_path, exist_ok=True)

    # DDP initialization
    init_process_group(
        backend=c.dist_config.dist_backend,
        init_method=c.dist_config.dist_url,
        world_size=c.dist_config.world_size * c.num_gpus,
        rank=rank,
    )
    
        # model and optimizer initialization
    device = torch.device("cuda:{:d}".format(rank))
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()
    model = MODEL(**vars(c.model)).to(device)  # to device
    optim = set_optimizer(model.parameters(), c.optimizer)
    scheduler = set_scheduler(optim, c.lr_scheduler)
    state_dict = None
    steps = 0
    last_epoch = -1
    ckpt = scan_checkpoint(a.ckpt_save_path, "ckpt_")
    if a.start_ckpt:
        if rank == 0:
            logger.info("Loading pretrained model from {}".format(a.start_ckpt))
        state_dict = load_checkpoint(a.start_ckpt, device)
        model.load_state_dict(state_dict["params"])
    else:
        if ckpt:
            if rank == 0:
                logger.info(
                    "Resume training process. Loading checkpoint from {}".format(ckpt)
                )
            state_dict = load_checkpoint(ckpt, device)
            model.load_state_dict(state_dict["params"])
            steps = state_dict["steps"] + 1
            last_epoch = state_dict["epoch"]
            # optim.load_state_dict(state_dict["optim"])
            # scheduler.load_state_dict(state_dict["scheduler"])
        else:
            if rank == 0:
                logger.info("New training process. Saved to {}".format(a.save_path))
                
    if rank == 0:
        logger.info(summary(model))
    if c.num_gpus > 1:  # DDP model
        model = DistributedDataParallel(
            model,
            device_ids=[rank],
            find_unused_parameters=False,
            broadcast_buffers=False,
        )

    # dataloader
    trainset = dataset(**vars(c.data_setting_train), **vars(c.stft_setting))
    train_sampler = (
        DistributedSampler(trainset) if c.num_gpus > 1 else RandomSampler(trainset)
    )
    train_loader = DataLoader(
        trainset,
        num_workers=c.num_workers,
        shuffle=False,
        sampler=train_sampler,
        batch_size=c.batch_size[0],
        pin_memory=True,
        drop_last=True,
        prefetch_factor=4,
        persistent_workers=True,
    )
    if rank == 0:
        validset = dataset(**vars(c.data_setting_val), **vars(c.stft_setting))
        validation_loader = DataLoader(
            validset,
            num_workers=c.num_workers,
            shuffle=False,
            sampler=None,
            batch_size=c.batch_size[1],
            pin_memory=True,
            drop_last=True,
            prefetch_factor=4,
            persistent_workers=True,
        )
        sw = SummaryWriter(os.path.join(a.save_path, "logs"))
        
    # training process
    model.train()
    if c.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    for epoch in range(max(0, last_epoch), a.training_epochs):
        if rank == 0:
            start = time.time()
            logger.info("Epoch: {}".format(epoch + 1))
        if c.num_gpus > 1:
            train_sampler.set_epoch(epoch)
        for i, batch in enumerate(train_loader):
            if rank == 0:
                start_b = time.time()
            optim.zero_grad()
            audio_input, audio_clean = batch
            audio_input = audio_input.to(device)
            audio_clean = audio_clean.to(device)
            input, input_phase = data_preprocess(audio_input, **vars(c.stft_setting))
            target, _ = data_preprocess(audio_clean, **vars(c.stft_setting))
            if c.use_amp:
                with torch.cuda.amp.autocast():
                    output, z_mean, z_logvar, _ = model(input)
                    loss_IS = LossF().loss_ISD(output, target)
                    loss_KL = LossF().loss_KLD(z_mean, z_logvar)
                    scale_KL = LossF().cal_KL_scale(
                        steps,
                        c.beta,
                        c.beta_zero_step,
                        c.beta_warmup_step,
                        c.beta_holdon_step,
                    )
                    loss_tot = loss_IS + loss_KL * scale_KL
                    assert torch.isnan(loss_tot) == False
                    scaler.scale(loss_tot).backward()
                    if c.gradient_clip:
                        torch.nn.utils.clip_grad.clip_grad_norm(
                            model.parameters(), max_norm=c.gradient_clip
                        )
                    scaler.step(optim)
                    scaler.update()
            else:
                output, z_mean, z_logvar, _ = model(input)
                loss_IS = LossF().loss_ISD(output, target)
                loss_KL = LossF().loss_KLD(z_mean, z_logvar)
                scale_KL = LossF().cal_KL_scale(
                    steps,
                    c.beta,
                    c.beta_zero_step,
                    c.beta_warmup_step,
                    c.beta_holdon_step,
                )
                loss_tot = loss_IS + loss_KL * scale_KL
                assert torch.isnan(loss_tot) == False
                loss_tot.backward()
                if c.gradient_clip:
                    torch.nn.utils.clip_grad.clip_grad_norm(
                        model.parameters(), max_norm=c.gradient_clip
                    )
                optim.step()
            if rank == 0:
                # stdout logger
                if steps % a.stdout_interval == 0:
                    logger.info(
                        "Steps:{:d}, Loss:{:.1f}, Loss_IS:{:.1f}, Loss_KL:{:.1f}, KL_scale:{:.1f}, lr:{:.2e}, time_cost:{:.1f}".format(
                            steps,
                            loss_tot,
                            loss_IS,
                            loss_KL,
                            scale_KL,
                            optim.param_groups[0]["lr"],
                            time.time() - start_b,
                        )
                    )

                # checkpointing
                if steps % a.checkpoint_interval == 0 and steps != 0:
                    checkpoint_filename = "ckpt_{:010d}".format(steps)
                    checkpoint_path = os.path.join(
                        a.ckpt_save_path, checkpoint_filename
                    )
                    save_checkpoint(
                        checkpoint_path,
                        {
                            "params": (
                                model.module if c.num_gpus > 1 else model
                            ).state_dict(),
                            "optim": optim.state_dict(),
                            "scheduler": scheduler.state_dict() if scheduler else None,
                            "steps": steps,
                            "epoch": epoch,
                        },
                    )

                # tensorboard summary logging
                if steps % a.summary_interval == 0:
                    sw.add_scalar("training/loss", loss_tot, steps)
                    sw.add_scalar("training/loss_IS", loss_IS, steps)
                    sw.add_scalar("training/loss_KL", loss_KL, steps)
                    sw.add_scalar("epoch", epoch, steps)
                    sw.add_scalar("lr", optim.param_groups[0]["lr"], steps)

                # validation
                if steps % a.validation_interval == 0:  # and steps != 0:
                    logger.info("validating ...")
                    model.eval()
                    torch.cuda.empty_cache()
                    val_loss_tot_tot = 0
                    val_loss_IS_tot = 0
                    val_loss_KL_tot = 0
                    metric_wbpesq_tot = 0
                    metric_estoi_tot = 0
                    tot_bs = 0
                    with torch.no_grad():
                        for j, batch in enumerate(validation_loader):
                            audio_input, audio_clean = batch
                            audio_input = audio_input.to(device)
                            audio_clean = audio_clean.to(device)
                            input, input_phase = data_preprocess(
                                audio_input, **vars(c.stft_setting)
                            )
                            target, _ = data_preprocess(
                                audio_clean, **vars(c.stft_setting)
                            )
                            bs = input.shape[0]

                            output, z_mean, z_logvar, _ = model(input)
                            loss_IS = LossF().loss_ISD(output, target)
                            loss_KL = LossF().loss_KLD(z_mean, z_logvar)
                            loss_tot = loss_IS + loss_KL

                            output_wavs = data_postprocess(
                                output, input_phase, **vars(c.stft_setting)
                            )

                            val_loss_tot_tot += loss_tot.item() * bs
                            val_loss_IS_tot += loss_IS.item() * bs
                            val_loss_KL_tot += loss_KL.item() * bs

                            tot_bs += bs

                            if j == 0 and rank == 0:
                                metric_wbpesq_tot = metric_wbpesq_tot + pesq(
                                    c.stft_setting.fs,
                                    audio_clean[0].cpu().numpy(),
                                    output_wavs[0].cpu().numpy(),
                                    "wb",
                                )
                                metric_estoi_tot = metric_estoi_tot + stoi(
                                    audio_clean[0].cpu().numpy(),
                                    output_wavs[0].cpu().numpy(),
                                    c.stft_setting.fs,
                                    extended=True,
                                )
                                spectrogram = (
                                    output[0].permute(1, 0).log().cpu().numpy()
                                )
                                sw.add_figure(
                                    "generated/y_hat_spec_{}".format(steps),
                                    plot_spectrogram(spectrogram),
                                    steps,
                                )
                                if steps == 0:
                                    spectrogram = (
                                        input[0].permute(1, 0).log().cpu().numpy()
                                    )
                                    sw.add_figure(
                                        "clean/y_spec_{}".format(steps),
                                        plot_spectrogram(spectrogram),
                                        steps,
                                    )

                        val_loss_tot = val_loss_tot_tot / tot_bs
                        val_loss_IS = val_loss_IS_tot / tot_bs
                        val_loss_KL = val_loss_KL_tot / tot_bs
                        metric_wbpesq = metric_wbpesq_tot
                        metric_estoi = metric_estoi_tot

                        sw.add_scalar("validation/loss", val_loss_tot, steps)
                        sw.add_scalar("validation/loss_IS", val_loss_IS, steps)
                        sw.add_scalar("validation/loss_KL", val_loss_KL, steps)
                        sw.add_scalar(
                            "validation/metric_wbpesq_1sample", metric_wbpesq, steps
                        )
                        sw.add_scalar(
                            "validation/metric_estoi_1sample", metric_estoi, steps
                        )

                    model.train()

            steps += 1
            
        if scheduler:
            scheduler.step()

        if rank == 0:
            print(
                "Time taken for epoch {} is {} sec\n".format(
                    epoch + 1, int(time.time() - start)
                )
            )

def main():

    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d-%H-%M")

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config", "-c", action="append", default=[], help="config files (.json)"
    )
    parser.add_argument(
        "--save_path", "-p", default="saved_model" + current_time, help="saved path"
    )
    parser.add_argument(
        "--training_epochs", default=100000, type=int, help="maximum epochs"
    )
    parser.add_argument(
        "--stdout_interval", default=10, type=int, help="stdout interval (steps)"
    )
    parser.add_argument(
        "--checkpoint_interval", default=1000, type=int, help="save interval (steps)"
    )
    parser.add_argument(
        "--summary_interval",
        default=100,
        type=int,
        help="tensorboard summary interval (steps)",
    )
    parser.add_argument(
        "--validation_interval",
        default=1000,
        type=int,
        help="validation interval (steps)",
    )
    parser.add_argument(
        "--start_ckpt", "-r", type=str, required=False, help="pretrained model (.ckpt)"
    )

    args = parser.parse_args()

    json_config = {}
    for filename in args.config:
        with open(filename, "r") as f:  # 读取config
            json_config.update(json.load(f))

    config = dict_to_namespace(json_config)  # 转换为属性字典
    build_env(
        config, "config.json", args.save_path
    )  # 将a.config复制到a.checkpoint_path/config.json

    logger.info("Initializing Unsupervised Training Process..")

    wav_path2txt(config.data_setting_train.wav_clean_root)
    wav_path2txt(os.path.join(config.data_setting_train.rir_root, "target"))
    wav_path2txt(os.path.join(config.data_setting_train.rir_root, "noisy"))
    wav_path2txt(config.data_setting_val.wav_clean_root)
    wav_path2txt(os.path.join(config.data_setting_val.rir_root, "target"))
    wav_path2txt(os.path.join(config.data_setting_val.rir_root, "noisy"))

    config.data_setting_train.spch_index_txt = os.path.join(
        config.data_setting_train.wav_clean_root, "index.txt"
    )
    config.data_setting_train.rir_index_txt = os.path.join(
        config.data_setting_train.rir_root, "noisy", "index.txt"
    )
    config.data_setting_train.rir_target_index_txt = os.path.join(
        config.data_setting_train.rir_root, "target", "index.txt"
    )
    config.data_setting_val.spch_index_txt = os.path.join(
        config.data_setting_val.wav_clean_root, "index.txt"
    )
    config.data_setting_val.rir_index_txt = os.path.join(
        config.data_setting_val.rir_root, "noisy", "index.txt"
    )
    config.data_setting_val.rir_target_index_txt = os.path.join(
        config.data_setting_val.rir_root, "target", "index.txt"
    )

    if torch.cuda.is_available():
        config.num_gpus = torch.cuda.device_count()
        config.batch_size[0] = int(config.batch_size[0] / config.num_gpus)
        logger.info("Batch size per GPU :" + str(config.batch_size))
    else:
        raise ValueError("No GPU Devices!")

    if config.num_gpus > 1:
        mp.spawn(
            train,
            nprocs=config.num_gpus,
            args=(
                vars(args),
                vars(config),
            ),
        )  # 多线程进行训练，每个GPU一个进程
    else:
        train(0, vars(args), vars(config))


if __name__ == "__main__":
    main()
        
        
        
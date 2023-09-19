import os
import argparse
import json
import torch
import time
import warnings
import numpy as np
import random
from glob import glob
from math import ceil
import logging
from torchaudio import save
from torchinfo import summary
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from datetime import datetime
from utils.env import build_env, dict_to_namespace
from utils.utils import (
    load_checkpoint,
)
from model.RVAE import RVAE as MODEL
from dataset.testdataset import TestDataset as dataset
from dataset.io import audio2EMinput_woDC_preprocess as data_preprocess
from dataset.io import EMoutput_woDC2audio_postprocess as data_postprocess
from model.my_EM import MyEM

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.simplefilter(action="ignore", category=UserWarning)

logger = logging.getLogger("mylogger")
logger.setLevel(logging.DEBUG)


def inference(rank, a, c):

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
    a.enhanced_path = a.save_path
    os.makedirs(a.enhanced_path, exist_ok=True)

    # DDP initialization
    init_process_group(
        backend=c.dist_config.dist_backend,
        init_method=c.dist_config.dist_url,
        world_size=c.dist_config.world_size * c.num_gpus,
        rank=rank,
    )

    # model initialization
    device = torch.device("cuda:{:d}".format(rank))
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()
    model = MODEL(**vars(c.model)).to(device)

    already_exist_wavs_filename = []
    for filepath_abs in glob(os.path.join(a.enhanced_path, "*")):
        if filepath_abs.endswith("wav"):
            already_exist_wavs_filename.append(os.path.basename(filepath_abs))

    logger.info("loading checkpoint from ", c.ckpt)
    state_dict = load_checkpoint(c.ckpt, device)
    model.load_state_dict(state_dict["params"])
    for name, param in model.named_parameters():
        param.requires_grad = False

    if rank == 0:
        logger.info(summary(model))

    # dataloader
    testset = dataset(**vars(c.data_setting_test), **vars(c.stft_setting))
    test_sampler = DistributedSampler(testset) if c.num_gpus > 1 else None
    test_loader = DataLoader(
        testset,
        num_workers=c.num_workers,
        shuffle=False,
        sampler=test_sampler,
        batch_size=c.batch_size[2],
        pin_memory=False,
        drop_last=False,
    )

    model.eval()
    algo = MyEM(model, vars(c.EM_kwargs))
    del model
    torch.cuda.empty_cache()
    start_time = time.time()

    # Inference
    with torch.cuda.amp.autocast():
        for i, batch in enumerate(test_loader):
            audio_input, filename = batch
            bs = audio_input.shape[0]
            if True:
                audio_input = audio_input.to(device)
                seq_len = audio_input.shape[1]
                seq_len_input = ceil(seq_len / c.stft_setting.hop) * c.stft_setting.hop
                audio_input_pad = torch.zeros([bs, seq_len_input]).to(device)
                audio_input_pad[:, :seq_len] = audio_input
                input = data_preprocess(audio_input_pad, **vars(c.stft_setting))
                output_sptm = algo.EM(input)
                output_wav = data_postprocess(output_sptm, **vars(c.stft_setting)).to(
                    "cpu"
                )
                output_wav = output_wav[:, :seq_len]
                for isample in range(bs):
                    save_path = os.path.join(a.enhanced_path, filename[isample])
                    save(
                        save_path,
                        (
                            output_wav[isample] / output_wav[isample].abs().max()
                        ).unsqueeze(0),
                        c.stft_setting.fs,
                    )
            if rank == 0:
                print(
                    "\r",
                    int(i + 1) / test_loader.__len__() * 100,
                    "%",
                    int((time.time() - start_time) / 60),
                    "min",
                    end="",
                    flush=True,
                )
    logger.info(
        "Total inference time: {:.1f} minutes".format((time.time() - start_time) / 60)
    )


def main():
    logger.info("Initializing Testing Process..")

    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d-%H-%M")

    parser = argparse.ArgumentParser()

    parser.add_argument("--config", "-c", action="append", default=[])  # config
    parser.add_argument("--ckpt", required=True)  # checkpoint
    parser.add_argument(
        "--save_path", "-p", default="inferenced_model" + current_time
    )  # output folder

    args = parser.parse_args()

    json_config = {}
    for filename in args.config:
        with open(filename, "r") as f:  # 读取config
            json_config.update(json.load(f))

    config = dict_to_namespace(json_config)  # 转换为属性字典
    config.ckpt = args.ckpt
    build_env(config, "config.json", args.save_path)

    if torch.cuda.is_available():
        config.num_gpus = torch.cuda.device_count()
        config.batch_size[2] = int(config.batch_size[2] / config.num_gpus)
        logger.info("Batch size per GPU :", config.batch_size)
    else:
        raise ValueError("No GPU Devices!")

    if config.num_gpus > 1:
        mp.spawn(
            inference,
            nprocs=config.num_gpus,
            args=(
                vars(args),
                vars(config),
            ),
        )
    else:
        inference(0, vars(args), vars(config))


if __name__ == "__main__":
    main()

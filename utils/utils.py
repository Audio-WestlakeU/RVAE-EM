import os
from glob import glob
import torch
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import yaml
import sys


def scan_checkpoint(cp_dir, prefix=""):
    pattern = os.path.join(cp_dir, prefix + "??????????")
    cp_list = glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    checkpoint_dict = torch.load(filepath, map_location="cpu")
    return checkpoint_dict


def set_optimizer(params, config_optim):

    """
    设置优化器
    :params
    params 优化的目标
    config_optim: 配置
    :ret
    optimizer
    """

    if config_optim.type.upper() == "SGD":
        optim = torch.optim.SGD(params, **vars(config_optim.kwargs))
    elif config_optim.type.upper() == "ADAM":
        optim = torch.optim.Adam(params, **vars(config_optim.kwargs))
    elif config_optim.type.upper() == "ADAMW":
        optim = torch.optim.AdamW(params, **vars(config_optim.kwargs))
    else:
        raise ValueError("Wrong Optimizer Type")
    return optim


def set_scheduler(optim, config_lrscheduler):
    if config_lrscheduler.type.upper() == "ReduceLROnPlateau".upper():
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, **vars(config_lrscheduler.kwargs)
        )
    elif config_lrscheduler.type.upper() == "none".upper():
        scheduler = None
    elif config_lrscheduler.type.upper() == "ExponentialLR".upper():
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optim, **vars(config_lrscheduler.kwargs)
        )
    elif config_lrscheduler.type.upper() == "CyclicLR".upper():
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optim, **vars(config_lrscheduler.kwargs), cycle_momentum=False
        )
    elif config_lrscheduler.type.upper() == "StepLR".upper():
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, **vars(config_lrscheduler.kwargs)
        )
    else:
        raise ValueError("Wrong Scheduler Type")
    return scheduler


def get_filelist_from_txt(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        files = [x for x in f.read().split("\n") if len(x) > 0]
    return files


def save_checkpoint(filepath, obj):
    print("Saving checkpoint to {}".format(filepath))
    torch.save(obj, filepath)


def plot_spectrogram(spectrogram):
    spectrogram[spectrogram < -8] = -8
    fig = plt.figure()
    plt.imshow(spectrogram, aspect="auto", origin="lower")
    plt.colorbar()
    plt.close()

    return fig


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def add_DC_and_save_wav(
    spectrogram_no_DC, phase_no_DC, stft_config, dir_path, filename
):
    spectrogram_no_DC = spectrogram_no_DC.detach()
    phase_no_DC = phase_no_DC.detach()
    device = spectrogram_no_DC.device
    bs, F, T = spectrogram_no_DC.shape
    F = F + 1
    spectrogram = torch.zeros([bs, F, T]).to(device) + 0j
    spectrogram[:, 1:, :] = spectrogram_no_DC
    spectrogram = spectrogram.sqrt()
    spectrogram = np.array(spectrogram.to("cpu"))
    phase = torch.zeros([bs, F, T]).to(device)
    phase[:, 1:, :] = phase_no_DC
    phase = torch.exp(1j * phase)
    phase = np.array(phase.to("cpu"))
    spectrogram = spectrogram * phase

    for isample in range(bs):
        wav = librosa.istft(
            spectrogram[isample],
            hop_length=stft_config.hop,
            win_length=stft_config.nfft,
            n_fft=stft_config.nfft,
            window=stft_config.win,
        )
        wav = wav / np.max(np.abs(wav))
        # wav = np.expand_dims(wav,0)
        filepath = os.path.join(dir_path, filename[isample])
        sf.write(filepath, wav, 16000)

    return None


def del_files_from_dir(filepath):
    del_list = os.listdir(filepath)
    for f in del_list:
        file_path = os.path.join(filepath, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
    return None


def read_yaml(config_path):
    # 读取yaml文件
    with open(config_path, encoding="utf-8") as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    return config


def wav_path2txt(root):
    """
    Function: save the path of .wavs in the root folder to index.txt
    Params:
        root: root folder
    """
    # 获取文件夹中wav文件的地址存入txt文件
    files = sorted(glob("{}/**/*.wav".format(root), recursive=True))
    f = open(os.path.join(root, "index.txt"), "w")
    for filename in files:
        f.write(filename + os.linesep)
    f.close()
    return None


class TxtLogger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

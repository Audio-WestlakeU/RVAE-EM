import torch
from torch.utils.data import Dataset
import random
import scipy.signal
import os
import numpy as np
import soundfile as sf


class MyDataset(Dataset):
    def __init__(
        self,
        spch_index_txt: str,
        rir_index_txt: str,
        rir_target_index_txt: str,
        fs: int = 16000,
        spch_len: float = 5.104,
        norm_mode: str = "noisy_time",
        **kwargs
    ):
        """
        spch_path: 存放干净语音文件名的txt文件地址
        rir_path: 存放RIR文件名的txt文件地址
        fs: 采样率Hz
        spch_len: 音频持续时间s
        norm_mode: 归一化方式
        """
        self.fs = fs
        self.spch_len = spch_len

        assert (
            os.path.splitext(spch_index_txt)[-1] == ".txt"
            and os.path.splitext(rir_index_txt)[-1] == ".txt"
            and os.path.splitext(rir_target_index_txt)[-1] == ".txt"
        )

        # 读取clean speech列表
        spch_list = []
        spch_f = open(spch_index_txt, "r")
        for i in spch_f.readlines():
            spch_list.append(i.rstrip())

        # 读取rir列表
        rir_list = []
        rir_f = open(rir_index_txt, "r")
        for i in rir_f.readlines():
            rir_list.append(i.rstrip())

        rir_target_list = []
        rir_target_f = open(rir_target_index_txt, "r")
        for i in rir_target_f.readlines():
            rir_target_list.append(i.rstrip())

        # 实际使用的音频与RIR列表
        self.rir_list = rir_list
        self.rir_target_list = rir_target_list
        self.valid_seq_list = self._get_valid_speech_list(
            spch_list,
        )

        self.norm_mode = norm_mode

    def _get_valid_speech_list(
        self,
        spch_filename_list,
    ):
        valid_seq_list = []  # 可用语音List
        for wavfile in spch_filename_list:
            # 读取音频
            x, fs_x = sf.read(wavfile, dtype="float32")
            x = x.squeeze()
            # 判断采样频率是否正确
            assert self.fs == fs_x
            # 计算音频开始、结束位置
            idx_beg = 0
            idx_end = len(x)
            # 音频切分
            file_len = idx_end - idx_beg  # 非沉默音频段长度
            samples = int(self.fs * self.spch_len)
            # seq_len = (seq_len - 1) * hop
            n_seq = int(file_len // samples)
            for i in range(n_seq):
                seq_start = int(i * samples + idx_beg)
                seq_end = int((i + 1) * samples + idx_beg)
                seq_info = (wavfile, seq_start, seq_end)
                valid_seq_list.append(seq_info)

        return valid_seq_list

    def __len__(self):

        return len(self.valid_seq_list)

    def __getitem__(self, index):
        """
        读取音频
        """
        rir_index_list = torch.randint(
            0, len(self.rir_list), (len(self.valid_seq_list),)
        )

        rir_list = [self.rir_list[i] for i in rir_index_list]
        rir_target_list = [self.rir_target_list[i] for i in rir_index_list]

        ##################################################################
        # 读取clean speech
        spch_filename, seq_start, seq_end = self.valid_seq_list[index]
        spch_ori, fs_spch = sf.read(spch_filename, dtype="float32")
        # 检查采样率
        assert self.fs == fs_spch
        spch_ori = spch_ori.squeeze()

        shift_max = min(seq_start - 0, spch_ori.shape[0] - seq_end, self.fs)
        shift = random.randint(-shift_max, shift_max)

        spch_ori = spch_ori[int(seq_start + shift) : int(seq_end + shift)]

        ##################################################################
        samples = int(self.fs * self.spch_len)
        # 加入混响
        # 读取RIR
        rir, fs_rir = sf.read(rir_list[index], dtype="float32")
        assert self.fs == fs_rir
        rir = rir.squeeze()
        spch_noisy = scipy.signal.fftconvolve(spch_ori, rir, mode="full")
        spch_noisy = spch_noisy[:samples]

        rir_target, fs_rir_target = sf.read(rir_target_list[index], dtype="float32")
        assert self.fs == fs_rir_target
        rir_target = rir_target.squeeze()
        spch_dry = scipy.signal.fftconvolve(spch_ori, rir_target, mode="full")
        spch_dry = spch_dry[:samples]

        # 归一化
        if np.max(np.abs(spch_noisy)) > 0:
            scale = np.max(np.abs(spch_noisy))
            spch_noisy = spch_noisy / scale
            spch_dry = spch_dry / scale
        if self.norm_mode.upper() == "TIME":
            spch_dry = spch_dry / np.max(np.abs(spch_dry))

        return torch.Tensor(spch_noisy), torch.Tensor(spch_dry)

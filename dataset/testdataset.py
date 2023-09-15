import torch
from torch.utils.data import Dataset
import random
import scipy.signal
import os
import numpy as np
import soundfile as sf
import glob


class TestDataset(Dataset):
    """
    WSJ0和BUT数据集组合
    """

    def __init__(
        self,
        spch_dir: str,
        fs: int = 16000,
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
        assert os.path.exists(spch_dir)

        self.wav_files = glob.glob(os.path.join(spch_dir, "*.wav"))


    def __len__(self):

        return len(self.wav_files)

    def __getitem__(self, index):
        """
        读取音频
        """
        
        input, fs_spch = sf.read(self.wav_files[index],dtype='float32')
        assert self.fs == fs_spch
        
        filename = os.path.basename(self.wav_files[index])
        if np.max(np.abs(input)) > 0:
            spch_scale = np.max(np.abs(input))
            input = input / spch_scale * 0.95

        return input, filename


if __name__ == "__main__":

    pass
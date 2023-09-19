from torch.utils.data import Dataset
import os
import numpy as np
import soundfile as sf
import glob


class TestDataset(Dataset):
    def __init__(self, spch_dir: str, fs: int = 16000, **kwargs):
        """
        Class: testset
        Params:
            spch_dir: Reverberant speech folder path
            fs: sample frequency
        """
        self.fs = fs
        assert os.path.exists(spch_dir)

        self.wav_files = glob.glob(os.path.join(spch_dir, "*.wav"))

    def __len__(self):

        return len(self.wav_files)

    def __getitem__(self, index):

        input, fs_spch = sf.read(self.wav_files[index], dtype="float32")
        assert self.fs == fs_spch

        filename = os.path.basename(self.wav_files[index])
        if np.max(np.abs(input)) > 0:
            spch_scale = np.max(np.abs(input))
            input = input / spch_scale * 0.95

        return input, filename


if __name__ == "__main__":

    pass

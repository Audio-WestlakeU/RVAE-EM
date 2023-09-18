from os.path import join
from glob import glob
from soundfile import read
from tqdm import tqdm
from pesq import pesq
import pandas as pd
import argparse
import torch
import numpy as np
import datetime
from pystoi import stoi
from utils.other import mean_std
from srmrpy import srmr
from torchmetrics.functional.audio import (
    scale_invariant_signal_distortion_ratio as sisdr,
)


def eval_metrics(a):

    output_root = a.out_path
    input_root = a.in_path
    ref_root = a.ref_path
    sr = a.sr
    device = "cpu"

    data = {
        "filename": [],
        "wbpesq": [],
        "nbpesq": [],
        "stoi": [],
        "estoi": [],
        "sisdr": [],
        "srmr": [],
    }

    # Evaluate standard metrics
    noisy_files = sorted(glob("{}/*.wav".format(input_root)))

    for noisy_file in tqdm(noisy_files, desc="Calculating metrics"):
        filename = noisy_file.split("/")[-1]
        x, sr_read = read(join(ref_root, filename))
        assert sr == sr_read
        x = x / np.max(x)
        x_method, _ = read(join(output_root, filename))
        x_method = x_method / np.max(x_method)

        data["filename"].append(filename)
        data["wbpesq"].append(pesq(sr, x, x_method, "wb"))
        data["nbpesq"].append(pesq(sr, x, x_method, "nb"))
        data["stoi"].append(stoi(x[: x_method.shape[-1]], x_method, sr, extended=False))
        data["estoi"].append(stoi(x[: x_method.shape[-1]], x_method, sr, extended=True))
        data["sisdr"].append(
            sisdr(torch.Tensor(x_method), torch.Tensor(x[: x_method.shape[-1]])).item()
        )
        data["srmr"].append(srmr(x_method, sr)[0])

    # Save results as DataFrame
    df = pd.DataFrame(data)

    # Print results
    with open(join(output_root, "ave_metrics.txt"), "w") as f:
        print(datetime.datetime.now(), file=f)
        print(output_root, file=f)
        print(
            "SISDR: {:.2f} ± {:.2f}".format(*mean_std(df["sisdr"].to_numpy())), file=f
        )
        print(
            "WBPESQ: {:.2f} ± {:.2f}".format(*mean_std(df["wbpesq"].to_numpy())), file=f
        )
        print(
            "NBPESQ: {:.2f} ± {:.2f}".format(*mean_std(df["nbpesq"].to_numpy())), file=f
        )
        print("STOI: {:.2f} ± {:.2f}".format(*mean_std(df["stoi"].to_numpy())), file=f)
        print(
            "ESTOI: {:.2f} ± {:.2f}".format(*mean_std(df["estoi"].to_numpy())), file=f
        )
        print("SRMR: {:.2f} ± {:.2f}".format(*mean_std(df["srmr"].to_numpy())), file=f)

    # Save DataFrame as csv file
    df.to_csv(join(output_root, "_results.csv"), index=False)


def main():
    print("Calculating the metrics of wavs..")

    parser = argparse.ArgumentParser()
    parser.add_argument("--out_path", "-o", required=True)  # output .wav folder
    parser.add_argument("--in_path", "-i", required=True)  # input .wav folder
    parser.add_argument("--ref_path", "-r", required=True)  # reference .wav folder
    parser.add_argument("--sr", required=False)  # sample rate

    args = parser.parse_args()

    eval_metrics(args)


if __name__ == "__main__":
    main()

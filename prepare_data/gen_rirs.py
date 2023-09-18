import os
import random
import gpuRIR
import numpy as np
import soundfile as sf
import argparse
from tqdm import tqdm
import shutil
from utils import read_yaml


def gen_RIR(room_sz, T60, pos_src, pos_rcv, fs):

    beta = gpuRIR.beta_SabineEstimation(room_sz, T60)
    num_images = gpuRIR.t2n(2, room_sz)
    RIR = gpuRIR.simulateRIR(
        room_sz, beta, pos_src, pos_rcv, num_images, 2, fs
    ).squeeze()

    RIR_target = gpuRIR.simulateRIR(
        room_sz, [0.01 for i in range(6)], pos_src, pos_rcv, num_images, 2, fs
    ).squeeze()

    return RIR, RIR_target


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Gen RIRs")

    parser.add_argument("-c", "--config", type=str, default="config/data/gen_data.yaml")
    args = parser.parse_args()

    config = read_yaml(args.config)

    RIR_save_root = config["RIR_save_root"]
    T60_range = config["T60_range"]
    room_sz_range = config["room_sz_range"]
    min_distance_to_wall = config["min_distance_to_wall"]
    num_RIRs = config["num_RIRs"]
    fs = config["sample_rate"]

    if config["random_seed"]:
        print("random seed: ", config["random_seed"])
        random.seed(config["random_seed"])

    RIR_save_root_noisy = os.path.join(RIR_save_root, "noisy")
    RIR_save_root_target = os.path.join(RIR_save_root, "target")

    os.makedirs(RIR_save_root, exist_ok=True)
    shutil.rmtree(RIR_save_root)
    os.makedirs(RIR_save_root_noisy, exist_ok=True)
    os.makedirs(RIR_save_root_target, exist_ok=True)

    RIRs_noisy = np.zeros((num_RIRs, 2 * fs))
    RIRs_target = np.zeros((num_RIRs, 2 * fs))

    T60s_noisy = np.zeros(num_RIRs)
    for i in tqdm(range(num_RIRs), desc="Generating"):

        room_sz = np.array(
            [
                np.random.uniform(
                    room_sz_range[0],
                    room_sz_range[1],
                ),
                np.random.uniform(
                    room_sz_range[2],
                    room_sz_range[3],
                ),
                np.random.uniform(
                    room_sz_range[4],
                    room_sz_range[5],
                ),
            ],
        )
        pos_src = np.array(
            [
                [
                    np.random.uniform(
                        min_distance_to_wall, room_sz[0] - min_distance_to_wall
                    ),
                    np.random.uniform(
                        min_distance_to_wall, room_sz[1] - min_distance_to_wall
                    ),
                    np.random.uniform(
                        min_distance_to_wall, room_sz[2] - min_distance_to_wall
                    ),
                ]
            ]
        )

        pos_rcv = np.array(
            [
                [
                    np.random.uniform(
                        min_distance_to_wall, room_sz[0] - min_distance_to_wall
                    ),
                    np.random.uniform(
                        min_distance_to_wall, room_sz[1] - min_distance_to_wall
                    ),
                    np.random.uniform(
                        min_distance_to_wall, room_sz[2] - min_distance_to_wall
                    ),
                ]
            ]
        )
        T60 = np.random.uniform(T60_range[0], T60_range[1])
        T60s_noisy[i] = T60
        RIRs_noisy[i], RIRs_target[i] = gen_RIR(room_sz, T60, pos_src, pos_rcv, fs)

        filename = "{0}_T60_{1:.2f}.wav".format(i, T60s_noisy[i])
        savepath_noisy = os.path.abspath(os.path.join(RIR_save_root_noisy, filename))
        savepath_target = os.path.abspath(os.path.join(RIR_save_root_target, filename))
        sf.write(savepath_noisy, RIRs_noisy[i], fs)
        sf.write(savepath_target, RIRs_target[i], fs)

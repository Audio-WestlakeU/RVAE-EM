import yaml
from glob import glob
import os


def read_yaml(config_path):
    # 读取yaml文件
    with open(config_path, encoding="utf-8") as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    return config


def wav_path2txt(root):
    # 获取wav文件的地址存入txt文件
    files = sorted(glob("{}/**/*.wav".format(root), recursive=True))
    f = open(os.path.join(root, "index.txt"), "w")
    for filename in files:
        f.write(filename + os.linesep)
    f.close()
    return None

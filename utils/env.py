import os
import json
import types


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def dict_to_namespace(d):
    """
    Return: namespace
    Params: dict
    """
    # 如果值是字典，则递归地将其转换为属性字典
    for key, value in d.items():
        if isinstance(value, dict):
            d[key] = dict_to_namespace(value)

    # 将字典转换为属性字典并返回
    return types.SimpleNamespace(**d)


def build_env(config, config_name, path):  # 将配置文件复制到对应目录

    """
    Copy config.json to the folder
    Params:
    config 要保存的字典
    config_name 要保存的文件名
    path 要保存的文件夹
    """
    t_path = os.path.join(path, config_name)
    print("project directory : ", path)
    os.makedirs(path, exist_ok=True)

    json_str = json.dumps(config, cls=NamespaceEncoder)

    with open(t_path, "w") as f:
        f.write(json_str)
    f.close()


class NamespaceEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, types.SimpleNamespace):
            return vars(obj)
        return super().default(obj)

from copy import deepcopy as dpcp
import yaml

from adet.config import get_cfg
from fvcore.common.config import CfgNode


__all__ = [
    "cfg_force_merge", "cfg_to_dict", "cfg_from_yaml"
]


def cfg_force_merge(*cfgs) -> CfgNode:
    cfg_base = cfgs[0].clone()

    for cfg in cfgs[1:]:
        cfg_base = _cfg_merge_regresser(cfg_base, cfg)
    
    return cfg_base


def cfg_to_dict(cfg):
    cfg = dict(dpcp(cfg))
    for key in cfg.keys():
        if isinstance(cfg[key], dict):
            cfg[key] = dict(cfg_to_dict(cfg[key]))
    return cfg


def _cfg_merge_regresser(cfg1, cfg2):
    if not isinstance(cfg1, dict) or not isinstance(cfg2, dict):
        return dpcp(cfg2)
    else:
        cfg = dpcp(cfg1)
        for key in cfg2:
            if (key not in cfg.keys()) or not isinstance(cfg2[key], dict):
                cfg[key] = cfg2[key]
            else:
                cfg[key] = _cfg_merge_regresser(cfg[key], cfg2[key])
        
        return cfg


def cfg_from_yaml(file_path, do_merge_to_default=False):
    # return CfgNode(CfgNode.load_yaml_with_base(file_path))
    if file_path is not None:
        with open(file_path, "r") as f:
            s = f.read()
        var = yaml.safe_load(s)

        def _convert(var):
            for key in var.keys():
                if isinstance(var[key], dict):
                    var[key] = _convert(var[key])
                else:
                    if isinstance(var[key], str) and var[key][0] == "(":
                        var[key] = eval(var[key])
            return var
        
        var = CfgNode(_convert(var))

    else:
        var = {}

    if do_merge_to_default is True:
        cfg_default = get_cfg()
        var = cfg_force_merge(cfg_default, var)
    
    return var

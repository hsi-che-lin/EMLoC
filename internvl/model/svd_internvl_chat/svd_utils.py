import re
import torch
import logging

from internvl.model.svd_internvl_chat.svd_modules import SVDLinear
from internvl.utils.utils import rgetattr, rsetattr


logger = logging.getLogger(__name__)


def processCompressConfig(model, config):
    newConfig = {}
    allLinears = [
        n for (n, m) in model.named_modules() if (isinstance(m, torch.nn.Linear))
    ]

    assert all([((ratio > 0) and (ratio < 1)) for ratio in config.values()])

    for name in allLinears:
        for (pattern, ratio) in config.items():
            if (re.fullmatch(pattern, name)):
                newConfig[name] = ratio
                break

    return newConfig


def get_random_svd_model(model, config):
    config = processCompressConfig(model, config)

    for (name, ratio) in config.items():
        linear = rgetattr(model, name)
        svd = SVDLinear(ratio, refLinear = linear)
        rsetattr(model, name, svd)
    
    return model

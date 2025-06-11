import re
import functools
import os
import torch
import logging
import math
from tqdm import tqdm
from safetensors.torch import save_file, load_file

from .svd_modules import SVDLinear

logger = logging.getLogger(__name__)

# from https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties/31174427#31174427
def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


def rsetattr(obj, attr, val):
    if ("." in attr):
        attr2obj, _, attr2set = attr.rpartition('.')
        setattr(rgetattr(obj, attr2obj), attr2set, val)
    else:
        setattr(obj, attr, val)

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


def get_svd_model(model, config, saveDir, scaling = {}):
    config = processCompressConfig(model, config)

    allKeys = []
    statePaths = []
    state = {}
    paramNum = 0
    threshold = 8 * 1024 * 1024 * 1024
    
    for (name, ratio) in tqdm(config.items()):
        linear = rgetattr(model, name)
        svd = SVDLinear(ratio, refLinear = linear)
        s = scaling.get(name, None)
        svd.init_weights_from_ref(linear, s)
        paramNum += (linear.weight.numel() * ratio)

        for (k, v) in svd.state_dict().items():
            state[f"{name}.{k}"] = v
            allKeys.append(f"{name}.{k}")

        if (paramNum >= threshold):
            savePath = os.path.join(saveDir, f"{len(statePaths)}.safetensors")
            logger.info(f"Save current state to {savePath} to prevent out of memory")
            statePaths.append(savePath)
            save_file(state, savePath)
            paramNum = 0
            state = {}
        
        svd = svd.to(device = "meta")
        rsetattr(model, name, svd)
    
    for path in statePaths:
        state.update(load_file(path))
    
    result = model.load_state_dict(state, strict = False, assign = True)
    missing = [key for key in result.missing_keys if (key in allKeys)]
    logger.info(f"Load temp state back, missing keys: {missing}, unexpected keys: {result.unexpected_keys}")

    return model


def decompose_weights(model, saveDir, scaling = {}, maxRatio = 0.5):
    config = processCompressConfig(model, {".*": maxRatio})

    os.makedirs(saveDir, exist_ok = True)
    resultPaths = []
    results = {}
    paramNum = 0
    threshold = 8 * 1024 * 1024 * 1024
    
    for (i, (name, ratio)) in enumerate(tqdm(config.items())):
        linear = rgetattr(model, name)
        svd = SVDLinear(ratio, refLinear = linear)
        s = scaling.get(name, None)
        U, S, VT, _ = svd.get_decompose_result(linear, s)
        paramNum += (U.numel() + S.numel() + VT.numel())
        
        results[f"{name}-U"] = U.cpu().contiguous()
        results[f"{name}-S"] = S.cpu().contiguous()
        results[f"{name}-VT"] = VT.cpu().contiguous()
        if (linear.bias is not None):
            results[f"{name}-bias"] = linear.bias.data

        if ((paramNum >= threshold) or (i == (len(config) - 1))):
            savePath = os.path.join(saveDir, f"{len(resultPaths)}.safetensors")
            logger.info(f"Save current SVD result to {savePath}")
            resultPaths.append(savePath)
            save_file(results, savePath)
            paramNum = 0
            results = {}
        
        rsetattr(model, name, linear.to(device = "meta"))


def get_svd_model_from_decompose_results(model, config, decomposeResultDir, saveDir, scaling = {}):
    config = processCompressConfig(model, config)

    dtype  = model.config.torch_dtype
    allKeys = []
    statePaths = []
    state = {}
    dcpResult = {}
    paramNum = 0
    threshold = 8 * 1024 * 1024 * 1024

    for file in os.listdir(decomposeResultDir):
        dcpResult.update(load_file(os.path.join(decomposeResultDir, file)))
    
    for (name, ratio) in tqdm(config.items()):
        linear = rgetattr(model, name)
        svd = SVDLinear(ratio, refLinear = linear)
        U = dcpResult[f"{name}-U"]
        S = dcpResult[f"{name}-S"]
        VT = dcpResult[f"{name}-VT"]
        bias = dcpResult.get(f"{name}-bias", None)
        s = scaling.get(name, None)
        svd.init_weights_from_decompose_results(U, S, VT, bias, dtype, s)
        paramNum += (linear.weight.numel() * ratio)

        for (k, v) in svd.state_dict().items():
            state[f"{name}.{k}"] = v
            allKeys.append(f"{name}.{k}")

        if (paramNum >= threshold):
            savePath = os.path.join(saveDir, f"{len(statePaths)}.safetensors")
            logger.info(f"Save current state to {savePath} to prevent out of memory")
            statePaths.append(savePath)
            save_file(state, savePath)
            paramNum = 0
            state = {}
        
        svd = svd.to(device = "meta")
        rsetattr(model, name, svd)
    
    for path in statePaths:
        state.update(load_file(path))
    
    result = model.load_state_dict(state, strict = False, assign = True)
    missing = [key for key in result.missing_keys if (key in allKeys)]
    logger.info(f"Load temp state back, missing keys: {missing}, unexpected keys: {result.unexpected_keys}")

    return model
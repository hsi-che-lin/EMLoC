import os
import torch
from tqdm import tqdm
from safetensors.torch import load_file, save_file
from tempfile import TemporaryDirectory

from model.svd_flux.svd_flux import SVDFluxTransformer2DModel
from model.svd_flux.svd_utils import processCompressConfig, rgetattr, rsetattr
from model.svd_flux.svd_modules import SVDLinear


if (__name__ == "__main__"):
    model = SVDFluxTransformer2DModel.from_pretrained(
        "black-forest-labs/FLUX.1-dev", subfolder="transformer", revision=None, variant=None,
        torch_dtype = torch.bfloat16,
        low_cpu_mem_usage = True,
        skip_svd = True
    )

    pretrainDir = "pretrained/emulator"
    tmpDir = TemporaryDirectory()
    saveDir = tmpDir.name
    config = processCompressConfig(model, {".*transformer_blocks.*": 0.45})
    
    allKeys = []
    statePaths = []
    state = {}
    paramNum = 0
    threshold = 8 * 1024 * 1024 * 1024
    
    for (name, ratio) in tqdm(config.items()):
        linear = rgetattr(model, name)
        svd = SVDLinear(ratio, refLinear = linear)
        svd.init_weights_from_ref(linear)
        paramNum += (linear.weight.numel() * ratio)

        for (k, v) in svd.state_dict().items():
            state[f"{name}.{k}"] = v
            allKeys.append(f"{name}.{k}")

        if (paramNum >= threshold):
            savePath = os.path.join(saveDir, f"{len(statePaths)}.safetensors")
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
    print(f"Load temp state back, missing keys: {missing}, unexpected keys: {result.unexpected_keys}")

    model.save_pretrained(pretrainDir)

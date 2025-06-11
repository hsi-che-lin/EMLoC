import json
import torch
import logging
import os
import warnings
import tempfile
import math
from torch.utils.data import DataLoader
from transformers import set_seed
from safetensors.torch import load_file, save_file

from internvl.patch import monkey_patch
from internvl.utils.args import getArgs
from internvl.utils.utils import initLogger, rgetattr, rsetattr
from internvl.data.build import build_datasets
from internvl.model.build import build_tokenizer_and_model
from internvl.model.svd_internvl_chat.svd_utils import processCompressConfig
from internvl.model.svd_internvl_chat.svd_modules import SVDLinear
from internvl.engine.emulatorConstruct_utils.forwards import setNewForward
from internvl.engine.emulatorConstruct_utils.module import TwoPathLinear


warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)
os.environ['TOKENIZERS_PARALLELISM'] = 'true'


def _getModel(model_args, data_args, training_args, logDir = None):
    set_seed(training_args.seed)
    tokenizer, model = build_tokenizer_and_model(model_args, data_args, logDir)
    model = model.eval()
    model.config.svd_config = {
        "original_model_name": model_args.model_name_or_path,
        "compress": model_args.compress_config,
        "scaling": model_args.scaling_config
    }
    compress_config = processCompressConfig(model, model_args.compress_config)
    
    if (logDir is not None):
        compress_config_path = os.path.join(logDir, "expanded_compress_config.json")
        
        with open(compress_config_path, "w") as f:
            json.dump(compress_config, f, indent = 4)

    return tokenizer, model, compress_config


def _getDataloader(tokenizer, model, data_args, training_args):
    set_seed(training_args.seed)
    train_dataset, collator = build_datasets(
        data_args,
        tokenizer,
        model,
        train_batch_size = training_args.train_batch_size,
        group_by_length = training_args.group_by_length
    )
    dataloader = DataLoader(
        train_dataset,
        batch_size = training_args.train_batch_size,
        shuffle = True,
        num_workers = 4,
        pin_memory = True,
        collate_fn = collator
    )

    return dataloader


def _preprocessBatch(_batch, device, dtype = None):
    getDtype = lambda v: dtype if torch.is_floating_point(v) else None
    
    if (isinstance(_batch, dict)):
        batch = {
            k: v.to(device = device, dtype = getDtype(v))
            for (k, v) in _batch.items()
        }
    elif (isinstance(_batch, tuple)):
        batch = tuple(
            v.to(device = device, dtype = getDtype(v)) for v in _batch
        )
    
    return batch


class StateMaintainer(dict):
    def __init__(self, tmpDir):
        super().__init__()
        self.tempDir = tmpDir
        self.threshold = 4 * 1024 * 1024 * 1024
        self.numParam = 0
        self.tempPaths = []

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        
        self.numParam += value.numel()
        if (self.numParam > self.threshold):
            toSave = {
                k: v for (k, v) in self.items() if (v is not None)
            }
            path = os.path.join(self.tempDir.name, f"{len(self.tempPaths)}.safetensors")
            save_file(toSave, path)
            self.tempPaths.append(path)
            self.numParam = 0
            logger.info(f"Offloaded some states to disk ({path})")

            for k in toSave:
                super().__setitem__(k, None)

    def get_state_dict(self):
        state_dict = {k: v for (k, v) in self.items()}
        
        for path in self.tempPaths:
            state_dict.update(load_file(path))
        
        return state_dict


@torch.no_grad()
def _get_svd_state(model_args, data_args, training_args, tmpDir):
    logDir = os.path.join(training_args.output_dir, "log-model")
    tokenizer, model, compress_config = _getModel(model_args, data_args, training_args, logDir)
    dataloader = _getDataloader(tokenizer, model, data_args, training_args)

    dtype = model.config.torch_dtype
    state_dict = StateMaintainer(tmpDir)
    
    maxSampleNum = model.config.svd_config["scaling"]["max_sample_num"]
    batchSize = dataloader.batch_sampler.batch_size
    maxItr = math.ceil(maxSampleNum / batchSize)

    listBatch = {}

    for batch in dataloader:
        batch = _preprocessBatch(batch, "cpu", dtype)
        for (k, v) in batch.items():
            if (k not in listBatch): listBatch[k] = []
            listBatch[k].append(v)
        if (len(listBatch[k]) == maxItr): break
    
    for (name, module) in model.named_modules():
        setNewForward(module)
        
        if (isinstance(module, torch.nn.Linear) and ("mlp1" not in name)):
            ratio = compress_config.get(name, None)
            linear = TwoPathLinear(name, module, state_dict, ratio)
            rsetattr(model, name, linear)
    
    model(**listBatch)
    state = state_dict.get_state_dict()
    
    return state


def _get_svd_model(model_args, data_args, training_args, state):
    tokenizer, model, compress_config = _getModel(model_args, data_args, training_args)
    
    oriParamCnt = sum([p.numel() for p in model.parameters()])
    logger.info(f"before svd, model parameters = {oriParamCnt}")
    
    for (name, ratio) in compress_config.items():
        linear = rgetattr(model, name)
        svd = SVDLinear(ratio, refLinear = linear)
        rsetattr(model, name, svd)
    
    svdParamCnt = sum([p.numel() for p in model.parameters()])
    logger.info(f"after svd, model parameters = {svdParamCnt}")

    with open(os.path.join(training_args.output_dir, "log-model", "model-svd.txt"), "w") as f:
        f.write(str(model))
        f.write(f"\ntotal number of parameters: {svdParamCnt}\n")
        f.write(f"compress ratio = {svdParamCnt / oriParamCnt} ({svdParamCnt} / {oriParamCnt})")
    
    result = model.load_state_dict(state, strict = False, assign = True)
    logger.info(f"Load temp state back, unexpected keys: {result.unexpected_keys}")
    
    return model, tokenizer


def main():
    model_args, data_args, training_args = getArgs()
    monkey_patch(data_args.use_packed_ds)
    initLogger(training_args, logger)
    
    model_args.quantization_config = {
        "low_cpu_mem_usage": True,
        "torch_dtype": torch.bfloat16
    }
    
    tmpDir = tempfile.TemporaryDirectory()
    state = _get_svd_state(model_args, data_args, training_args, tmpDir)
    model, tokenizer = _get_svd_model(model_args, data_args, training_args, state)
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    tmpDir.cleanup()


if (__name__ == "__main__"):
    main()

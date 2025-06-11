import argparse
import dataclasses
import json
import logging
import warnings
import os
import sys
import shutil
import torch
import torch.distributed as dist
from transformers import HfArgumentParser, TrainingArguments, BitsAndBytesConfig
from dataclasses import dataclass, field
from typing import Literal, Optional


warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments for specifying model, tokenizer, and configurations.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to a pretrained model (local or from huggingface.co/models).'}
    )
    quantization_config_path: str = field(
        default=None,
        metadata={'help': 'The path of the meta file of model quantization.'}
    )
    quantization_config: dict = field(
        default=None,
        metadata={'help': 'Processed result of quantization_config_path'}
    )
    peft_config_path: str = field(
        default=None,
        metadata={'help': 'The path of the meta file of peft configuration.'}
    )
    peft_config: str = field(
        default=None,
        metadata={'help': 'Processed result of peft_config_path'}
    )
    use_svd: bool = field(
        default=False,
        metadata={'help': 'Set to True to use SVDInternVLChatModel. Default is False.'}
    )
    use_flash_attn: bool = field(
        default=False,
        metadata={'help': 'Set to True to use flash attention. Default is False.'}
    )

    compress_config_path: str = field(
        default=None,
        metadata={'help': 'The path of the meta file of compress configuration. (only for preparing SVDInternVLChatModel pretrained checkpoint)'}
    )
    compress_config: str = field(
        default=None,
        metadata={'help': 'Processed result of compress_config_path'}
    )
    scaling_config_path: str = field(
        default=None,
        metadata={'help': 'The path of the meta file of scaling configuration. (only for preparing SVDInternVLChatModel pretrained checkpoint)'}
    )
    scaling_config: str = field(
        default=None,
        metadata={'help': 'Processed result of scaling_config_path'}
    )
    
    freeze_llm: bool = field(
        default=True,
        metadata={'help': 'Set to True to freeze the LLM. Default is True.'},
    )
    freeze_backbone: bool = field(
        default=True,
        metadata={'help': 'Set to True to freeze the ViT. Default is True.'},
    )
    freeze_mlp: bool = field(
        default=True,
        metadata={'help': 'Set to True to freeze the MLP. Default is True.'},
    )
    unfreeze_lm_head: bool = field(
        default=False,
        metadata={'help': 'Set to True to unfreeze the head of LLM. Default is False.'},
    )
    unfreeze_vit_layers: int = field(
        default=0,
        metadata={'help': 'Specify the number of ViT layers to unfreeze. Default is 0.'},
    )
    vision_select_layer: int = field(
        default=-1,
        metadata={'help': 'Specify the layer of ViT feature map to use. Default is -1 for the last layer.'},
    )
    drop_path_rate: float = field(
        default=0.0,
        metadata={'help': 'Set the drop path rate for the ViT. Default is 0.'},
    )
    ps_version: Literal['v1', 'v2'] = field(
        default='v2',
        metadata={'help': 'Specify the version of pixel shuffle implementation. Default is v2.'}
    )
    grad_checkpoint: bool = field(
        default=True,
        metadata={'help': 'Set to True to use gradient checkpointing. Default is True.'},
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={'help': 'Set to True to use the fast mode of the tokenizer.'}
    )
    use_liger: bool = field(
        default=False,
        metadata={'help': 'Set to True to use the liger kernel.'}
    )

    vision_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to a pretrained model (local or from huggingface.co/models).'}
    )
    llm_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to a pretrained model (local or from huggingface.co/models).'}
    )
    mlp_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to a pretrained model (local or from huggingface.co/models).'}
    )
    use_backbone_lora: int = field(
        default=0,
        metadata={'help': 'Set the LoRA adapter rank for the ViT. Default is 0.'}
    )
    use_llm_lora: int = field(
        default=0,
        metadata={'help': 'Set the LoRA adapter rank for the LLM. Default is 0.'}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments for specifying data input for training and evaluation.
    """
    meta_path: str = field(
        default=None,
        metadata={'help': 'The path of the meta file of datasets.'},
    )
    
    max_seq_length: int = field(
        default=8192,
        metadata={
            'help': (
                'The maximum total input sequence length after tokenization. Sequences longer '
                'than this will be truncated, sequences shorter will be padded.'
            )
        },
    )
    force_image_size: int = field(
        default=448,
        metadata={'help': 'Set the desired size for the image. Default is 448.'},
    )
    down_sample_ratio: float = field(
        default=0.5,
        metadata={'help': 'Set the desired down-sampling ratio for the image. Default is 0.5.'},
    )
    pad2square: bool = field(
        default=False,
        metadata={'help': 'Pad the image to a square shape if set to True. Default is False.'},
    )
    conv_style: str = field(
        default='internvl2_5', metadata={'help': 'Prompt style for a conversation.'}
    )
    use_data_resampling: bool = field(
        default=False,
        metadata={'help': 'Set to True to use data resampling. Default is False.'},
    )
    dynamic_image_size: bool = field(
        default=True,
        metadata={'help': 'Set to True to use dynamic high resolution strategy. Default is False.'},
    )
    use_thumbnail: bool = field(
        default=True,
        metadata={'help': 'Set to True to add a thumbnail image. Default is False.'},
    )
    min_dynamic_patch: int = field(
        default=1,
        metadata={'help': 'The minimum number of dynamic patches. Default is 1.'},
    )
    max_dynamic_patch: int = field(
        default=6,
        metadata={'help': 'The maximum number of dynamic patches. Default is 6.'},
    )
    min_num_frame: int = field(
        default=8,
        metadata={'help': 'The minimum number of frames for video data. Default is 8.'},
    )
    max_num_frame: int = field(
        default=32,
        metadata={'help': 'The maximum number of frames for video data. Default is 32.'},
    )
    normalize_type: Literal['imagenet', 'clip', 'siglip'] = field(
        default='imagenet',
        metadata={'help': 'The normalization type for the image. Default is imagenet.'},
    )
    use_packed_ds: bool = field(
        default=False,
        metadata={'help': 'Whether to use packed dataset for efficient training. Default is False.'},
    )
    num_images_expected: int = field(
        default=40,
        metadata={'help': 'The maximum number of images per packed sample. Default is 40.'},
    )
    max_packed_tokens: int = field(
        default=8192,
        metadata={'help': 'The required token length of per packed sample. Default is 8192.'},
    )
    max_buffer_size: int = field(
        default=20,
        metadata={'help': 'The buffer size of the packed dataset. Default is 20.'},
    )
    log_freq: int = field(
        default=1000,
        metadata={'help': 'The log frequency of the packed dataset. Default is 1000.'},
    )
    strict_mode: bool = field(
        default=True,
        metadata={'help': 'Whether to pad the number of images to satisfy num_images_expected. Default is True.'},
    )
    replacement: bool = field(
        default=False,
        metadata={'help': 'Whether to restart the dataset after it is exhausted. Default is False.'},
    )
    allow_overflow: bool = field(
        default=False,
        metadata={'help': 'Whether to drop the sample over the specified max_packed_tokens. Default is False.'},
    )
    loss_reduction: str = field(
        default='token',
        metadata={'help': 'Loss reduction method. Default is token.'},
    )
    loss_reduction_all_gather: bool = field(
        default=False,
        metadata={'help': 'Whether to gather all during loss reduction. Default is False.'},
    )


def _readJSON(path):
    if (not os.path.isfile(path)):
        logger.warning(f"config file \"{path}\" does not exist, return empty config")
        return {}
    
    with open(path, "r") as f:
        data = json.load(f)
    
    return data


def _process_quantization_config(path):
    def _str2dtype(d):
        mapping = {
            "torch.float16": torch.float16,
            "torch.bfloat16": torch.bfloat16
        }

        new = {
            k: mapping.get(v, v) for (k, v) in d.items()
        }

        return new
    
    config = _readJSON(path)
    bnbConfig = config.pop("quantization_config", None)
    quantization_config = _str2dtype(config)

    if (bnbConfig is not None):
        bnbConfig = _str2dtype(bnbConfig)
        quantization_config["quantization_config"] = BitsAndBytesConfig(**bnbConfig)
    
    return quantization_config


def parseMyArgs():
    parser = argparse.ArgumentParser()
    # model_args
    parser.add_argument("--model_name_or_path", type = str, default = "")
    parser.add_argument("--quantization_config", type = str, default = "")
    parser.add_argument("--peft_config", type = str, default = "")
    parser.add_argument("--use_svd", type = str, default = "False", choices = ["True", "False"])
    parser.add_argument("--use_flash_attn", type = str, default = "False", choices = ["True", "False"])
    parser.add_argument("--compress_config", type = str, default = "")
    parser.add_argument("--scaling_config", type = str, default = "")
    # data_args
    parser.add_argument("--meta_path", type = str, default = "")
    # training_args
    parser.add_argument("--output_dir", type = str, default = "")
    parser.add_argument("--training_args", type = str, default = "")

    args = parser.parse_args()
    model_args = ModelArguments(
        model_name_or_path = args.model_name_or_path,
        quantization_config_path = args.quantization_config,
        peft_config_path = args.peft_config,
        use_svd = (args.use_svd == "True"),
        use_flash_attn = (args.use_flash_attn == "True"),
        compress_config_path = args.compress_config,
        scaling_config_path = args.scaling_config
    )
    data_args = DataTrainingArguments(meta_path = args.meta_path)
    _training_args = _readJSON(args.training_args)
    training_args = TrainingArguments(
        output_dir = args.output_dir,
        **_training_args
    )
    
    return model_args, data_args, training_args, args.training_args


def saveArgs(model_args, data_args, training_args, training_args_path, saveDir):
    def _is_json_serializable(variable):
        try:
            json.dumps(variable)
            return True
        except TypeError:
            return False

    def _toJSON(d):
        new = {}
        for (k, v) in d.items():
            if (hasattr(v, "to_dict")):
                new[k] = _toJSON(v.to_dict())
            elif (isinstance(v, dict)):
                new[k] = _toJSON(v)
            elif _is_json_serializable(v):
                new[k] = v
            else:
                new[k] = str(v)
        
        return new
    
    if ((not dist.is_initialized()) or (dist.get_rank() == 0)):
        os.makedirs(saveDir, exist_ok = True)
        args = _toJSON({
            "model_args": dataclasses.asdict(model_args),
            "data_args": dataclasses.asdict(data_args),
            "training_args": dataclasses.asdict(training_args)
        })

        with open(os.path.join(saveDir, "args.json"), "w") as f:
            json.dump(args, f, indent = 4)
        
        shutil.copyfile(model_args.quantization_config_path, os.path.join(saveDir, "quantization.json"))
        
        if (training_args_path is not None):
            shutil.copyfile(training_args_path, os.path.join(saveDir, "training.json"))


def getArgs():
    if (sys.argv[1] == "--useHfArgumentParser"):
        sys.argv.pop(1)
        parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        training_args_path = None
    else:
        model_args, data_args, training_args, training_args_path = parseMyArgs()
    
    model_args.quantization_config = _process_quantization_config(model_args.quantization_config_path)
    model_args.peft_config = _readJSON(model_args.peft_config_path)
    model_args.compress_config = _readJSON(model_args.compress_config_path)
    model_args.scaling_config = _readJSON(model_args.scaling_config_path)
    training_args.use_packed_ds = data_args.use_packed_ds
    saveDir = os.path.join(training_args.output_dir, "log-args")
    saveArgs(model_args, data_args, training_args, training_args_path, saveDir)

    return model_args, data_args, training_args

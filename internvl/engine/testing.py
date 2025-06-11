import argparse
import json
import logging
import os
import warnings
import torch
from tqdm import tqdm
from PIL import Image, ImageFile, PngImagePlugin
from torch.utils.data import DataLoader, Dataset
from transformers import set_seed, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from safetensors.torch import load_file
from peft.utils import set_peft_model_state_dict

from internvl.data.dataset import build_transform, dynamic_preprocess
from internvl.model.build import build_tokenizer_and_model
from internvl.patch import monkey_patch
from internvl.utils.args import ModelArguments, DataTrainingArguments, saveArgs, _process_quantization_config
from internvl.utils.utils import initLogger


Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte
warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)
os.environ['TOKENIZERS_PARALLELISM'] = 'true'


def getCkptArgs(args):
    with open(os.path.join(args.ckpt, "log-args", "args.json"), "r") as f:
        ckpt = json.load(f)

    model_args = ModelArguments(**ckpt["model_args"])
    model_args.quantization_config = _process_quantization_config(model_args.quantization_config_path)
    
    if (args.plugback != ""):
        model_args.model_name_or_path = args.plugback
    
    if (args.zeroshot == "True"):
        model_args.peft_config["name"] = "no"
    
    data_args = DataTrainingArguments(**ckpt["data_args"])
    data_args.meta_path = args.meta_path
    
    with open(args.training_args, "r") as f:
        training_args = TrainingArguments(
            output_dir = args.output_dir,
            **json.load(f)
        )

    savePath = os.path.join(training_args.output_dir, "log-args")
    saveArgs(model_args, data_args, training_args, args.training_args, savePath)

    return model_args, data_args, training_args


def loadPEFT(ckpt, model, plugback, device_map = "auto", max_memory = None, offload_folder = None, offload_index = None):
    # didn't use load_adapter since it'll be troublesome to add adapter to SVDLinear by this function
    # copy and modify from PeftAdapterMixin at transformers.integrations.peft

    model = model.base_model.model  # In PeftAdapterMixin self is PretrainedModel not PEFTModel
    path = os.path.join(get_last_checkpoint(ckpt), "adapter_model.safetensors")
    logger.info(f"load lora from {path}")
    adapter_state_dict = load_file(path)
    
    processed_adapter_state_dict = {}
    prefix = "base_model.model."
    for key, value in adapter_state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix) :]
        else:
            new_key = key
        processed_adapter_state_dict[new_key] = value

    # Load state dict
    incompatible_keys = set_peft_model_state_dict(model, processed_adapter_state_dict)

    if incompatible_keys is not None:
        # check only for unexpected keys
        if hasattr(incompatible_keys, "unexpected_keys") and len(incompatible_keys.unexpected_keys) > 0:
            logger.warning(
                f"Loading adapter weights from {path} led to unexpected keys not found in the model: "
                f" {incompatible_keys.unexpected_keys}. "
            )

    # Re-dispatch model and hooks in case the model is offloaded to CPU / Disk.
    if (
        (getattr(model, "hf_device_map", None) is not None)
        and (len(set(model.hf_device_map.values()).intersection({"cpu", "disk"})) > 0)
        and len(model.peft_config) == 1
    ):
        model._dispatch_accelerate_model(
            device_map=device_map,
            max_memory=max_memory,
            offload_folder=offload_folder,
            offload_index=offload_index,
        )
    
    return model


class TestingDataset(Dataset):
    def __init__(self, data_args):
        super(TestingDataset, self).__init__()
        self.data_args = data_args
        self.data = self.getData()
        self.transform = build_transform(
            is_train = False,
            input_size = data_args.force_image_size,
            pad2square = data_args.pad2square,
            normalize_type = data_args.normalize_type
        )
        
   
    def getData(self):
        data = []
        
        with open(self.data_args.meta_path, "r") as f:
            meta = list(json.load(f).values())[0]
            root = meta["root"]
            jsonl = meta["annotation"]
        
        with open(jsonl, "r") as f:
            for l in f:
                sample = json.loads(l)
                data.append((
                    os.path.join(root, sample["image"]),
                    sample["conversations"][0]["value"],
                    sample["conversations"][1]["value"]
                ))
        
        return data
    

    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, idx):
        image_path, question, answer = self.data[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.data_args.dynamic_image_size:
            images = dynamic_preprocess(
                image,
                min_num = self.data_args.min_dynamic_patch,
                max_num = self.data_args.max_dynamic_patch,
                image_size = self.data_args.force_image_size,
                use_thumbnail = self.data_args.use_thumbnail
            )
        else:
            images = [image]
        
        pixel_values = [self.transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)

        return pixel_values, question, answer
    

    def collate(self, batch):
        images = torch.concat([x[0] for x in batch])
        num_list = [x[0].shape[0] for x in batch]
        questions = [x[1] for x in batch]
        answers = [x[2] for x in batch]

        return (images, num_list, questions, answers)


@torch.no_grad()
def main():
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type = str, default = "")
    parser.add_argument("--meta_path", type = str, default = "")
    parser.add_argument("--output_dir", type = str, default = "")
    parser.add_argument("--plugback", default = "")
    parser.add_argument("--zeroshot", default = "False", choices = ["True", "False"])
    parser.add_argument("--generation_config", type = str, default = "")
    parser.add_argument("--training_args", type = str, default = "")
    args = parser.parse_args()
    model_args, data_args, training_args = getCkptArgs(args)
    monkey_patch(data_args.use_packed_ds)
    initLogger(training_args, logger)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # model
    set_seed(training_args.seed)
    logDir = os.path.join(training_args.output_dir, "log-model")
    tokenizer, model = build_tokenizer_and_model(model_args, data_args, logDir)
    
    if (args.zeroshot != "True"):
        model = loadPEFT(args.ckpt, model, args.plugback)
    
    if ("quantization_config" not in model_args.quantization_config):
        model = model.to(device)
        
    model.requires_grad_(False)
    model.eval()

    # data
    set_seed(training_args.seed)
    dataset = TestingDataset(data_args)
    dataloader = DataLoader(
        dataset,
        batch_size = training_args.per_device_train_batch_size,
        shuffle = False,
        num_workers = 4,
        pin_memory = True,
        collate_fn = dataset.collate
    )
    
    # inference
    with open(args.generation_config, "r") as f:
        generation_config = json.load(f)
    generation_config["pad_token_id"] = tokenizer.eos_token_id
    result = []
    dtype = model.config.torch_dtype
    
    for (pixel_values, num_patches_list, questions, answers) in tqdm(dataloader):
        pixel_values = pixel_values.to(dtype = dtype, device = device)
        responses = model.batch_chat(
            tokenizer,
            pixel_values,
            questions,
            generation_config,
            num_patches_list
        )
        for (ans, res) in zip(answers, responses):
            result.append((ans, res))
    
    with open(os.path.join(training_args.output_dir, "result.json"), "w", encoding = "utf-8") as f:
        json.dump(result, f, indent = 4, ensure_ascii = False)


if (__name__ == "__main__"):
    main()
import os
import logging
import torch
import warnings
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model
from peft.tuners.lora.layer import Linear as LinearWithLora

from internvl.utils.constants import (BOX_END_TOKEN, BOX_START_TOKEN,
                                      IMG_CONTEXT_TOKEN, IMG_END_TOKEN,
                                      IMG_START_TOKEN, QUAD_END_TOKEN,
                                      QUAD_START_TOKEN, REF_END_TOKEN,
                                      REF_START_TOKEN)
from internvl.model.internvl_chat import InternVLChatConfig, InternVLChatModel
from internvl.model.svd_internvl_chat import SVDInternVLChatConfig, SVDInternVLChatModel
from internvl.model.svd_internvl_chat.svd_modules import SVDLinear


warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


def build_tokenizer(model_args, data_args):
    # Load pretrained model, tokenizer, and image processor
    tokenizer_path = model_args.model_name_or_path or model_args.llm_path
    logger.info(f'Loading Tokenizer: {tokenizer_path}')
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        add_eos_token = False,
        trust_remote_code = True,
        use_fast = model_args.use_fast_tokenizer
    )
    tokenizer.tokenizer_path = tokenizer_path
    tokenizer.model_max_length = data_args.max_seq_length

    return tokenizer


def _maybe_change_image_size(model, data_args):
    patch_size = model.config.vision_config.patch_size
    logger.info(f'model.config.force_image_size: {model.config.force_image_size}')
    logger.info(f'data_args.force_image_size: {data_args.force_image_size}')
    logger.info(f'model.config.vision_config.image_size: {model.config.vision_config.image_size}')
    
    if (model.config.vision_config.image_size != data_args.force_image_size):
        logger.info(f'Resizing position embedding from '
                    f'{model.config.vision_config.image_size} '
                    f'to {data_args.force_image_size}...')
        model.vision_model.resize_pos_embeddings(
            old_size = model.config.vision_config.image_size,
            new_size = data_args.force_image_size,
            patch_size = patch_size
        )
        model.config.vision_config.image_size = data_args.force_image_size
        
    model.config.force_image_size = data_args.force_image_size
    model.num_image_token = int((data_args.force_image_size // patch_size) ** 2 * (data_args.down_sample_ratio ** 2))


def build_model(model_args, data_args):
    logger.info('Loading InternVLChatModel...')
    
    if (model_args.use_svd):
        configCLS = SVDInternVLChatConfig
        modelCLS = SVDInternVLChatModel
    else:
        configCLS = InternVLChatConfig
        modelCLS = InternVLChatModel
    
    config = configCLS.from_pretrained(model_args.model_name_or_path)
    
    logger.info('Overriding model config...')
    config.vision_config.drop_path_rate = model_args.drop_path_rate
    config.use_flash_attn     = model_args.use_flash_attn
    config.ps_version         = model_args.ps_version
    config.select_layer       = model_args.vision_select_layer
    config.template           = data_args.conv_style
    config.dynamic_image_size = data_args.dynamic_image_size
    config.use_thumbnail      = data_args.use_thumbnail
    config.min_dynamic_patch  = data_args.min_dynamic_patch
    config.max_dynamic_patch  = data_args.max_dynamic_patch
    model = modelCLS.from_pretrained(
        model_args.model_name_or_path,
        config = config,
        **model_args.quantization_config
    )
    _maybe_change_image_size(model, data_args)
    assert (model.config.downsample_ratio == data_args.down_sample_ratio)
    
    return model


def _maybe_extend_vocabulary(tokenizer, model):
    token_list = [IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN,
                  QUAD_START_TOKEN, QUAD_END_TOKEN, REF_START_TOKEN,
                  REF_END_TOKEN, BOX_START_TOKEN, BOX_END_TOKEN]
    num_new_tokens = tokenizer.add_tokens(token_list, special_tokens = True)
    model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    
    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        output_embeddings = model.language_model.get_output_embeddings().weight.data
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim = 0, keepdim = True)
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)


def freeze_and_load(model_args, model):
    def _freeze_params(module):
        for param in module.parameters():
            param.requires_grad = False
    
    if model_args.freeze_backbone:
        _freeze_params(model.vision_model)

    if model_args.freeze_llm:
        model.language_model = model.language_model.eval()
        _freeze_params(model.language_model)

    if model_args.unfreeze_lm_head:
        model.language_model.lm_head.requires_grad = True

    if model_args.freeze_mlp:
        _freeze_params(model.mlp1)

    if model_args.unfreeze_vit_layers != 0:
        layers = model.vision_model.encoder.layers[model_args.unfreeze_vit_layers:]
        for k, v in layers.named_parameters():
            logger.info(f'Unfreezing ViT layer: {k}')
            v.requires_grad = True

    if model_args.mlp_path is not None:
        logger.info('Loading pretrained MLP projector...')
        state_dict = torch.load(model_args.mlp_path, map_location='cpu')
        message = model.mlp1.load_state_dict(state_dict)
        logger.info(message)


def addPEFT(model_args, model):
    if (model_args.peft_config.get("name", "no") == "no"):
        pass
    elif (model_args.peft_config["name"] == "lora"):
        loraConfig = LoraConfig(**model_args.peft_config["config"])
        if (model_args.use_svd):
            loraConfig._register_custom_module({SVDLinear: LinearWithLora})
        model = get_peft_model(model, loraConfig)
        model.enable_input_require_grads()
    else:
        raise

    return model


def build_tokenizer_and_model(model_args, data_args, logDir = None):
    if model_args.use_liger:
        from internvl.patch import apply_liger_kernel_to_internvit
        from liger_kernel.transformers import (apply_liger_kernel_to_llama,
                                               apply_liger_kernel_to_qwen2)
        apply_liger_kernel_to_llama()
        apply_liger_kernel_to_qwen2()
        # apply_liger_kernel_to_internvit()
    
    tokenizer = build_tokenizer(model_args, data_args)
    model = build_model(model_args, data_args)
    _maybe_extend_vocabulary(tokenizer, model)

    # cache and gradient checkpoint
    model.language_model.config.use_cache = False
    if model_args.grad_checkpoint:
        model.vision_model.gradient_checkpointing = True
        model.vision_model.encoder.gradient_checkpointing = True
        model.language_model._set_gradient_checkpointing()
    
    freeze_and_load(model_args, model)
    model = addPEFT(model_args, model)

    if (logDir is not None):
        os.makedirs(logDir, exist_ok = True)
        
        with open(os.path.join(logDir, "model.txt"), "w") as f:
            f.write(str(model))
        
        with open(os.path.join(logDir, "trainable.txt"), "w") as f:
            total = 0
            trainable = 0
            
            for name, param in model.named_parameters():
                # these are from PeftMixedModel
                numel = param.numel()
                
                if numel == 0 and hasattr(param, "ds_numel"):
                    numel = param.ds_numel

                if param.__class__.__name__ == "Params4bit":
                    numel = numel * 2

                total += numel
                
                if param.requires_grad:
                    trainable += numel
                    f.write(name + "\n")
            
            f.write(f"\ntotal: {total}, trainable: {trainable} ({trainable / total * 100:.2f}%)")
    
    return tokenizer, model

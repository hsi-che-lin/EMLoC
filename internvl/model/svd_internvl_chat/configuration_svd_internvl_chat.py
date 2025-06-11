# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import copy

from internvl.model.internvl_chat.modeling_intern_vit import InternVisionConfig
from internvl.model.internlm2.configuration_internlm2 import InternLM2Config
from internvl.model.phi3.configuration_phi3 import Phi3Config
from transformers import LlamaConfig, Qwen2Config
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class SVDInternVLChatConfig(PretrainedConfig):
    model_type = 'svd_internvl_chat'
    is_composition = True

    def __init__(
            self,
            svd_config=None,
            vision_config=None,
            llm_config=None,
            use_flash_attn=False,
            pad2square=False,
            select_layer=-1,
            force_image_size=None,
            downsample_ratio=0.5,
            template=None,
            dynamic_image_size=False,
            use_thumbnail=False,
            ps_version='v1',
            min_dynamic_patch=1,
            max_dynamic_patch=6,
            **kwargs):
        
        if ("use_backbone_lora" in kwargs):
            logger.warning("Remove 'use_backbone_lora' argument.")
            kwargs.pop("use_backbone_lora")
        if ("use_llm_lora" in kwargs):
            logger.warning("Remove 'use_llm_lora' argument.")
            kwargs.pop("use_llm_lora")

        super().__init__(**kwargs)

        if (svd_config is None):
            self.svd_config = {
                "original_model_name": "",
                "compress": {},
                "scaling": {}
            }
            logger.info('svd_config is None. Initializing with default values.')
        else:
            self.svd_config = svd_config

        if vision_config is None:
            vision_config = {'architectures': ['InternVisionModel']}
            logger.info('vision_config is None. Initializing the InternVisionConfig with default values.')

        if llm_config is None:
            # TODO: There might still be a bug in transformers version 4.44 and above.
            llm_config = {'architectures': ['']}
            logger.info('llm_config is None. Initializing the LlamaConfig config with default values (`LlamaConfig`).')

        self.vision_config = InternVisionConfig(**vision_config)
        if llm_config['architectures'][0] == 'LlamaForCausalLM':
            self.llm_config = LlamaConfig(**llm_config)
        elif llm_config['architectures'][0] == 'InternLM2ForCausalLM':
            self.llm_config = InternLM2Config(**llm_config)
        elif llm_config['architectures'][0] == 'Phi3ForCausalLM':
            self.llm_config = Phi3Config(**llm_config)
        elif llm_config['architectures'][0] == 'Qwen2ForCausalLM':
            self.llm_config = Qwen2Config(**llm_config)
        else:
            raise ValueError('Unsupported architecture: {}'.format(llm_config['architectures'][0]))
        
        self.use_flash_attn = use_flash_attn
        self.pad2square = pad2square
        self.select_layer = select_layer
        self.force_image_size = force_image_size
        self.downsample_ratio = downsample_ratio
        self.template = template
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.ps_version = ps_version  # pixel shuffle version
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch

        self.hidden_size = self.llm_config.hidden_size
        # By default, we use tie_word_embeddings=False for models of all sizes.
        self.tie_word_embeddings = False
        self.llm_config.tie_word_embeddings = self.tie_word_embeddings

        logger.info(f'vision_select_layer: {self.select_layer}')
        logger.info(f'ps_version: {self.ps_version}')
        logger.info(f'min_dynamic_patch: {self.min_dynamic_patch}')
        logger.info(f'max_dynamic_patch: {self.max_dynamic_patch}')
    
    
    @property
    def use_flash_attn(self):
        if (self.llm_config.model_type == "internlm2"):
            attn_implement = self.llm_config.attn_implementation
        else:
            attn_implement = self.llm_config._attn_implementation
        
        use_flash_attn = (attn_implement == "flash_attention_2") and (self.vision_config.use_flash_attn)
        
        return use_flash_attn
    
    
    @use_flash_attn.setter
    def use_flash_attn(self, value):
        self.vision_config.use_flash_attn = bool(value)
        attn_implement = "flash_attention_2" if value else "eager"
        
        if (self.llm_config.model_type == "internlm2"):
            self.llm_config.attn_implementation = attn_implement
        else:
            self.llm_config._attn_implementation = attn_implement


    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output['svd_config'] = self.svd_config
        output['vision_config'] = self.vision_config.to_dict()
        output['llm_config'] = self.llm_config.to_dict()
        output['model_type'] = self.__class__.model_type
        output['use_flash_attn'] = self.use_flash_attn
        output['select_layer'] = self.select_layer
        output['force_image_size'] = self.force_image_size
        output['downsample_ratio'] = self.downsample_ratio
        output['template'] = self.template
        output['dynamic_image_size'] = self.dynamic_image_size
        output['use_thumbnail'] = self.use_thumbnail
        output['ps_version'] = self.ps_version
        output['min_dynamic_patch'] = self.min_dynamic_patch
        output['max_dynamic_patch'] = self.max_dynamic_patch

        return output

import logging
import types

from internvl.model.svd_internvl_chat import SVDInternVLChatModel
from internvl.engine.emulatorConstruct_utils.modified_forwards.intern_vit import forwardMap as vitForwardMap
from internvl.engine.emulatorConstruct_utils.modified_forwards.internlm import forwardMap as internForwardMap
from internvl.engine.emulatorConstruct_utils.modified_forwards.qwen import forwardMap as qwenForwardMap

logger = logging.getLogger(__name__)

def setNewForward(module):
    if (isinstance(module, SVDInternVLChatModel)):
        module.forward = types.MethodType(SVDInternVLChatModel_forward, module)
        module.extract_feature = types.MethodType(extract_feature, module)
    else:
        for (t, forward) in (vitForwardMap + internForwardMap + qwenForwardMap):
            if (isinstance(module, t)):
                module.forward = types.MethodType(forward, module)
        

def extract_feature(self, pixel_values):
    if self.select_layer == -1:
        vit_embeds_list = self.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=False,
            return_dict=True).last_hidden_state
    else:
        raise
    
    self.mlp1.to("cuda")
    
    result = []
    for vit_embeds in vit_embeds_list:
        vit_embeds = vit_embeds[:, 1:, :]
        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        result.append(vit_embeds)
    
    self.mlp1.to("meta")
    
    return result

def SVDInternVLChatModel_forward(
    self,
    pixel_values,
    input_ids = None,
    attention_mask = None,
    position_ids = None,
    image_flags = None,
    past_key_values = None,
    labels = None,
    use_cache = None,
    output_attentions = None,
    output_hidden_states = None,
    return_dict = None,
    statistics = None,
    loss_weight = None,
    loss_reduction_all_gather = False,
):
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    vit_embeds = self.extract_feature(pixel_values)
    vit_embeds = [emb[flag.squeeze(-1) == 1] for (emb, flag) in zip(vit_embeds, image_flags)]

    input_embeds = []
    
    self.language_model.get_input_embeddings().to("cuda")
    for (ids, vEmbed) in zip(input_ids, vit_embeds):
        ids = ids.to("cuda")
        input_embed = self.language_model.get_input_embeddings()(ids).clone()
        B, N, C = input_embed.shape
        input_embed = input_embed.reshape(B * N, C)
        ids = ids.reshape(B * N)
        selected = (ids == self.img_context_token_id)
        
        try:
            input_embed[selected] = input_embed[selected] * 0.0 + vEmbed.reshape(-1, C)
        except Exception as e:
            raise

        input_embed = input_embed.reshape(B, N, C)
        input_embeds.append(input_embed)
    self.language_model.get_input_embeddings().to("meta")

    outputs = self.language_model(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

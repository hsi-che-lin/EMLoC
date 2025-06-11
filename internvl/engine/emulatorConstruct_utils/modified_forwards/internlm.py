import torch
import math
from einops import rearrange

from internvl.model.internlm2.modeling_internlm2 import (InternLM2ForCausalLM,
                                                         InternLM2Model,
                                                         InternLM2DecoderLayer,
                                                         InternLM2Attention,
                                                         InternLM2MLP)
from internvl.model.internlm2.modeling_internlm2 import apply_rotary_pos_emb, repeat_kv


def InternLM2ForCausalLM_forward(
    self,
    input_ids = None,
    attention_mask = None,
    position_ids = None,
    past_key_values = None,
    inputs_embeds = None,
    labels = None,
    use_cache = None,
    output_attentions = None,
    output_hidden_states = None,
    return_dict = None,
):
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )


def InternLM2Model_forward(
    self,
    input_ids = None,
    attention_mask = None,
    position_ids = None,
    past_key_values = None,
    inputs_embeds = None,
    use_cache = None,
    output_attentions = None,
    output_hidden_states = None,
    return_dict = None,
):
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if self.config.attn_implementation == 'flash_attention_2': raise

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
    elif input_ids is not None:
        raise
    elif inputs_embeds is not None:
        pass
    else:
        raise ValueError('You have to specify either input_ids or inputs_embeds')

    if past_key_values is not None: raise
    if position_ids is None: raise
    if attention_mask is None: raise
    if self.gradient_checkpointing and self.training: raise
    
    _attention_mask = []
    for (embed, attn_mask) in zip(inputs_embeds, attention_mask):
        batch_size, seq_length = embed.shape[:2]
        past_key_values_length = 0
        mask = self._prepare_decoder_attention_mask(
            attn_mask, (batch_size, seq_length), embed, past_key_values_length
        )
        _attention_mask.append(mask)

    # embed positions
    attention_mask = _attention_mask
    hidden_states = inputs_embeds

    # decoder layers
    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states: raise
        past_key_value = past_key_values[idx] if past_key_values is not None else None

        if self.gradient_checkpointing and self.training:
            raise
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            decoder_layer.to("meta")

        hidden_states = layer_outputs[0]

        if use_cache: raise
        if output_attentions: raise


def InternLM2DecoderLayer_forward(
    self,
    hidden_states,
    attention_mask = None,
    position_ids = None,
    past_key_value = None,
    output_attentions = False,
    use_cache = False,
    **kwargs,
):
    if 'padding_mask' in kwargs: raise

    # Self Attention
    residual = hidden_states
    
    self.attention_norm.to("cuda")
    hidden_states = [self.attention_norm(x) for x in hidden_states]
    self.attention_norm.to("meta")

    hidden_states, self_attn_weights, present_key_value = self.attention(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        **kwargs,
    )
    self.attention.to("meta")
    hidden_states = [(residual.pop(0) + hidden_states.pop(0)) for _ in range(len(residual))]

    # Fully Connected
    residual = hidden_states
    
    self.ffn_norm.to("cuda")
    hidden_states = [self.ffn_norm(x) for x in hidden_states]
    self.ffn_norm.to("meta")
    
    hidden_states = self.feed_forward(hidden_states)
    self.feed_forward.to("meta")
    hidden_states = [(residual.pop(0) + hidden_states.pop(0)) for _ in range(len(residual))]

    outputs = (hidden_states,)
    if output_attentions: raise
    if use_cache: raise

    return outputs


def computeAttention(
    rotary_emb,
    num_key_value_groups,
    head_dim,
    num_heads,
    hidden_size,
    qkv_states,
    bsz,
    q_len,
    attention_mask,
    position_ids
):
    qkv_states = rearrange(
        qkv_states,
        'b q (h gs d) -> b q h gs d',
        gs=2 + num_key_value_groups,
        d=head_dim,
    )

    query_states = qkv_states[..., : num_key_value_groups, :]
    query_states = rearrange(query_states, 'b q h gs d -> b q (h gs) d')
    key_states = qkv_states[..., -2, :]
    value_states = qkv_states[..., -1, :]

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    cos, sin = rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    key_states = repeat_kv(key_states, num_key_value_groups)
    value_states = repeat_kv(value_states, num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)

    if attn_weights.size() != (bsz, num_heads, q_len, kv_seq_len):
        raise ValueError(
            f'Attention weights should be of size {(bsz, num_heads, q_len, kv_seq_len)}, but is'
            f' {attn_weights.size()}'
        )

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f'Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}'
            )
        attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, num_heads, q_len, head_dim):
        raise ValueError(
            f'`attn_output` should be of size {(bsz, num_heads, q_len, head_dim)}, but is'
            f' {attn_output.size()}'
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, hidden_size)

    del qkv_states, query_states, key_states, value_states, cos, sin, attn_weights
    torch.cuda.empty_cache()
    
    return attn_output


def InternLM2Attention_forward(
    self,
    hidden_states,
    attention_mask = None,
    position_ids = None,
    past_key_value = None,
    output_attentions = False,
    use_cache = False,
    **kwargs,
):
    if 'padding_mask' in kwargs: raise
    if past_key_value is not None: raise

    bszList, qLenList = [], []
    for x in hidden_states:
        bsz, q_len, _ = x.size()
        bszList.append(bsz)
        qLenList.append(q_len)
    
    self.wqkv.to("cuda")
    hidden_states = self.wqkv(hidden_states)
    self.wqkv.to("meta")
    
    self.rotary_emb.to("cuda")
    attn_output = []
    for (bsz, q_len, mask, pos_ids) in zip(bszList, qLenList, attention_mask, position_ids):
        attn_output.append(computeAttention(
            self.rotary_emb,
            self.num_key_value_groups,
            self.head_dim,
            self.num_heads,
            self.hidden_size,
            hidden_states.pop(0),
            bsz,
            q_len,
            mask,
            pos_ids
        ))
    self.rotary_emb.to("meta")

    self.wo.to("cuda")
    attn_output = self.wo(attn_output)
    self.wo.to("meta")

    if output_attentions: raise
    else: attn_weights = None

    return attn_output, attn_weights, past_key_value


def InternLM2MLP_forward(self, listX):
    self.w1.to("cuda")
    listX1 = self.w1(listX, popInputs = False)
    self.w1.to("meta")

    self.w3.to("cuda")
    listX3 = self.w3(listX)
    self.w3.to("meta")

    listX = [self.act_fn(listX1.pop(0)) * listX3.pop(0) for _ in range(len(listX1))]
    torch.cuda.empty_cache()
    
    self.w2.to("cuda")
    listX = self.w2(listX)
    self.w2.to("meta")

    return listX


forwardMap = (
    (InternLM2ForCausalLM, InternLM2ForCausalLM_forward),
    (InternLM2Model, InternLM2Model_forward),
    (InternLM2DecoderLayer, InternLM2DecoderLayer_forward),
    (InternLM2Attention, InternLM2Attention_forward),
    (InternLM2MLP, InternLM2MLP_forward)
)
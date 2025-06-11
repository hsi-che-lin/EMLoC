import math
import torch
from transformers.models.qwen2.modeling_qwen2 import (Qwen2ForCausalLM,
                                                      Qwen2Model,
                                                      Qwen2DecoderLayer,
                                                      Qwen2Attention,
                                                      Qwen2MLP)

from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask


def Qwen2ForCausalLM_forward(
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


def Qwen2Model_forward(
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

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
    elif input_ids is not None:
        raise
    elif inputs_embeds is not None:
        pass
    else:
        raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

    if self.gradient_checkpointing and self.training: raise
    if use_cache: raise
    if position_ids is None: raise
    if attention_mask is not None and self._attn_implementation == "flash_attention_2" and use_cache:
        raise
    if self._attn_implementation == "flash_attention_2":
        raise
    elif self._attn_implementation == "sdpa" and not output_attentions:
        raise

    attention_mask_ = []
    position_ids_ = []
    for embed in inputs_embeds:
        mask = attention_mask.pop(0)
        pos = position_ids.pop(0)

        past_key_values_length = 0
        batch_size, seq_length, _ = embed.shape
        pos = pos.view(-1, seq_length).long()
        mask = _prepare_4d_causal_attention_mask(
            mask,
            (batch_size, seq_length),
            embed,
            past_key_values_length,
            sliding_window=self.config.sliding_window,
        ).to("cuda")
        attention_mask_.append(mask)
        position_ids_.append(pos)

    hidden_states = inputs_embeds
    attention_mask = attention_mask_
    position_ids = position_ids_

    # decoder layers
    for decoder_layer in self.layers:
        if output_hidden_states: raise

        if self.gradient_checkpointing and self.training:
            raise
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            decoder_layer.to("meta")

        hidden_states = layer_outputs[0]

        if use_cache: raise
        if output_attentions: raise


def Qwen2DecoderLayer_forward(
    self,
    hidden_states,
    attention_mask = None,
    position_ids = None,
    past_key_value = None,
    output_attentions = False,
    use_cache = False,
    **kwargs,
):
    if "padding_mask" in kwargs: raise

    # Self Attention
    residual = hidden_states

    self.input_layernorm.to("cuda")
    hidden_states = [self.input_layernorm(x) for x in hidden_states]
    self.input_layernorm.to("meta")

    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
    )
    self.self_attn.to("meta")
    hidden_states = [(residual.pop(0) + hidden_states.pop(0)) for _ in range(len(residual))]

    # Fully Connected
    residual = hidden_states
    
    self.post_attention_layernorm.to("cuda")
    hidden_states = [self.post_attention_layernorm(x) for x in hidden_states]
    self.post_attention_layernorm.to("meta")

    hidden_states = self.mlp(hidden_states)
    self.mlp.to("meta")
    hidden_states = [(residual.pop(0) + hidden_states.pop(0)) for _ in range(len(residual))]

    outputs = (hidden_states,)

    if output_attentions: raise
    if use_cache: raise

    return outputs


def computeAttention(
    rotary_emb,
    num_heads,
    head_dim,
    num_key_value_heads,
    num_key_value_groups,
    attention_dropout,
    training,
    hidden_size,
    query_states,
    key_states,
    value_states,
    bsz,
    q_len,
    position_ids,
    attention_mask
):
    query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    cos, sin = rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, num_key_value_groups)
    value_states = repeat_kv(value_states, num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)

    if attn_weights.size() != (bsz, num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )

        attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = torch.nn.functional.dropout(attn_weights, p=attention_dropout, training=training)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, num_heads, q_len, head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, num_heads, q_len, head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, hidden_size)
    
    del query_states, key_states, value_states, cos, sin, attn_weights
    torch.cuda.empty_cache()
    
    return attn_output


def Qwen2Attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask = None,
    position_ids = None,
    past_key_value = None,
    output_attentions = False,
    use_cache = False,
    **kwargs,
):
    if "padding_mask" in kwargs: raise
    if past_key_value is not None: raise

    bszList, qLenList = [], []
    for x in hidden_states:
        bsz, q_len, _ = x.size()
        bszList.append(bsz)
        qLenList.append(q_len)

    self.q_proj.to("cuda")
    query_statesList = self.q_proj(hidden_states, popInputs = False)
    self.q_proj.to("meta")

    self.k_proj.to("cuda")
    key_statesList = self.k_proj(hidden_states, popInputs = False)
    self.k_proj.to("meta")

    self.v_proj.to("cuda")
    value_statesList = self.v_proj(hidden_states)
    self.v_proj.to("meta")

    self.rotary_emb.to("cuda")
    attn_output = []
    for (bsz, q_len, mask, pos_ids) in zip(bszList, qLenList, attention_mask, position_ids):
        attn_output.append(computeAttention(
            self.rotary_emb,
            self.num_heads,
            self.head_dim,
            self.num_key_value_heads,
            self.num_key_value_groups,
            self.attention_dropout,
            self.training,
            self.hidden_size,
            query_statesList.pop(0),
            key_statesList.pop(0),
            value_statesList.pop(0),
            bsz,
            q_len,
            pos_ids,
            mask
        ))
    self.rotary_emb.to("meta")

    self.o_proj.to("cuda")
    attn_output = self.o_proj(attn_output)
    self.o_proj.to("meta")

    if output_attentions: raise
    else: attn_weights = None

    return attn_output, attn_weights, past_key_value


def Qwen2MLP_forward(self, listX):
    self.gate_proj.to("cuda")
    listG = self.gate_proj(listX, popInputs = False)
    self.gate_proj.to("meta")
    
    self.up_proj.to("cuda")
    listU = self.up_proj(listX)
    self.up_proj.to("meta")
    
    listX = [self.act_fn(listG.pop(0)) * listU.pop(0) for _ in range(len(listU))]
    torch.cuda.empty_cache()

    self.down_proj.to("cuda")
    listX = self.down_proj(listX)
    self.down_proj.to("meta")

    return listX


forwardMap = (
    (Qwen2ForCausalLM, Qwen2ForCausalLM_forward),
    (Qwen2Model, Qwen2Model_forward),
    (Qwen2DecoderLayer, Qwen2DecoderLayer_forward),
    (Qwen2Attention, Qwen2Attention_forward),
    (Qwen2MLP, Qwen2MLP_forward)
)

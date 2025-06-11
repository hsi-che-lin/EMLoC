from transformers.modeling_outputs import BaseModelOutputWithPooling, BaseModelOutput

from internvl.model.internvl_chat.modeling_intern_vit import (InternVisionModel,
                                                              InternVisionEncoder,
                                                              InternVisionEncoderLayer,
                                                              InternAttention,
                                                              InternMLP)


def InternVisionModel_forward(
    self,
    pixel_values = None,
    output_hidden_states = None,
    return_dict = None,
    pixel_embeds = None,
):
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if pixel_values is None and pixel_embeds is None:
        raise ValueError('You have to specify pixel_values or pixel_embeds')

    if pixel_embeds is not None:
        hidden_states = pixel_embeds
    else:
        self.embeddings.to("cuda")
        hidden_states = [self.embeddings(pixel_values.pop(0).to("cuda")) for _ in range(len(pixel_values))]
        self.embeddings.to("meta")
    
    encoder_outputs = self.encoder(
        inputs_embeds=hidden_states,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    if not return_dict:
        raise

    return BaseModelOutputWithPooling(
        last_hidden_state=encoder_outputs.last_hidden_state,
        hidden_states=encoder_outputs.hidden_states,
        attentions=encoder_outputs.attentions,
    )
    

def InternVisionEncoder_forward(self, inputs_embeds, output_hidden_states = None, return_dict = None):
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    encoder_states = () if output_hidden_states else None
    hidden_states = inputs_embeds

    for idx, encoder_layer in enumerate(self.layers):
        if output_hidden_states: raise
        if self.gradient_checkpointing and self.training:
            raise
        else:
            hidden_states = encoder_layer(hidden_states)
            encoder_layer.to("meta")
            
    if output_hidden_states: raise
    if not return_dict: raise

    return BaseModelOutput(
        last_hidden_state=hidden_states, hidden_states=encoder_states
    )


def InternVisionEncoderLayer_forward(self, hidden_states):
    self.ls1.data = self.ls1.data.to("cuda")
    self.ls2.data = self.ls2.data.to("cuda")
    self.norm1.to("cuda")
    self.norm2.to("cuda")
    dtype = hidden_states[0].dtype
    
    residual = hidden_states
    hidden_states = [self.norm1(x).to(dtype) for x in hidden_states]
    hidden_states = [x * self.ls1 for x in self.attn(hidden_states)]
    hidden_states = [self.drop_path1(x) for x in hidden_states]
    hidden_states = [(residual.pop(0) + hidden_states.pop(0)) for _ in range(len(residual))]

    residual = hidden_states
    hidden_states = [self.norm2(x).to(dtype) for x in hidden_states]
    hidden_states = [x * self.ls2 for x in self.mlp(hidden_states)]
    hidden_states = [self.drop_path2(x) for x in hidden_states]
    hidden_states = [(residual.pop(0) + hidden_states.pop(0)) for _ in range(len(residual))]

    return hidden_states


def computeAttention(num_heads, scale, attn_drop, qkv, shape, q_norm, k_norm):
    B, N, C = shape
    qkv = qkv.reshape(B, N, 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

    if (q_norm is not None):
        B_, H_, N_, D_ = q.shape
        q = q_norm(q.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
        k = k_norm(k.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)

    attn = ((q * scale) @ k.transpose(-2, -1))
    attn = attn.softmax(dim=-1)
    attn = attn_drop(attn)
    x = (attn @ v).transpose(1, 2).reshape(B, N, C)

    return x


def InternAttention_forward(self, listX):
    shapes = [x.shape for x in listX]

    self.qkv.to("cuda")
    listX = self.qkv(listX)
    self.qkv.to("meta")

    if self.qk_normalization:
        q_norm = self.q_norm.to("cuda")
        k_norm = self.k_norm.to("cuda")
    else:
        q_norm = None
        k_norm = None
    
    for shape in shapes:
        listX.append(computeAttention(
            self.num_heads,
            self.scale,
            self.attn_drop,
            listX.pop(0),
            shape,
            q_norm,
            k_norm
        ))
    
    if self.qk_normalization:
        self.q_norm.to("meta")
        self.k_norm.to("meta")
    
    self.proj.to("cuda")
    listX = self.proj(listX)
    self.proj.to("meta")
    
    listX = [self.proj_drop(x) for x in listX]
    
    return listX
    

def InternMLP_forward(self, listX):
    self.fc1.to("cuda")
    listX = self.fc1(listX)
    self.fc1.to("meta")

    listX = [self.act(listX.pop(0)) for _ in range(len(listX))]
    
    self.fc2.to("cuda")
    listX = self.fc2(listX)
    self.fc2.to("meta")
    
    return listX
    

forwardMap = (
    (InternVisionModel, InternVisionModel_forward),
    (InternVisionEncoder, InternVisionEncoder_forward),
    (InternVisionEncoderLayer, InternVisionEncoderLayer_forward),
    (InternAttention, InternAttention_forward),
    (InternMLP, InternMLP_forward)
)
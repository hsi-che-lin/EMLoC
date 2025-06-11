import torch
import os
from diffusers import FluxPipeline

if (__name__ == "__main__"):
    path = "trained-flux"
    prompt = "a sks dog in the jungle"
    outDir = "result"
    os.makedirs(outDir, exist_ok = True)

    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16
    )
    pipe.load_lora_weights(path)
    pipe.enable_model_cpu_offload()

    for j in range(6):
        image = pipe(
            prompt,
            height=512,
            width=512,
            guidance_scale=3.5,
            num_inference_steps=50,
            max_sequence_length=64,
            generator=torch.Generator("cpu").manual_seed(j)
        ).images[0]
        image.save(f"{outDir}/{j}.png")

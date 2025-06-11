from safetensors.torch import load_file, save_file
from internvl.model.svd_internvl_chat import SVDInternVLChatModel
from internvl.utils.utils import rgetattr
import torch
from tqdm import tqdm


def clamp(B_, correction, L):
    b = torch.abs(B_).mean(dim = 1, keepdim = True)
    c = torch.abs(correction).mean(dim = 1, keepdim = True)
    ratio = c / b
    scale = torch.where(ratio > L, L / ratio, torch.ones_like(ratio))
    B_ = B_ - correction * scale
    
    return B_


@torch.no_grad()
def getCorrectedLoRA(A, B, svdModule, fullModule, L):
    A = A.to("cuda").transpose(0, 1)
    B = B.to("cuda").transpose(0, 1)
    svdModule.to("cuda")
    fullModule.to("cuda")

    A_, Sa, Va = torch.linalg.svd(A.float(), full_matrices = False)
    B_ = Sa.unsqueeze(1) * torch.matmul(Va, B.float())
    A_ = A_.to(torch.bfloat16)
    B_ = B_.to(torch.bfloat16)
    correction = fullModule(A_.transpose(0, 1)) - svdModule(A_.transpose(0, 1))
    
    B_ = clamp(B_, correction, L)

    return A_.transpose(0, 1), B_.transpose(0, 1)


def loraCorrection(svdModelPath, loraPath, L = 3):
    fullModelPath = "pretrained/InternVL2_5-8B"
    
    lora = load_file(loraPath)
    svdModel = SVDInternVLChatModel.from_pretrained(svdModelPath, torch_dtype = torch.bfloat16, low_cpu_mem_usage = True)
    fullModel = SVDInternVLChatModel.from_pretrained(fullModelPath, torch_dtype = torch.bfloat16, low_cpu_mem_usage = True)
    modules = []

    for k in lora.keys():
        name = ".".join(k.split(".")[2:-2])
        if (name not in modules): modules.append(name)

    for name in tqdm(modules):
        svdModule =  rgetattr(svdModel, name)
        fullModule = rgetattr(fullModel, name)
        A = lora[f"base_model.model.{name}.lora_A.weight"]
        B = lora[f"base_model.model.{name}.lora_B.weight"]
        A, B = getCorrectedLoRA(A, B, svdModule, fullModule, L)
        lora[f"base_model.model.{name}.lora_A.weight"] = A.contiguous()
        lora[f"base_model.model.{name}.lora_B.weight"] = B.contiguous()
        svdModule.to("meta")
        fullModule.to("meta")

    save_file(lora, loraPath)

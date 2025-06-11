import torch
import logging

from internvl.model.svd_internvl_chat.svd_modules import SVDLinear


logger = logging.getLogger(__name__)


@torch.no_grad()
def getXtX(inputs):
    XtX = 0

    for x in inputs:
        x = x.reshape(-1, x.shape[-1]).to(dtype = torch.float)
        XtX += torch.matmul(x.transpose(0, 1), x)

    return XtX


@torch.no_grad()
def getScaling(XtX):
    factor = torch.trace(XtX) / XtX.shape[0]
    success = False
    retry = 1
    eps = 1e-7
    
    while (not success):
        try:
            L, Q = torch.linalg.eigh(XtX / factor)
            L = torch.clamp(L, 1e-7) * factor
            success = True
        except:
            if (retry == 5): raise
            retry += 1
            XtX += torch.eye(XtX.shape[0], device = XtX.device) * eps
            eps *= 5
    
    S = Q * torch.sqrt(L)
    
    return S


@torch.no_grad()
def SVD(W, r, scaling = None):
    if (scaling is not None):
        scaling = scaling.to(W.device)
        W = torch.matmul(W, scaling)
        scalingInv = torch.linalg.inv(scaling)
    
    U, S, VT = torch.linalg.svd(W, full_matrices = False)
    truc_s = S[:r]
    truc_s = torch.diag(truc_s)
    sqrtSigma = torch.sqrt(truc_s)
    truc_v = VT[:r, :]
    truc_v = torch.matmul(truc_v, scalingInv) if (scaling is not None) else truc_v
    truc_u = U[:, :r]
    Wv = torch.matmul(sqrtSigma, truc_v)
    Wu = torch.matmul(truc_u, sqrtSigma)
    
    return Wv, Wu


class TwoPathLinear(torch.nn.Module):
    def __init__(self, name, linear, state, ratio):
        super().__init__()
        self.name = name
        self.origLinear = linear
        self.hasBias = (linear.bias is not None)
        self.state = state
        self.ratio = ratio
        self.svd = None


    def normal_forward(self, inputs, popInputs = True):
        if (popInputs):
            return [self.origLinear(inputs.pop(0)) for _ in range(len(inputs))]
        else:
            return [self.origLinear(x) for x in inputs]
    
    
    def forward(self, inputs, popInputs = True):
        if (self.ratio is None):
            return self.normal_forward(inputs)
        
        logger.info(self.name)
        XtX = getXtX(inputs)
        self.initSVD(XtX)
        
        return self.normal_forward(inputs, popInputs)

    
    def initSVD(self, XtX):
        dtype = self.origLinear.weight.dtype

        self.svd = SVDLinear(
            ratio = self.ratio,
            in_features = self.origLinear.in_features,
            out_features = self.origLinear.out_features,
            bias = self.hasBias,
            dtype = dtype
        )
    
        W = self.origLinear.weight.data.to(dtype = torch.float)
        
        scaling = getScaling(XtX)
        Wv, Wu = SVD(W, self.svd.rank, scaling)
        self.svd.v.weight.data = Wv.to(dtype = dtype)
        self.svd.u.weight.data = Wu.to(dtype = dtype)
        if (self.hasBias): self.svd.u.bias.data = self.origLinear.bias.data
        
        for (k, v) in self.svd.state_dict().items():
            self.state[f"{self.name}.{k}"] = v.to("cpu")

        self.svd.to("meta")

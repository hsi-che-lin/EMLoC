import torch
import logging
from torch import nn


logger = logging.getLogger(__name__)


class SVDLinear(nn.Module):
    def __init__(
        self,
        ratio,
        in_features = None,
        out_features = None,
        bias = False,
        dtype = None,
        device = None,
        refLinear = None
    ):
        super().__init__()

        if (refLinear is not None):
            assert (isinstance(refLinear, nn.Linear))
            in_features = refLinear.in_features
            out_features = refLinear.out_features
            bias = (refLinear.bias is not None)
            dtype = refLinear.weight.dtype
            device = refLinear.weight.device
        else:
            assert (isinstance(in_features, int) and isinstance(out_features, int))

        self.in_features = in_features
        self.out_features = out_features
        self.rank = int(self.in_features * self.out_features * ratio / (self.in_features + self.out_features))
        self.v = nn.Linear(self.in_features, self.rank, bias = False, device = device, dtype = dtype)
        self.u = nn.Linear(self.rank, self.out_features, bias = bias, device = device, dtype = dtype)
    

    def get_decompose_result(self, refLinear, scaling = None):
        W = refLinear.weight.data
        deviceDtype = {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "dtype": torch.float32
        }
        W = W.to(**deviceDtype)
        
        if (scaling is not None):
            scaling = scaling.to(**deviceDtype)
            W = torch.matmul(W, scaling)
        
        U, S, VT = torch.linalg.svd(W, full_matrices=False)
        S = S[:self.rank]
        U = U[:, :self.rank]
        VT = VT[:self.rank, :]

        return U, S, VT, scaling
    
    
    def init_weights_from_decompose_results(self, U, S, VT, bias, dtype, scaling = None):
        assert (len(S) >= self.rank)
        deviceDtype = {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "dtype": torch.float32
        }
        U = U.to(**deviceDtype)
        S = S.to(**deviceDtype)
        VT = VT.to(**deviceDtype)
        
        if (scaling is not None):
            scaling = scaling.to(**deviceDtype)
            try:
                scalingInv = torch.linalg.inv(scaling)
            except:
                logging.warning("scaling_diag_matrix is not full rank!")
                scaling += 1e-6 * torch.eye(scaling.shape[0], **deviceDtype)
                scalingInv = torch.linalg.inv(scaling)
        
        truc_s = S[:self.rank]
        truc_u = U[:, :self.rank]
        truc_v = VT[:self.rank, :]
        truc_v = torch.matmul(truc_v, scalingInv) if (scaling is not None) else truc_v
        truc_s = torch.diag(truc_s)
        sqrtSigma = torch.sqrt(truc_s)
        self.u.weight.data = torch.matmul(truc_u, sqrtSigma).to(device = "cpu", dtype = dtype)
        self.v.weight.data = torch.matmul(sqrtSigma, truc_v).to(device = "cpu", dtype = dtype)
        if (bias is not None):
            self.u.bias.data = bias.data

    
    def init_weights_from_ref(self, refLinear, scaling = None):
        dtype = refLinear.weight.data.dtype
        U, S, VT, scaling = self.get_decompose_result(refLinear, scaling)
        self.init_weights_from_decompose_results(U, S, VT, refLinear.bias, dtype, scaling)
     
    
    def forward(self, x):
        return self.u(self.v(x))

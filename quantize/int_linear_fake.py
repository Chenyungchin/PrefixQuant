import torch
import torch.nn as nn
import torch.nn.functional as F
from quantize.quantizer import UniformAffineQuantizer
import utils.hadamard_utils as hadamard_utils
import matplotlib.pyplot as plt





class QuantLinear(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    def __init__(
        self,
        org_module: nn.Linear,
        name: str = "", # jim: added to register the name
    ):
        super().__init__()
        self.fwd_kwargs = dict()
        self.fwd_func = F.linear
        self.register_parameter('weight',org_module.weight) # trainable
        if org_module.bias is not None:
            self.register_buffer('bias',org_module.bias)
        else:
            self.bias = None
        self.in_features = org_module.in_features
        self.out_features = org_module.out_features
        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False
        self.wbits = 16
        self.input_bits = 16
        self.output_bits = 16
        self.online_full_had=False
        self.use_temporary_parameter=False
        self.name = name # jim: added to register the name

    
    
    def forward(self, input: torch.Tensor):
        input_dtype = input.dtype

        # # jim: inspect layer 1 down projection input
        # if self.name == "model.layers.1.mlp.down_proj":
        #     if getattr(self, 'eval_mode', False):
        #         # plot token-wise max value
        #         token_max = input.amax(dim=(0, -1)).cpu().numpy()
        #         plt.figure(figsize=(12, 6))
        #         plt.plot(token_max, linewidth=1)
        #         plt.title("Token-wise Max Value before Layer 1 Down Projection")
        #         plt.xlabel("Token Index")
        #         plt.ylabel("Max Value")
        #         plt.savefig("plt/token_max_before_layer1_down_proj.png")
        #         breakpoint()
        if getattr(self, 'outlier_token_detection', False):
            activation_abs = input.abs()
            activation_abs = activation_abs.max(dim=-1).values
            ratio = activation_abs / activation_abs.median()
            outlier_token_ids = (ratio > self.outlier_threshold)
            # Share outlier_token_ids via the shared state container
            if hasattr(self, 'outlier_info'):
                self.outlier_info['outlier_token_ids'] = outlier_token_ids

        # Rotate, if needed
        if self.online_full_had:
            if self.fp32_had: # Full Hadamard in FP32
                input = hadamard_utils.matmul_hadU_cuda(input.float(), self.had_K, self.K).to(input_dtype)
            else: # Full Hadamard in FP16
                input = hadamard_utils.matmul_hadU_cuda(input, self.had_K, self.K)
                
        if self.use_temporary_parameter:
            weight = self.temp_weight
        else:
            weight = self.weight

        bias = self.bias
            
        if self.use_weight_quant and self.wbits < 16:
            weight = self.weight_quantizer(weight)

        if self.use_act_quant and self.input_bits < 16:
            input = self.input_quantizer(input)

        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)

        # jim: add residual template token
        if getattr(self, 'eval_mode', False) and hasattr(self, 'residual_template_token'):
            breakpoint()
            # jim: out shape: (batch_size, seq_len, dim)
            # Get the online-detected outlier token ids from shared state
            outlier_token_ids = None
            if hasattr(self, 'outlier_info') and 'outlier_token_ids' in self.outlier_info:
                outlier_token_ids = self.outlier_info['outlier_token_ids']
            # use outlier_token_ids and residual_template_token to modify out
            if outlier_token_ids is not None:
                out[outlier_token_ids] += self.residual_template_token

        if self.use_act_quant and self.output_bits < 16:
            out = self.output_quantizer(out)

        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant





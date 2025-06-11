import torch.nn as nn
import torch.nn.functional as F
import torch
from math import sqrt
from ukan_archs import *
import os
import sys
sys.path.append('/home/ps/torch-conv-kan')

import  kan_convs

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    if classname.find("Convx") != -1:
        torch.nn.init.normal_(m.conv.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

##############################

#           generator: DnCNN

##############################


##############################

#        Discriminator: PatchGAN

##############################





class Discriminator(nn.Module):
    def __init__(self, in_channels= 1 ):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(			
            *discriminator_block(in_channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img):
        img_input = img
        return self.model(img_input)


## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange, repeat
from functools import partial
from timm.models.layers import DropPath, trunc_normal_

import math
import copy
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count

# from selective_scan import selective_scan_fn as selective_scan_fn_v1

######v1 selective_scan_fn_v1

import selective_scan_cuda_core as selective_scan_cuda


class SelectiveScanFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1):
        # input_t: float, fp16, bf16; weight_t: float;
        # u, B, C, delta: input_t
        # D, delta_bias: float
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if B.dim() == 3:
            B = rearrange(B, "b dstate l -> b 1 dstate l")
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = rearrange(C, "b dstate l -> b 1 dstate l")
            ctx.squeeze_C = True
        if D is not None and (D.dtype != torch.float):
            ctx._d_dtype = D.dtype
            D = D.float()
        if delta_bias is not None and (delta_bias.dtype != torch.float):
            ctx._delta_bias_dtype = delta_bias.dtype
            delta_bias = delta_bias.float()

        assert u.shape[1] % (B.shape[1] * nrows) == 0
        assert nrows in [1, 2, 3, 4]  # 8+ is too slow to compile

        out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)
        ctx.delta_softplus = delta_softplus
        ctx.nrows = nrows
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
            # u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, ctx.nrows,
        )
        dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC

        _dD = None
        if D is not None:
            if dD.dtype != getattr(ctx, "_d_dtype", dD.dtype):
                _dD = dD.to(ctx._d_dtype)
            else:
                _dD = dD

        _ddelta_bias = None
        if delta_bias is not None:
            if ddelta_bias.dtype != getattr(ctx, "_delta_bias_dtype", ddelta_bias.dtype):
                _ddelta_bias = ddelta_bias.to(ctx._delta_bias_dtype)
            else:
                _ddelta_bias = ddelta_bias

        return (du, ddelta, dA, dB, dC, _dD, _ddelta_bias, None, None)


def selective_scan_fn_v1(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1):
    """if return_last_state is True, returns (out, last_state)
    last_state has shape (batch, dim, dstate). Note that the gradient of the last state is
    not considered in the backward pass.
    """
    return SelectiveScanFn.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)


# fvcore flops =======================================

def flops_selective_scan_fn(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu]
    """
    assert not with_complex
    # https://github.com/state-spaces/mamba/issues/110
    flops = 9 * B * L * D * N
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    return flops


def print_jit_input_names(inputs):
    print("input params: ", end=" ", flush=True)
    try:
        for i in range(10):
            print(inputs[i].debugName(), end=" ", flush=True)
    except Exception as e:
        pass
    print("", flush=True)


def selective_scan_flop_jit(inputs, outputs):
    print_jit_input_names(inputs)
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_selective_scan_fn(B=B, L=L, D=D, N=N, with_D=True, with_Z=False, with_Group=True)
    return flops


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
import torch
import torch.nn as nn
import torch.nn.functional as F



class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class SS2D_1(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            simple_init=False,
            # ======================
            softmax_version=False,
            forward_type="v2",
            # ======================
            **kwargs,
    ):
        """
        ssm_rank_ratio would be used in the future...
        """
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        # d_model model dim
        d_expand = int(ssm_ratio * d_model)
        # d_inner  dim in model, for channel it should be 2
        d_inner = int(min(ssm_rank_ratio, ssm_ratio) * d_model) if ssm_rank_ratio > 0 else d_expand
        self.softmax_version = softmax_version
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_state = math.ceil(d_model / 6) if d_state == "auto" else d_state  # 20240109
        self.d_conv = d_conv

        # cccc
        dc_inner = 4
        self.dtc_rank = 6  # 6
        self.dc_state = 16  # 16
        # self.conv_innerc =  nn.Conv2d(in_channels=1, out_channels=dc_inner, kernel_size=1, stride=1, padding=0)
        # self.conv_innerc = nn.Linear(1, dc_inner, bias=bias, **factory_kwargs)
        self.conv_cin = nn.Conv2d(in_channels=1, out_channels=dc_inner, kernel_size=1, stride=1, padding=0)
        self.conv_cout = nn.Conv2d(in_channels=dc_inner, out_channels=1, kernel_size=1, stride=1, padding=0)
        # self.conv_outc = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.forward_core = self.forward_corev1

        self.K = 4 if forward_type not in ["share_ssm"] else 1
        self.K2 = self.K if forward_type not in ["share_a"] else 1
        self.KC = 2
        self.K2C = self.KC if forward_type not in ["share_a"] else 1

        self.cforward_core = self.cforward_corev1
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.channel_norm = LayerNorm(d_inner, LayerNorm_type='WithBias')

        # in proj =======================================
        self.in_conv = nn.Conv2d(in_channels=d_model, out_channels=d_expand * 2, kernel_size=1, stride=1, padding=0)
        self.act: nn.Module = act_layer()

        # conv =======================================
        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=d_expand,
                out_channels=d_expand,
                groups=d_expand,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        self.out_norm = LayerNorm(d_inner, LayerNorm_type='WithBias')

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj
        # xc proj ============================
        self.xc_proj = [
            nn.Linear(dc_inner, (self.dtc_rank + self.dc_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.KC)
        ]
        self.xc_proj_weight = nn.Parameter(torch.stack([tc.weight for tc in self.xc_proj], dim=0))  # (K, N, inner)
        del self.xc_proj

        # dt proj ============================
        self.dt_projs = [
            self.dt_init(self.dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K, inner)
        del self.dt_projs

        # A, D =======================================
        self.A_logs = self.A_log_init(self.d_state, d_inner, copies=self.K2, merge=True)  # (K * D, N)
        self.Ds = self.D_init(d_inner, copies=self.K2, merge=True)  # (K * D)

        # out proj =======================================
        self.out_conv = nn.Conv2d(in_channels=d_expand, out_channels=d_model, kernel_size=1, stride=1, padding=0)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        self.Dsc = nn.Parameter(torch.ones((self.K2C * dc_inner)))
        self.Ac_logs = nn.Parameter(
            torch.randn((self.K2C * dc_inner, self.dc_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
        self.dtc_projs_weight = nn.Parameter(torch.randn((self.KC, dc_inner, self.dtc_rank)).contiguous())
        self.dtc_projs_bias = nn.Parameter(torch.randn((self.KC, dc_inner)))

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev1(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W

        def cross_scan_2d(x):
            # (B, C, H, W) => (B, K, C, H * W) with K = len([HW, WH, FHW, FWH])
            x_hwwh = torch.stack([x.flatten(2, 3), x.transpose(dim0=2, dim1=3).contiguous().flatten(2, 3)],
                                 dim=1)  # 一个h,w展开，一个w,h展开，然后堆在一起
            xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l) #把上面那俩再翻译下，然后堆在一起
            return xs

            # 四个方向

        if self.K == 4:
            # K = 4
            xs = cross_scan_2d(x)  # (b, k, d, l) #[batch_size, 4, channels, height * width]

            # print("x shape", x.shape) # 8,96,128,128
            # print("xs shape", xs.shape) # 8,4,96, 16384
            # print("Ac_logs shape", self.A_logs.shape) #384,16
            # print("dtc_projs_weight shape", self.dt_projs_weight.shape) #4,96,6
            # print("xc_proj_weight shape", self.x_proj_weight.shape) #4,38,96
            # print("Dsc shape", self.Ds.shape) # 384
            # print("dtc_projs_bias shape", self.dt_projs_bias.shape) #4,96

            x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
            # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
            dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
            dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

            xs = xs.view(B, -1, L)  # (b, k * d, l)
            dts = dts.contiguous().view(B, -1, L)  # (b, k * d, l)
            As = -torch.exp(self.A_logs.float())  # (k * d, d_state)
            Ds = self.Ds  # (k * d)
            dt_projs_bias = self.dt_projs_bias.view(-1)  # (k * d)

            # assert len(xs.shape) == 3 and len(dts.shape) == 3 and len(Bs.shape) == 4 and len(Cs.shape) == 4
            # assert len(As.shape) == 2 and len(Ds.shape) == 1 and len(dt_projs_bias.shape) == 1
            # print(self.Ds.dtype, self.A_logs.dtype, self.dt_projs_bias.dtype, flush=True) # fp16, fp16, fp16

            # print("xs shape", xs.shape) #8,768,1024
            # print("dts shape", dts.shape) #8, 768, 1024
            # print("As shape", As.shape)#768, 16
            # print("Bs shape", Bs.shape)#8, 4, 16, 1024
            # print("Cs shape", Cs.shape)#8, 4, 16, 1024
            # print("Ds shape", Ds.shape)#768
            # print("dt_projs_bias shape", dt_projs_bias.shape) #768
            out_y = self.selective_scan(
                xs, dts,
                As, Bs, Cs, Ds,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
            ).view(B, 4, -1, L)
            # assert out_y.dtype == torch.float16

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0].float() + inv_y[:, 0].float() + wh_y.float() + invwh_y.float()

        y = y.view(B, C, H, W)
        y = self.out_norm(y).to(x.dtype)

        return y

    def cforward_corev1(self, xc: torch.Tensor):
        self.selective_scanC = selective_scan_fn_v1

        b, d, h, w = xc.shape

        # xc = self.pooling(xc).squeeze(-1).permute(0,2,1).contiguous() #b,1,d, >1!
        # print("xc shape", xc.shape) # 8,1,96
        xc = self.pooling(xc)  # b,d,1,1
        xc = xc.permute(0, 2, 1, 3).contiguous()  # b,1,d,1
        xc = self.conv_cin(xc)  # b,4,d,1
        xc = xc.squeeze(-1)  # b,4,d

        # xc = xc.permute(0,2,1).contiguous()
        # xc = self.conv_innerc(xc)
        # xc = xc.permute(0,2,1).contiguous()

        B, D, L = xc.shape  # b,1,c
        D, N = self.Ac_logs.shape  # 2,16
        K, D, R = self.dtc_projs_weight.shape  # 2,1,6

        # print("Ac_logs shape", self.Ac_logs.shape) #2,16
        # print("dtc_projs_weight shape", self.dtc_projs_weight.shape) #2,1,6
        # print("xc_proj_weight shape", self.xc_proj_weight.shape) #2,38,1
        # print("Dsc shape", self.Dsc.shape) # 2
        # print("dtc_projs_bias shape", self.dtc_projs_bias.shape) #2,1

        xsc = torch.stack([xc, torch.flip(xc, dims=[-1])], dim=1)  # input:b,d,l output:b,2,d,l
        # print("xsc shape", xsc.shape) # 8,2,1,96

        xc_dbl = torch.einsum("b k d l, k c d -> b k c l", xsc, self.xc_proj_weight)  # 8,2,1,96; 2,38,1 ->8,2,38,96

        dts, Bs, Cs = torch.split(xc_dbl, [self.dtc_rank, self.dc_state, self.dc_state], dim=2)  # 8,2,38,96-> 6,16,16
        # dts:8,2,6,96 bs,cs:8,2,16,96
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dtc_projs_weight).contiguous()

        xsc = xsc.view(B, -1, L)  # (b, k * d, l) 8,2,96
        dts = dts.contiguous().view(B, -1, L).contiguous()  # (b, k * d, l) 8,2,96
        As = -torch.exp(self.Ac_logs.float())  # (k * d, d_state) 2,16
        Ds = self.Dsc  # (k * d) 2
        dt_projs_bias = self.dtc_projs_bias.view(-1)  # (k * d)2

        # assert len(xs.shape) == 3 and len(dts.shape) == 3 and len(Bs.shape) == 4 and len(Cs.shape) == 4
        # assert len(As.shape) == 2 and len(Ds.shape) == 1 and len(dt_projs_bias.shape) == 1
        # print(self.Ds.dtype, self.A_logs.dtype, self.dt_projs_bias.dtype, flush=True) # fp16, fp16, fp16

        # print("channel xs shape", xsc.shape) #8, 2, 192
        # print("channel dts shape", dts.shape)#8, 2, 192
        # print("channel As shape", As.shape)#2, 16
        # print("channel Bs shape", Bs.shape)#8, 2, 16, 192
        # print("channel Cs shape", Cs.shape)#8, 2, 16, 192
        # print("channel Ds shape", Ds.shape)#2
        # print("channel dt_projs_bias shape", dt_projs_bias.shape) #2
        out_y = self.selective_scanC(
            xsc, dts,
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, 2, -1, L)

        y = out_y[:, 0].float() + torch.flip(out_y[:, 1], dims=[-1]).float()
        # y = xsc[:, 0].float() + torch.flip(xsc[:, 1], dims=[-1]).float()

        # y: b,4,d
        y = y.unsqueeze(-1)  # b,4,d,1
        y = self.conv_cout(y)  # b,1,d,1
        y = y.transpose(dim0=1, dim1=2).contiguous()  # b,d,1,1
        y = self.channel_norm(y)
        y = y.to(xc.dtype)

        # y = y.transpose(dim0=1, dim1=2).contiguous().unsqueeze(-1).contiguous()
        # y = self.channel_norm(y)
        # y = y.to(xc.dtype)

        # y = y.permute(0,2,1,3).contiguous()
        # y = self.conv_outc(y)
        # y = y.permute(0,2,1,3).contiguous()

        return y

    def forward(self, x: torch.Tensor, **kwargs):
        # input: b,d,h,w
        # output: b,d,h,w
        xz = self.in_conv(x)
        x, z = xz.chunk(2, dim=1)  # (b, d, h, w)
        if not self.softmax_version:
            z = self.act(z)
        x = self.act(self.conv2d(x))  # (b, d, h, w)
        y1 = self.forward_core(x)
        y2 = y1 * z
        c = self.cforward_core(y2)  # x:b,d,h,w; output:b,d,1,1
        y2 = y2 + c
        out = self.out_conv(y2)
        return out

##########################################################################
class MamberBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(MamberBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = SS2D_1(d_model=dim, ssm_ratio=1)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)



##########################################################################
##---------- Mamber -----------------------

class Mamber32(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 out_channels=64,
                 dim=48,
                 num_blocks=[6, 6, 7, 8],
                 num_refinement_blocks=2,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 dual_pixel_task=True  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
                 ):

        super(Mamber32, self).__init__()

        # self.scale = scale

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim - 10)

        self.encoder_level1 = nn.Sequential(*[
            MamberBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                        LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[
            MamberBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                        LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[
            MamberBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                        LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[
            MamberBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                        LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            MamberBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                        LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            MamberBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                        LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[
            MamberBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                        LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[
            MamberBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                        LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        ###for super-resolution, upsampler
        # modules_tail = [common.Upsampler(common.default_conv, 4, int(dim*2**1), act=False),common.default_conv(int(dim*2**1), out_channels, 3)]
        # self.tail = nn.Sequential(*modules_tail)

        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        ###########################

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.out_dim = out_channels

    def forward(self, inp_img, ker_code):
        B, C, H, W = inp_img.size()  # I_LR batch
        B_h, C_h = ker_code.size()  # Batch, Len=10
        ker_code_exp = ker_code.view((B_h, C_h, 1, 1)).expand(
            (B_h, C_h, H, W)
        )  # kernel_map stretch

        inp_enc_level1 = self.patch_embed(inp_img)
        inp_enc_level1 = torch.cat([inp_enc_level1, ker_code_exp], dim=1)

        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            # print(out_dec_level1.shape)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1

    def flops(self, shape=(3, 256, 256)):
        # shape = self.__input_shape__[1:]
        supported_ops = {
            "aten::silu": None,  # as relu is in _IGNORED_OPS
            "aten::neg": None,  # as relu is in _IGNORED_OPS
            "aten::exp": None,  # as relu is in _IGNORED_OPS
            "aten::flip": None,  # as permute is in _IGNORED_OPS
            # "prim::PythonOp.CrossScan": None,
            # "prim::PythonOp.CrossMerge": None,
            "prim::PythonOp.SelectiveScan": selective_scan_flop_jit,
        }

        model = copy.deepcopy(self)
        model.cuda().eval()

        input = torch.randn((1, *shape), device=next(model.parameters()).device)
        params = parameter_count(model)[""]
        Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)

        del model, input
        # return sum(Gflops.values()) * 1e9
        return f"params(M) {params / 1e6} GFLOPs {sum(Gflops.values())}"



class Mamba_f(nn.Module):
    def __init__(self,
                 dim=64,
                 heads=1,
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
                 ):
        super(Mamba_f, self).__init__()

        self.encoder_inp1 = MamberBlock(dim=dim, num_heads=heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                        LayerNorm_type=LayerNorm_type)
        self.encoder_inp2 = MamberBlock(dim=dim, num_heads=heads, ffn_expansion_factor=ffn_expansion_factor,
                                        bias=bias, LayerNorm_type=LayerNorm_type)
        self.encoder_inp3 = MamberBlock(dim=dim, num_heads=heads, ffn_expansion_factor=ffn_expansion_factor,
                                        bias=bias, LayerNorm_type=LayerNorm_type)
        self.encoder_ref1 = MamberBlock(dim=dim, num_heads=heads, ffn_expansion_factor=ffn_expansion_factor,
                                        bias=bias, LayerNorm_type=LayerNorm_type)
        self.encoder_ref2 = MamberBlock(dim=dim, num_heads=heads, ffn_expansion_factor=ffn_expansion_factor,
                                        bias=bias, LayerNorm_type=LayerNorm_type)
        self.encoder_ref3 = MamberBlock(dim=dim, num_heads=heads, ffn_expansion_factor=ffn_expansion_factor,
                                        bias=bias, LayerNorm_type=LayerNorm_type)
        self.before_fusion1 = nn.Sequential(nn.Conv2d(dim + dim, dim, 1),
                                            nn.ReLU())

        self.encoder_fusion1 = MamberBlock(dim=dim, num_heads=heads, ffn_expansion_factor=ffn_expansion_factor,
                                           bias=bias, LayerNorm_type=LayerNorm_type)
        self.encoder_fusion2 = MamberBlock(dim=dim, num_heads=heads, ffn_expansion_factor=ffn_expansion_factor,
                                           bias=bias, LayerNorm_type=LayerNorm_type)

    def forward(self, inp_img, inp_ref):
        inp_c = torch.cat([inp_img, inp_ref], dim=1)
        inp_img1 = self.encoder_inp1(inp_img)
        inp_ref1 = self.encoder_ref1(inp_ref)
        inp_c = self.before_fusion1(inp_c)

        inp_c_1 = self.encoder_fusion1(inp_c)

        inp_img1 = inp_img1 * inp_c_1
        inp_ref1 = inp_ref1 * inp_c_1
        inp_img2 = self.encoder_inp2(inp_img1)
        inp_ref2 = self.encoder_ref2(inp_ref1)
        inp_c_2 = self.encoder_fusion2(inp_c_1)
        inp_img2 = inp_img2 * inp_c_2
        inp_ref2 = inp_ref2 * inp_c_2
        inp_img3 = self.encoder_inp3(inp_img2)
        inp_ref3 = self.encoder_ref3(inp_ref2)
        inp_img_out = inp_img + inp_img3
        inp_ref_out = inp_ref + inp_ref3
        inp_out = inp_img_out + inp_ref_out
        return inp_out


class Mamba_ets(nn.Module):
    def __init__(self,
        heads =1,
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',
                 in_nc=1, nf=64, num_blocks=5
                 ## Other option 'BiasFree'
      ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
    ):

        super(Mamba_ets, self).__init__()


        self.head_LR = nn.Conv2d(in_nc, nf // 2, 1, 1, 0)
        self.head_HR = nn.Conv2d(in_nc, nf // 2, kernel_size=9, padding=4)

        body = [MamberBlock(dim=nf, num_heads=heads, ffn_expansion_factor=ffn_expansion_factor,
                                        bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(num_blocks)]
        self.body = nn.Sequential(*body)

        self.out = nn.Conv2d(nf, 10, 3, 1, 1)
        self.globalPooling = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, LR, GT):
        # print('LR:',LR.shape)
        # print('GT:', GT.shape)
        lrf = self.head_LR(LR)
        hrf = self.head_HR(GT)
        hrf = nn.AdaptiveAvgPool2d((lrf.size(2), lrf.size(3)))(hrf)
        # print(lrf.shape,hrf.shape)
        f = torch.cat([lrf,hrf],dim=1)
        f = self.body(f)
        f = self.out(f)
        f = self.globalPooling(f)
        f = f.view(f.size()[:2])
        # print('ker map:', f.shape)
        return f


class DConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 groups=None, bias=True):
        super(DConv2d, self).__init__()

        # 如果未指定 groups，则默认为 in_channels（深度卷积）
        if groups is None:
            groups = in_channels

        # 深度卷积层 (Depthwise Convolution)，每个输入通道独立卷积
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=groups, bias=bias)

        # 逐点卷积层 (Pointwise Convolution)，1x1 卷积
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        # 深度卷积
        x = self.depthwise(x)
        # 逐点卷积
        x = self.pointwise(x)
        return x

class LFeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(LFeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = DConv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = DConv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = DConv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class LSS2D_1(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            simple_init=False,
            # ======================
            softmax_version=False,
            forward_type="v2",
            # ======================
            **kwargs,
    ):
        """
        ssm_rank_ratio would be used in the future...
        """
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        # d_model model dim
        d_expand = int(ssm_ratio * d_model)
        # d_inner  dim in model, for channel it should be 2
        d_inner = int(min(ssm_rank_ratio, ssm_ratio) * d_model) if ssm_rank_ratio > 0 else d_expand
        self.softmax_version = softmax_version
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_state = math.ceil(d_model / 6) if d_state == "auto" else d_state  # 20240109
        self.d_conv = d_conv

        # cccc
        dc_inner = 4
        self.dtc_rank = 6  # 6
        self.dc_state = 16  # 16
        # self.conv_innerc =  nn.Conv2d(in_channels=1, out_channels=dc_inner, kernel_size=1, stride=1, padding=0)
        # self.conv_innerc = nn.Linear(1, dc_inner, bias=bias, **factory_kwargs)
        self.conv_cin = DConv2d(in_channels=1, out_channels=dc_inner, kernel_size=1, stride=1, padding=0)
        self.conv_cout = DConv2d(in_channels=dc_inner, out_channels=1, kernel_size=1, stride=1, padding=0)
        # self.conv_outc = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.forward_core = self.forward_corev1

        self.K = 4 if forward_type not in ["share_ssm"] else 1
        self.K2 = self.K if forward_type not in ["share_a"] else 1
        self.KC = 2
        self.K2C = self.KC if forward_type not in ["share_a"] else 1

        self.cforward_core = self.cforward_corev1
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.channel_norm = LayerNorm(d_inner, LayerNorm_type='WithBias')

        # in proj =======================================
        self.in_conv = DConv2d(in_channels=d_model, out_channels=d_expand * 2, kernel_size=1, stride=1, padding=0)
        self.act: nn.Module = act_layer()

        # conv =======================================
        if self.d_conv > 1:
            self.conv2d = DConv2d(
                in_channels=d_expand,
                out_channels=d_expand,
                groups=d_expand,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                # **factory_kwargs,
            )

        self.out_norm = LayerNorm(d_inner, LayerNorm_type='WithBias')

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj
        # xc proj ============================
        self.xc_proj = [
            nn.Linear(dc_inner, (self.dtc_rank + self.dc_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.KC)
        ]
        self.xc_proj_weight = nn.Parameter(torch.stack([tc.weight for tc in self.xc_proj], dim=0))  # (K, N, inner)
        del self.xc_proj

        # dt proj ============================
        self.dt_projs = [
            self.dt_init(self.dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K, inner)
        del self.dt_projs

        # A, D =======================================
        self.A_logs = self.A_log_init(self.d_state, d_inner, copies=self.K2, merge=True)  # (K * D, N)
        self.Ds = self.D_init(d_inner, copies=self.K2, merge=True)  # (K * D)

        # out proj =======================================
        self.out_conv = DConv2d(in_channels=d_expand, out_channels=d_model, kernel_size=1, stride=1, padding=0)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        self.Dsc = nn.Parameter(torch.ones((self.K2C * dc_inner)))
        self.Ac_logs = nn.Parameter(
            torch.randn((self.K2C * dc_inner, self.dc_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
        self.dtc_projs_weight = nn.Parameter(torch.randn((self.KC, dc_inner, self.dtc_rank)).contiguous())
        self.dtc_projs_bias = nn.Parameter(torch.randn((self.KC, dc_inner)))

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev1(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W

        def cross_scan_2d(x):
            # (B, C, H, W) => (B, K, C, H * W) with K = len([HW, WH, FHW, FWH])
            x_hwwh = torch.stack([x.flatten(2, 3), x.transpose(dim0=2, dim1=3).contiguous().flatten(2, 3)],
                                 dim=1)  # 一个h,w展开，一个w,h展开，然后堆在一起
            xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l) #把上面那俩再翻译下，然后堆在一起
            return xs

            # 四个方向

        if self.K == 4:
            # K = 4
            xs = cross_scan_2d(x)  # (b, k, d, l) #[batch_size, 4, channels, height * width]

            # print("x shape", x.shape) # 8,96,128,128
            # print("xs shape", xs.shape) # 8,4,96, 16384
            # print("Ac_logs shape", self.A_logs.shape) #384,16
            # print("dtc_projs_weight shape", self.dt_projs_weight.shape) #4,96,6
            # print("xc_proj_weight shape", self.x_proj_weight.shape) #4,38,96
            # print("Dsc shape", self.Ds.shape) # 384
            # print("dtc_projs_bias shape", self.dt_projs_bias.shape) #4,96

            x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
            # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
            dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
            dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

            xs = xs.view(B, -1, L)  # (b, k * d, l)
            dts = dts.contiguous().view(B, -1, L)  # (b, k * d, l)
            As = -torch.exp(self.A_logs.float())  # (k * d, d_state)
            Ds = self.Ds  # (k * d)
            dt_projs_bias = self.dt_projs_bias.view(-1)  # (k * d)

            # assert len(xs.shape) == 3 and len(dts.shape) == 3 and len(Bs.shape) == 4 and len(Cs.shape) == 4
            # assert len(As.shape) == 2 and len(Ds.shape) == 1 and len(dt_projs_bias.shape) == 1
            # print(self.Ds.dtype, self.A_logs.dtype, self.dt_projs_bias.dtype, flush=True) # fp16, fp16, fp16

            # print("xs shape", xs.shape) #8,768,1024
            # print("dts shape", dts.shape) #8, 768, 1024
            # print("As shape", As.shape)#768, 16
            # print("Bs shape", Bs.shape)#8, 4, 16, 1024
            # print("Cs shape", Cs.shape)#8, 4, 16, 1024
            # print("Ds shape", Ds.shape)#768
            # print("dt_projs_bias shape", dt_projs_bias.shape) #768
            out_y = self.selective_scan(
                xs, dts,
                As, Bs, Cs, Ds,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
            ).view(B, 4, -1, L)
            # assert out_y.dtype == torch.float16

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0].float() + inv_y[:, 0].float() + wh_y.float() + invwh_y.float()

        y = y.view(B, C, H, W)
        y = self.out_norm(y).to(x.dtype)

        return y

    def cforward_corev1(self, xc: torch.Tensor):
        self.selective_scanC = selective_scan_fn_v1

        b, d, h, w = xc.shape

        # xc = self.pooling(xc).squeeze(-1).permute(0,2,1).contiguous() #b,1,d, >1!
        # print("xc shape", xc.shape) # 8,1,96
        xc = self.pooling(xc)  # b,d,1,1
        xc = xc.permute(0, 2, 1, 3).contiguous()  # b,1,d,1
        xc = self.conv_cin(xc)  # b,4,d,1
        xc = xc.squeeze(-1)  # b,4,d

        # xc = xc.permute(0,2,1).contiguous()
        # xc = self.conv_innerc(xc)
        # xc = xc.permute(0,2,1).contiguous()

        B, D, L = xc.shape  # b,1,c
        D, N = self.Ac_logs.shape  # 2,16
        K, D, R = self.dtc_projs_weight.shape  # 2,1,6

        # print("Ac_logs shape", self.Ac_logs.shape) #2,16
        # print("dtc_projs_weight shape", self.dtc_projs_weight.shape) #2,1,6
        # print("xc_proj_weight shape", self.xc_proj_weight.shape) #2,38,1
        # print("Dsc shape", self.Dsc.shape) # 2
        # print("dtc_projs_bias shape", self.dtc_projs_bias.shape) #2,1

        xsc = torch.stack([xc, torch.flip(xc, dims=[-1])], dim=1)  # input:b,d,l output:b,2,d,l
        # print("xsc shape", xsc.shape) # 8,2,1,96

        xc_dbl = torch.einsum("b k d l, k c d -> b k c l", xsc, self.xc_proj_weight)  # 8,2,1,96; 2,38,1 ->8,2,38,96

        dts, Bs, Cs = torch.split(xc_dbl, [self.dtc_rank, self.dc_state, self.dc_state], dim=2)  # 8,2,38,96-> 6,16,16
        # dts:8,2,6,96 bs,cs:8,2,16,96
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dtc_projs_weight).contiguous()

        xsc = xsc.view(B, -1, L)  # (b, k * d, l) 8,2,96
        dts = dts.contiguous().view(B, -1, L).contiguous()  # (b, k * d, l) 8,2,96
        As = -torch.exp(self.Ac_logs.float())  # (k * d, d_state) 2,16
        Ds = self.Dsc  # (k * d) 2
        dt_projs_bias = self.dtc_projs_bias.view(-1)  # (k * d)2

        # assert len(xs.shape) == 3 and len(dts.shape) == 3 and len(Bs.shape) == 4 and len(Cs.shape) == 4
        # assert len(As.shape) == 2 and len(Ds.shape) == 1 and len(dt_projs_bias.shape) == 1
        # print(self.Ds.dtype, self.A_logs.dtype, self.dt_projs_bias.dtype, flush=True) # fp16, fp16, fp16

        # print("channel xs shape", xsc.shape) #8, 2, 192
        # print("channel dts shape", dts.shape)#8, 2, 192
        # print("channel As shape", As.shape)#2, 16
        # print("channel Bs shape", Bs.shape)#8, 2, 16, 192
        # print("channel Cs shape", Cs.shape)#8, 2, 16, 192
        # print("channel Ds shape", Ds.shape)#2
        # print("channel dt_projs_bias shape", dt_projs_bias.shape) #2
        out_y = self.selective_scanC(
            xsc, dts,
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, 2, -1, L)

        y = out_y[:, 0].float() + torch.flip(out_y[:, 1], dims=[-1]).float()
        # y = xsc[:, 0].float() + torch.flip(xsc[:, 1], dims=[-1]).float()

        # y: b,4,d
        y = y.unsqueeze(-1)  # b,4,d,1
        y = self.conv_cout(y)  # b,1,d,1
        y = y.transpose(dim0=1, dim1=2).contiguous()  # b,d,1,1
        y = self.channel_norm(y)
        y = y.to(xc.dtype)

        # y = y.transpose(dim0=1, dim1=2).contiguous().unsqueeze(-1).contiguous()
        # y = self.channel_norm(y)
        # y = y.to(xc.dtype)

        # y = y.permute(0,2,1,3).contiguous()
        # y = self.conv_outc(y)
        # y = y.permute(0,2,1,3).contiguous()

        return y

    def forward(self, x: torch.Tensor, **kwargs):
        # input: b,d,h,w
        # output: b,d,h,w
        xz = self.in_conv(x)
        x, z = xz.chunk(2, dim=1)  # (b, d, h, w)
        if not self.softmax_version:
            z = self.act(z)
        x = self.act(self.conv2d(x))  # (b, d, h, w)
        y1 = self.forward_core(x)
        y2 = y1 * z
        c = self.cforward_core(y2)  # x:b,d,h,w; output:b,d,1,1
        y2 = y2 + c
        out = self.out_conv(y2)
        return out

class LMamberBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(LMamberBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = LSS2D_1(d_model=dim, ssm_ratio=1)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = LFeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x
class Mamba_generator(nn.Module):
    def __init__(self,
        heads =1,
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',
                 in_nc=1, nf=8, num_blocks=5
                 ## Other option 'BiasFree'
      ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
    ):

        super(Mamba_generator, self).__init__()
        kernel_size = 3
        padding = 1

        layers = []
        layers.append(DConv2d(in_channels=in_nc, out_channels=nf, kernel_size=kernel_size, padding=padding,
                                bias=False))
        layers.append(nn.ReLU(inplace=True))
        self.relu1 = nn.Sigmoid()

        for _ in range(4):
            layers.append(
                LMamberBlock(dim=nf, num_heads=heads, ffn_expansion_factor=ffn_expansion_factor,
                            bias=bias, LayerNorm_type=LayerNorm_type))
            # layers.append(nn.BatchNorm2d(nf))
            # layers.append(nn.ReLU(inplace=True))

        layers.append(
            DConv2d(in_channels=nf, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False))
        self.dnm= nn.Sequential(*layers)


    def forward(self, x0):
        out = self.dnm(x0)
        return out



class Generator(nn.Module):
    def __init__(self, channels = 1, num_of_layers=10):
        super(Generator, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.LeakyReLU(inplace=True))
        self.relu1 = nn.Sigmoid()

        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.LeakyReLU(inplace=True))

        layers.append(nn.Conv2d(in_channels=features, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x0):
        out = self.dncnn(x0)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class KGenerator(nn.Module):
    def __init__(self, channels = 1, num_of_layers=10):
        super(KGenerator, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.LeakyReLU(inplace=True))
        self.relu1 = nn.Sigmoid()
        dpr = [x.item() for x in torch.linspace(0, 0, 3)]
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.LeakyReLU(inplace=True))

        layers.append(nn.Conv2d(in_channels=features, out_channels=8, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)

        self.patch_embed = PatchEmbed(img_size=512// 4, patch_size=3, stride=1, in_chans=8,
                                       embed_dim=8)
        self.kblock= KANBlock(
            dim=8,
            drop=0, drop_path=dpr[1], norm_layer=nn.LayerNorm
        )
        self._initialize_weights()

        self.norm= nn.LayerNorm(8)
        self.last=nn.Conv2d(in_channels=8, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False)
    def forward(self, x0):
        B = x0.shape[0]
        out = self.dncnn(x0)
        out, H, W = self.patch_embed(out)
        # print(out.shape)
        out = self.kblock(out, H, W)
        # print(out.shape)
        out = self.norm(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out=self.last(out)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()









class LDConv(nn.Module):
    def __init__(self, inc, outc, num_param, stride=1, bias=None):
        super(LDConv, self).__init__()
        self.num_param = num_param
        self.stride = stride
        self.conv = nn.Sequential(nn.Conv2d(inc, outc, kernel_size=(num_param, 1), stride=(num_param, 1), bias=bias),
                                  nn.BatchNorm2d(outc),
                                  nn.SiLU())  # the conv adds the BN and SiLU to compare original Conv in YOLOv5.
        self.p_conv = nn.Conv2d(inc, 2 * num_param, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_full_backward_hook(self._set_lr)
        self.register_buffer("p_n", self._get_p_n(N=self.num_param))

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        # N is num_param.
        offset = self.p_conv(x)
        dtype = offset.data.type()
        N = offset.size(1) // 2
        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # resampling the features based on the modified coordinates.
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # bilinear
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        x_offset = self._reshape_x_offset(x_offset, self.num_param)
        out = self.conv(x_offset)

        return out

    # generating the inital sampled shapes for the LDConv with different sizes.
    def _get_p_n(self, N):
        base_int = round(math.sqrt(self.num_param))
        row_number = self.num_param // base_int
        mod_number = self.num_param % base_int
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(0, row_number),
            torch.arange(0, base_int))
        p_n_x = torch.flatten(p_n_x)
        p_n_y = torch.flatten(p_n_y)
        if mod_number > 0:
            mod_p_n_x, mod_p_n_y = torch.meshgrid(
                torch.arange(row_number, row_number + 1),
                torch.arange(0, mod_number))

            mod_p_n_x = torch.flatten(mod_p_n_x)
            mod_p_n_y = torch.flatten(mod_p_n_y)
            p_n_x, p_n_y = torch.cat((p_n_x, mod_p_n_x)), torch.cat((p_n_y, mod_p_n_y))
        p_n = torch.cat([p_n_x, p_n_y], 0)
        p_n = p_n.view(1, 2 * N, 1, 1)
        return p_n

    # no zero-padding
    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(0, h * self.stride, self.stride),
            torch.arange(0, w * self.stride, self.stride))

        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        # p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + self.p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    #  Stacking resampled features in the row direction.
    @staticmethod
    def _reshape_x_offset(x_offset, num_param):
        b, c, h, w, n = x_offset.size()
        # using Conv3d
        # x_offset = x_offset.permute(0,1,4,2,3), then Conv3d(c,c_out, kernel_size =(num_param,1,1),stride=(num_param,1,1),bias= False)
        # using 1 × 1 Conv
        # x_offset = x_offset.permute(0,1,4,2,3), then, x_offset.view(b,c×num_param,h,w)  finally, Conv2d(c×num_param,c_out, kernel_size =1,stride=1,bias= False)
        # using the column conv as follow， then, Conv2d(inc, outc, kernel_size=(num_param, 1), stride=(num_param, 1), bias=bias)

        x_offset = rearrange(x_offset, 'b c h w n -> b c (h n) w')
        return x_offset


import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Convx(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.LeakyReLU(inplace=True)  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class PConv(nn.Module):
    ''' Pinwheel-shaped Convolution using the Asymmetric Padding method. '''

    def __init__(self, c1, c2, k, s):
        super().__init__()

        # self.k = k
        p = [(k, 0, 1, 0), (0, k, 0, 1), (0, 1, k, 0), (1, 0, 0, k)]
        self.pad = [nn.ZeroPad2d(padding=(p[g])) for g in range(4)]
        self.cw = Convx(c1, c2 // 4, (1, k), s=s, p=0)
        self.ch = Convx(c1, c2 // 4, (k, 1), s=s, p=0)
        self.cat = Convx(c2, c2, 2, s=1, p=0)

    def forward(self, x):
        yw0 = self.cw(self.pad[0](x))
        yw1 = self.cw(self.pad[1](x))
        yh0 = self.ch(self.pad[2](x))
        yh1 = self.ch(self.pad[3](x))
        return self.cat(torch.cat([yw0, yw1, yh0, yh1], dim=1))
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



class PGenerator(nn.Module):
    def __init__(self, channels = 1, num_of_layers=10):
        super(PGenerator, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.LeakyReLU(inplace=True))
        self.relu1 = nn.Sigmoid()

        for _ in range(num_of_layers-6):
            layers.append(PConv(features, features, kernel_size, padding))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.LeakyReLU(inplace=True))

        layers.append(nn.Conv2d(in_channels=features, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)




    def forward(self, x0):
        out = self.dncnn(x0)

        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

if __name__ == "__main__":
    # model=Kan_Generator().cuda()
    # x=torch.randn([1,1,640,640]).cuda()
    #
    # y=model(x)
    # print(y.shape)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(1, 1, 640, 640).to(device)
    pconv = PConv(c1=1, c2=64, k=3, s=1)
    print(pconv)
    pconv = pconv.to(device)
    output = pconv(x)

    print("输入张量形状:", x.shape)

    print("输出张量形状:", output.shape)
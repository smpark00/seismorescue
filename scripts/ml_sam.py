"""SAM 2.1 (Meta, open weights, Apache-2.0) fine-tuned as a promptable helicorder trace segmenter (docs/RESULTS.md).

usage: python -I scripts/ml_sam.py <command> [options]
  selftest                      load the checkpoint into the re-implemented image path, zero-shot sanity masks, timing
  cache-b <split>               proposals A/B (arbiter.proposals) for val or holdout3 images -> out/ml/sam/cache_b/
  train [--minutes M] [--windows N --lora stages23|all --lora-r R --amp]
                                fine-tune on the training groups of out/film_train -> out/ml/sam/ckpt_<tag>_{best,last}.pt
  eval <split> [--ckpt PATH --lam L]   split = val | holdout3 | zeroshot-val: per-line fail % vs A/B -> eval_<split>.json
  tune-lam <probs tag> [--lams ...]    final-Viterbi smoothness on the cached validation probabilities only
  analyze <paths tag>           quake-line failures by GT slope, crossing zones, off-image columns, long swaps
  real <KIP|MAT> [--ckpt PATH --lam L] [--check-only]
                                real paired scan with real_paired_eval's stored layout, alignment and metrics
Frozen method (out/ml/sam/frozen_method.json): ckpt_tiny_best.pt, --lam 1e-5 (chosen on validation groups only).

Model: only the single-image path of SAM 2.1 is needed (Hiera trunk + FPN neck + prompt encoder + two-way mask decoder,
no memory modules). It is re-implemented here in plain PyTorch with the official parameter names, so the official
checkpoint (dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_<size>.pt) loads with strict key/shape checks
and no extra packages (the brief allows network only for weights). `selftest` verifies the port zero-shot.

Task: a window of the film-geometry helicorder (all rows x WIN_W columns, resized to the native 1024 x 1024) and point
prompts on one target line (its position at the window's left edge, plus optional points along the first OVERLAP
columns) -> mask of that line's pen trace in the window. Target = the GT pen polyline drawn ~5 px wide (inst-like,
covers the near-vertical strokes). Windows are chained along x: the first window is prompted with the layout start
(GT at x0, the same start information the trackers get); every later window with points on the path decoded so far.
The stitched mask probabilities are decoded into one y per column by a Viterbi path with a quadratic continuity prior.
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
OUT = ROOT / "out" / "ml" / "sam"
WEIGHTS = OUT / "weights"

VARIANTS = {   # official SAM 2.1 configs (sam2.1_hiera_{t,s,b+,l}.yaml)
    "tiny": dict(trunk=dict(embed_dim=96, num_heads=1, stages=(1, 2, 7, 2), global_att_blocks=(5, 7, 9),
                            window_pos_embed_bkg_spatial_size=(7, 7), window_spec=(8, 4, 14, 7)),
                 channels=(768, 384, 192, 96), file="sam2.1_hiera_tiny.pt"),
    "small": dict(trunk=dict(embed_dim=96, num_heads=1, stages=(1, 2, 11, 2), global_att_blocks=(7, 10, 13),
                             window_pos_embed_bkg_spatial_size=(7, 7), window_spec=(8, 4, 14, 7)),
                  channels=(768, 384, 192, 96), file="sam2.1_hiera_small.pt"),
    "base_plus": dict(trunk=dict(embed_dim=112, num_heads=2, stages=(2, 3, 16, 3), global_att_blocks=(12, 16, 20),
                                 window_pos_embed_bkg_spatial_size=(14, 14), window_spec=(8, 4, 14, 7)),
                      channels=(896, 448, 224, 112), file="sam2.1_hiera_base_plus.pt"),
    "large": dict(trunk=dict(embed_dim=144, num_heads=2, stages=(2, 6, 36, 4), global_att_blocks=(23, 33, 43),
                             window_pos_embed_bkg_spatial_size=(7, 7), window_spec=(8, 4, 16, 8)),
                  channels=(1152, 576, 288, 144), file="sam2.1_hiera_large.pt"),
}
WEIGHT_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/{file}"
IMG = 1024                       # SAM native input size
LOWRES = 256                     # mask decoder output size
PIX_MEAN = (0.485, 0.456, 0.406)
PIX_STD = (0.229, 0.224, 0.225)


# =============================================================================================== SAM 2.1 image path
class LayerNorm2d(nn.Module):
    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.weight, self.bias, self.eps = nn.Parameter(torch.ones(c)), nn.Parameter(torch.zeros(c)), eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        return self.weight[:, None, None] * ((x - u) / torch.sqrt(s + self.eps)) + self.bias[:, None, None]


class MLP(nn.Module):
    def __init__(self, din, dh, dout, n, act=nn.ReLU, sigmoid_output=False):
        super().__init__()
        h = [dh] * (n - 1)
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip([din] + h, h + [dout]))
        self.n, self.act, self.sigmoid_output = n, act(), sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.n - 1 else layer(x)
        return torch.sigmoid(x) if self.sigmoid_output else x


class LoRALinear(nn.Module):
    """y = W x + b + (alpha/r) B A x; W, b frozen (wrapped after the checkpoint is loaded)"""
    def __init__(self, base, r=8, alpha=16):
        super().__init__()
        self.base, self.scale = base, alpha / r
        self.lora_a = nn.Parameter(torch.empty(r, base.in_features, device=base.weight.device))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + (x @ self.lora_a.t() @ self.lora_b.t()) * self.scale


def window_partition(x, ws):
    B, H, W, C = x.shape
    ph, pw = (ws - H % ws) % ws, (ws - W % ws) % ws
    if ph or pw:
        x = F.pad(x, (0, 0, 0, pw, 0, ph))
    Hp, Wp = H + ph, W + pw
    x = x.view(B, Hp // ws, ws, Wp // ws, ws, C).permute(0, 1, 3, 2, 4, 5).reshape(-1, ws, ws, C)
    return x, (Hp, Wp)


def window_unpartition(win, ws, pad_hw, hw):
    Hp, Wp = pad_hw
    H, W = hw
    B = win.shape[0] // (Hp * Wp // ws // ws)
    x = win.reshape(B, Hp // ws, Wp // ws, ws, ws, -1).permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, -1)
    return x[:, :H, :W, :] if (Hp > H or Wp > W) else x


def do_pool(x, pool):
    return x if pool is None else pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)


class MultiScaleAttention(nn.Module):
    def __init__(self, dim, dim_out, num_heads, q_pool=None):
        super().__init__()
        self.num_heads, self.q_pool = num_heads, q_pool
        self.qkv = nn.Linear(dim, dim_out * 3)
        self.proj = nn.Linear(dim_out, dim_out)

    def forward(self, x):
        B, H, W, _ = x.shape
        q, k, v = torch.unbind(self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1), 2)
        if self.q_pool is not None:
            q = do_pool(q.reshape(B, H, W, -1), self.q_pool)
            H, W = q.shape[1:3]
            q = q.reshape(B, H * W, self.num_heads, -1)
        x = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        return self.proj(x.transpose(1, 2).reshape(B, H, W, -1))


class MultiScaleBlock(nn.Module):
    def __init__(self, dim, dim_out, num_heads, q_stride=None, window_size=0, mlp_ratio=4.0):
        super().__init__()
        self.dim, self.dim_out, self.window_size, self.q_stride = dim, dim_out, window_size, q_stride
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.pool = nn.MaxPool2d(kernel_size=q_stride, stride=q_stride, ceil_mode=False) if q_stride else None
        self.attn = MultiScaleAttention(dim, dim_out, num_heads, q_pool=self.pool)
        self.norm2 = nn.LayerNorm(dim_out, eps=1e-6)
        self.mlp = MLP(dim_out, int(dim_out * mlp_ratio), dim_out, 2, act=nn.GELU)
        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        if self.dim != self.dim_out:
            shortcut = do_pool(self.proj(x), self.pool)
        ws = self.window_size
        H, W = x.shape[1], x.shape[2]
        if ws > 0:
            x, pad_hw = window_partition(x, ws)
        x = self.attn(x)
        if self.q_stride:
            ws = self.window_size // self.q_stride[0]
            H, W = shortcut.shape[1:3]
            pad_hw = (H + (ws - H % ws) % ws, W + (ws - W % ws) % ws)
        if self.window_size > 0:
            x = window_unpartition(x, ws, pad_hw, (H, W))
        x = shortcut + x
        return x + self.mlp(self.norm2(x))


class PatchEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, kernel_size=7, stride=4, padding=3)

    def forward(self, x):
        return self.proj(x).permute(0, 2, 3, 1)


class Hiera(nn.Module):
    def __init__(self, embed_dim, num_heads, stages, global_att_blocks, window_pos_embed_bkg_spatial_size, window_spec,
                 q_pool=3, q_stride=(2, 2), dim_mul=2.0, head_mul=2.0):
        super().__init__()
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]
        q_pool_blocks = [x + 1 for x in self.stage_ends[:-1]][:q_pool]
        self.patch_embed = PatchEmbed(embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, *window_pos_embed_bkg_spatial_size))
        self.pos_embed_window = nn.Parameter(torch.zeros(1, embed_dim, window_spec[0], window_spec[0]))
        self.blocks = nn.ModuleList()
        cur_stage = 1
        for i in range(sum(stages)):
            dim_out = embed_dim
            window_size = 0 if i in global_att_blocks else window_spec[cur_stage - 1]   # lags by one block
            if i - 1 in self.stage_ends:
                dim_out, num_heads, cur_stage = int(embed_dim * dim_mul), int(num_heads * head_mul), cur_stage + 1
            self.blocks.append(MultiScaleBlock(embed_dim, dim_out, num_heads,
                                               q_stride=q_stride if i in q_pool_blocks else None, window_size=window_size))
            embed_dim = dim_out
        self._pe_cache = None

    def pos(self, hw):
        if self._pe_cache is None or self._pe_cache.shape[1:3] != hw or self._pe_cache.device != self.pos_embed.device:
            with torch.no_grad():                      # pos embeds stay frozen; bicubic resize done once on CPU
                pe = F.interpolate(self.pos_embed.detach().float().cpu(), size=hw, mode="bicubic")
                w = self.pos_embed_window.detach().float().cpu()
                pe = pe + w.tile([a // b for a, b in zip(pe.shape, w.shape)])
            self._pe_cache = pe.permute(0, 2, 3, 1).to(self.pos_embed.device)
        return self._pe_cache

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos(tuple(x.shape[1:3])).to(x.dtype)
        outs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.stage_ends:
                outs.append(x.permute(0, 3, 1, 2))
        return outs


class FpnNeck(nn.Module):
    def __init__(self, channels, d_model=256, top_down_levels=(2, 3)):
        super().__init__()
        self.convs = nn.ModuleList()
        for c in channels:
            m = nn.Sequential()
            m.add_module("conv", nn.Conv2d(c, d_model, 1))
            self.convs.append(m)
        self.top_down = list(top_down_levels)

    def forward(self, xs):
        n = len(self.convs) - 1
        out, prev = [None] * len(self.convs), None
        for i in range(n, -1, -1):
            lat = self.convs[n - i](xs[i])
            if i in self.top_down and prev is not None:
                prev = lat + F.interpolate(prev, scale_factor=2.0, mode="nearest")
            else:
                prev = lat
            out[i] = prev
        return out


class ImageEncoder(nn.Module):
    def __init__(self, trunk, neck):
        super().__init__()
        self.trunk, self.neck = trunk, neck

    def forward(self, x):
        return self.neck(self.trunk(x))[:-1]         # scalp = 1: drop the stride-32 level


class PositionEmbeddingRandom(nn.Module):
    def __init__(self, n=128):
        super().__init__()
        self.register_buffer("positional_encoding_gaussian_matrix", torch.randn((2, n)))

    def _pe(self, c):
        c = (2 * c - 1) @ self.positional_encoding_gaussian_matrix
        c = 2 * math.pi * c
        return torch.cat([torch.sin(c), torch.cos(c)], dim=-1)

    def dense(self, h, w):
        g = torch.ones((h, w), device=self.positional_encoding_gaussian_matrix.device)
        y, x = (g.cumsum(0) - 0.5) / h, (g.cumsum(1) - 0.5) / w
        return self._pe(torch.stack([x, y], dim=-1)).permute(2, 0, 1)

    def coords(self, pts, hw):
        c = pts.clone()
        c[..., 0] = c[..., 0] / hw[1]
        c[..., 1] = c[..., 1] / hw[0]
        return self._pe(c.float())


class PromptEncoder(nn.Module):
    def __init__(self, d=256, emb_hw=(64, 64), img_hw=(IMG, IMG), mask_in_chans=16):
        super().__init__()
        self.emb_hw, self.img_hw = emb_hw, img_hw
        self.pe_layer = PositionEmbeddingRandom(d // 2)
        self.point_embeddings = nn.ModuleList(nn.Embedding(1, d) for _ in range(4))
        self.not_a_point_embed = nn.Embedding(1, d)
        self.mask_downscaling = nn.Sequential(                       # unused here, kept for the checkpoint
            nn.Conv2d(1, mask_in_chans // 4, 2, 2), LayerNorm2d(mask_in_chans // 4), nn.GELU(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, 2, 2), LayerNorm2d(mask_in_chans), nn.GELU(),
            nn.Conv2d(mask_in_chans, d, 1))
        self.no_mask_embed = nn.Embedding(1, d)

    def dense_pe(self):
        return self.pe_layer.dense(*self.emb_hw).unsqueeze(0)

    def forward(self, pts, labels):
        """pts (N, P, 2) in 1024-input pixels (x, y); labels (N, P): 1 pos, 0 neg, -1 padding"""
        N = pts.shape[0]
        pts = torch.cat([pts + 0.5, torch.zeros((N, 1, 2), device=pts.device)], 1)
        labels = torch.cat([labels, -torch.ones((N, 1), device=labels.device, dtype=labels.dtype)], 1)
        pe = self.pe_layer.coords(pts, self.img_hw)
        lab = labels.unsqueeze(-1)
        pe = torch.where(lab == -1, torch.zeros_like(pe) + self.not_a_point_embed.weight, pe)
        pe = torch.where(lab == 0, pe + self.point_embeddings[0].weight, pe)
        pe = torch.where(lab == 1, pe + self.point_embeddings[1].weight, pe)
        dense = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(N, -1, *self.emb_hw)
        return pe, dense


class Attention(nn.Module):
    def __init__(self, d, heads, downsample_rate=1):
        super().__init__()
        di = d // downsample_rate
        self.heads = heads
        self.q_proj, self.k_proj, self.v_proj, self.out_proj = nn.Linear(d, di), nn.Linear(d, di), nn.Linear(d, di), nn.Linear(di, d)

    def forward(self, q, k, v):
        def sep(x):
            b, n, c = x.shape
            return x.reshape(b, n, self.heads, c // self.heads).transpose(1, 2)
        o = F.scaled_dot_product_attention(sep(self.q_proj(q)), sep(self.k_proj(k)), sep(self.v_proj(v)))
        b, h, n, c = o.shape
        return self.out_proj(o.transpose(1, 2).reshape(b, n, h * c))


class TwoWayBlock(nn.Module):
    def __init__(self, d, heads, mlp_dim, skip_first_layer_pe):
        super().__init__()
        self.self_attn = Attention(d, heads)
        self.norm1 = nn.LayerNorm(d)
        self.cross_attn_token_to_image = Attention(d, heads, 2)
        self.norm2 = nn.LayerNorm(d)
        self.mlp = MLP(d, mlp_dim, d, 2, act=nn.ReLU)
        self.norm3 = nn.LayerNorm(d)
        self.norm4 = nn.LayerNorm(d)
        self.cross_attn_image_to_token = Attention(d, heads, 2)
        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(self, queries, keys, qpe, kpe):
        if self.skip_first_layer_pe:
            queries = self.self_attn(queries, queries, queries)
        else:
            q = queries + qpe
            queries = queries + self.self_attn(q, q, queries)
        queries = self.norm1(queries)
        queries = self.norm2(queries + self.cross_attn_token_to_image(queries + qpe, keys + kpe, keys))
        queries = self.norm3(queries + self.mlp(queries))
        keys = self.norm4(keys + self.cross_attn_image_to_token(keys + kpe, queries + qpe, queries))
        return queries, keys


class TwoWayTransformer(nn.Module):
    def __init__(self, depth=2, d=256, heads=8, mlp_dim=2048):
        super().__init__()
        self.layers = nn.ModuleList(TwoWayBlock(d, heads, mlp_dim, i == 0) for i in range(depth))
        self.final_attn_token_to_image = Attention(d, heads, 2)
        self.norm_final_attn = nn.LayerNorm(d)

    def forward(self, img, img_pe, pts):
        keys = img.flatten(2).permute(0, 2, 1)
        kpe = img_pe.flatten(2).permute(0, 2, 1)
        queries = pts
        for layer in self.layers:
            queries, keys = layer(queries, keys, pts, kpe)
        queries = self.norm_final_attn(queries + self.final_attn_token_to_image(queries + pts, keys + kpe, keys))
        return queries, keys


class MaskDecoder(nn.Module):
    def __init__(self, d=256, n_multi=3):
        super().__init__()
        self.transformer = TwoWayTransformer(2, d, 8, 2048)
        self.iou_token = nn.Embedding(1, d)
        self.num_mask_tokens = n_multi + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, d)
        self.obj_score_token = nn.Embedding(1, d)
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(d, d // 4, 2, 2), LayerNorm2d(d // 4), nn.GELU(), nn.ConvTranspose2d(d // 4, d // 8, 2, 2), nn.GELU())
        self.conv_s0 = nn.Conv2d(d, d // 8, 1)
        self.conv_s1 = nn.Conv2d(d, d // 4, 1)
        self.output_hypernetworks_mlps = nn.ModuleList(MLP(d, d, d // 8, 3) for _ in range(self.num_mask_tokens))
        self.iou_prediction_head = MLP(d, 256, self.num_mask_tokens, 3, sigmoid_output=True)
        self.pred_obj_score_head = MLP(d, d, 1, 3)

    def forward(self, emb, img_pe, sparse, dense, hr):
        """emb (N, 256, 64, 64) per prompt; returns all 4 mask logits (N, 4, 256, 256) and iou (N, 4)"""
        tok = torch.cat([self.obj_score_token.weight, self.iou_token.weight, self.mask_tokens.weight], 0)
        tokens = torch.cat([tok.unsqueeze(0).expand(sparse.shape[0], -1, -1), sparse], 1)
        src = emb + dense
        b, c, h, w = src.shape
        hs, src = self.transformer(src, img_pe.expand(b, -1, -1, -1), tokens)
        iou_tok, mask_tok = hs[:, 1], hs[:, 2:2 + self.num_mask_tokens]
        src = src.transpose(1, 2).reshape(b, c, h, w)
        dc1, ln1, act1, dc2, act2 = self.output_upscaling
        f0, f1 = hr
        up = act1(ln1(dc1(src) + f1))
        up = act2(dc2(up) + f0)
        hyper = torch.stack([m(mask_tok[:, i]) for i, m in enumerate(self.output_hypernetworks_mlps)], 1)
        b, c, h, w = up.shape
        masks = (hyper @ up.view(b, c, h * w)).view(b, -1, h, w)
        return masks, self.iou_prediction_head(iou_tok)


class SAM21Image(nn.Module):
    def __init__(self, variant="tiny"):
        super().__init__()
        cfg = VARIANTS[variant]
        self.variant = variant
        self.image_encoder = ImageEncoder(Hiera(**cfg["trunk"]), FpnNeck(cfg["channels"]))
        self.sam_prompt_encoder = PromptEncoder()
        self.sam_mask_decoder = MaskDecoder()
        self.no_mem_embed = nn.Parameter(torch.zeros(1, 1, 256))

    def encode(self, x):
        """x (B, 3, 1024, 1024) normalised -> image embedding (B, 256, 64, 64) and high-res feats (s4, s8)"""
        f = self.image_encoder(x)
        emb = f[2] + self.no_mem_embed.view(1, -1, 1, 1)          # single-image mode: directly add no_mem_embed
        return emb, (self.sam_mask_decoder.conv_s0(f[0]), self.sam_mask_decoder.conv_s1(f[1]))

    def decode(self, emb, hr, idx, pts, labels):
        """idx (N,): image of each prompt; pts (N, P, 2) 1024-input px; -> logits (N, 4, 256, 256), iou (N, 4)"""
        sparse, dense = self.sam_prompt_encoder(pts, labels)
        return self.sam_mask_decoder(emb[idx], self.sam_prompt_encoder.dense_pe(), sparse, dense, (hr[0][idx], hr[1][idx]))


def load_sam21(variant="tiny", device="cpu"):
    """build + load the official checkpoint (strict on every image-path key)"""
    path = WEIGHTS / f"sam2.1_hiera_{variant}" / VARIANTS[variant]["file"]
    sd = torch.load(path, map_location="cpu", weights_only=True)["model"]
    model = SAM21Image(variant)
    own = model.state_dict()
    keep = {k: v for k, v in sd.items() if k in own}
    missing = [k for k in own if k not in keep]
    bad = [k for k in keep if keep[k].shape != own[k].shape]
    assert not missing and not bad, f"checkpoint mismatch: missing {missing[:5]} shape {bad[:5]}"
    model.load_state_dict(keep, strict=True)
    skipped = sorted({k.split(".")[0] for k in sd if k not in own})        # video-memory modules
    return model.to(device), {"checkpoint": str(path.relative_to(ROOT)), "loaded_tensors": len(keep),
                              "skipped_prefixes": skipped}


def add_lora(model, r=8, alpha=16, blocks="all"):
    """LoRA on the trunk's attention qkv/proj; base weights frozen"""
    n = 0
    for i, blk in enumerate(model.image_encoder.trunk.blocks):
        if blocks != "all" and i not in blocks:
            continue
        blk.attn.qkv = LoRALinear(blk.attn.qkv, r, alpha)
        blk.attn.proj = LoRALinear(blk.attn.proj, r, alpha)
        n += 2
    return n


def to_input(win_u8, device):
    """grayscale window (h, w) uint8 -> normalised (1, 3, 1024, 1024)"""
    x = cv2.resize(win_u8, (IMG, IMG), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    t = torch.from_numpy(x)[None, None].expand(1, 3, IMG, IMG)
    m = torch.tensor(PIX_MEAN).view(1, 3, 1, 1)
    s = torch.tensor(PIX_STD).view(1, 3, 1, 1)
    return ((t - m) / s).to(device)


def get_device():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# =============================================================================================== selftest
def selftest(args):
    dev = get_device()
    model, info = load_sam21(args.variant, dev)
    model.eval()
    print("loaded", info, flush=True)
    # zero-shot sanity: shapes on paper-like background, one positive point each -> best multimask IoU vs truth
    H = W = 600
    img = np.full((H, W), 235, np.uint8)
    cv2.circle(img, (170, 200), 90, 40, -1)
    cv2.rectangle(img, (330, 330), (520, 470), 90, -1)
    cv2.ellipse(img, (420, 150), (120, 40), 20, 0, 360, 140, -1)
    truth = {}
    for name, draw in (("disk", lambda m: cv2.circle(m, (170, 200), 90, 1, -1)),
                       ("rect", lambda m: cv2.rectangle(m, (330, 330), (520, 470), 1, -1)),
                       ("ellipse", lambda m: cv2.ellipse(m, (420, 150), (120, 40), 20, 0, 360, 1, -1))):
        m = np.zeros((H, W), np.uint8)
        draw(m)
        truth[name] = m > 0
    pts = {"disk": (170, 200), "rect": (425, 400), "ellipse": (420, 150)}
    x = to_input(img, dev)
    res = {}
    with torch.no_grad():
        t = time.time()
        emb, hr = model.encode(x)
        torch.mps.synchronize() if dev.type == "mps" else None
        names = list(pts)
        p = torch.tensor([[[pts[n][0] * IMG / W, pts[n][1] * IMG / H]] for n in names], dtype=torch.float32, device=dev)
        lab = torch.ones((len(names), 1), device=dev)
        logits, iou = model.decode(emb, hr, torch.zeros(len(names), dtype=torch.long, device=dev), p, lab)
        for i, n in enumerate(names):
            ious = []
            for j in range(4):
                m = F.interpolate(logits[i:i + 1, j:j + 1], (H, W), mode="bilinear", align_corners=False)[0, 0].cpu().numpy() > 0
                ious.append(float((m & truth[n]).sum() / max(1, (m | truth[n]).sum())))
            jb = 1 + int(torch.argmax(iou[i, 1:]))
            res[n] = {"iou_per_token": [round(v, 3) for v in ious], "picked_multimask": jb, "iou_picked": round(ious[jb], 3),
                      "pred_iou": [round(float(v), 3) for v in iou[i]]}
        print("zero-shot sanity:", json.dumps(res), f"({time.time() - t:.2f}s)", flush=True)
    # timing: encoder forward (no grad), and forward+backward with LoRA on the trunk
    for bs in (1, 2):
        xb = x.expand(bs, -1, -1, -1).contiguous()
        with torch.no_grad():
            for _ in range(2):
                model.encode(xb)
            torch.mps.synchronize() if dev.type == "mps" else None
            t = time.time()
            for _ in range(3):
                model.encode(xb)
            torch.mps.synchronize() if dev.type == "mps" else None
        print(f"encoder fwd no-grad bs{bs}: {(time.time() - t) / 3:.3f}s", flush=True)
    add_lora(model, 8, 16)
    for p_ in model.parameters():
        p_.requires_grad_(False)
    for n_, p_ in model.named_parameters():
        if "lora_" in n_ or n_.startswith(("sam_mask_decoder", "image_encoder.neck")):
            p_.requires_grad_(True)
    model.train()
    for bs in (1, 2):
        xb = x.expand(bs, -1, -1, -1).contiguous()
        times = []
        for it in range(4):
            t = time.time()
            emb, hr = model.encode(xb)
            idx = torch.arange(bs, device=dev).repeat_interleave(8)
            p = torch.rand((bs * 8, 3, 2), device=dev) * IMG
            logits, iou = model.decode(emb, hr, idx, p, torch.ones((bs * 8, 3), device=dev))
            loss = logits[:, 0].float().pow(2).mean()
            loss.backward()
            torch.mps.synchronize() if dev.type == "mps" else None
            times.append(time.time() - t)
        print(f"train step (LoRA trunk + decoder, {bs} windows x 8 prompts): {np.median(times[1:]):.3f}s; "
              f"mps mem {torch.mps.current_allocated_memory() / 1e9 if dev.type == 'mps' else 0:.2f} GB", flush=True)


# =============================================================================================== data
FILM_TRAIN = ROOT / "out" / "film_train"
FILM_HOLD = ROOT / "out" / "film_holdout3"
WIN_H, WIN_W, STEP = 406, 256, 192          # window = 406 rows (a whole film render) x 256 cols -> 1024 x 1024
PROMPT_COLS = (0, 16, 32, 48)                # chained windows: points on the decoded path inside the overlap
N_VAL_GROUPS = 5                             # of 22 event groups (~20 %)
LORA_BLOCKS = tuple(range(3, 12))            # stride-16/32 stages of hiera-tiny (stages 0-1 stay frozen)


def lora_blocks_for(variant, scope="stages23"):
    """blocks that get LoRA: the stride-16/32 stages (default) or every trunk block ("all")"""
    st = VARIANTS[variant]["trunk"]["stages"]
    first = 0 if scope == "all" else st[0] + st[1]
    return tuple(range(first, sum(st)))


def records(root):
    return sorted(p for p in root.iterdir() if (p / "film" / "meta.json").exists())


def split_records(seed=0):
    recs = records(FILM_TRAIN)
    groups = sorted({r.name[:8] for r in recs})
    g = groups[:]
    random.Random(seed).shuffle(g)
    val = sorted(g[:N_VAL_GROUPS])
    return [r for r in recs if r.name[:8] not in val], [r for r in recs if r.name[:8] in val], val


def load_item(rec, g):
    d = rec / "film"
    meta = json.loads((d / "meta.json").read_text())
    img = cv2.imread(str(d / f"heli_g{g:g}.png"), 0)
    assert img.shape[0] == WIN_H, img.shape
    pen = np.load(d / f"gt_g{g:g}.npz")["pen"].astype(np.float32)
    return {"record": rec.name, "group": rec.name[:8], "gain": float(g), "img": img, "pen": pen, "S": meta["spacing"],
            "x0": meta["margin_x"], "x1": img.shape[1] - meta["margin_x"], "active": meta["active_line"]}


def items_of(recs):
    return [(r, g) for r in recs for g in json.loads((r / "film" / "meta.json").read_text())["gains"]]


def crop(img, top, c, w, fill):
    """rows [top, top + WIN_H) x cols [c, c + w), paper fill outside the image"""
    H, W = img.shape
    out = np.full((WIN_H, w), fill, np.uint8)
    r0, r1, c0, c1 = max(0, top), min(H, top + WIN_H), max(0, c), min(W, c + w)
    if r1 > r0 and c1 > c0:
        out[r0 - top:r1 - top, c0 - c:c1 - c] = img[r0:r1, c0:c1]
    return out


def target_mask(pen_k, top, c, w, thick=3):
    """GT pen polyline in the 256 x 256 decoder frame, ~3 cells (~2.4 px either side) wide, incl. steep strokes"""
    xs = np.arange(max(0, c - 2), min(len(pen_k), c + w + 2))
    ok = np.isfinite(pen_k[xs])
    m = np.zeros((LOWRES, LOWRES), np.uint8)
    if ok.sum() < 2:
        return m
    u = (xs[ok] - c + 0.5) * LOWRES / w - 0.5
    v = np.clip((pen_k[xs[ok]] - top + 0.5) * LOWRES / WIN_H - 0.5, -60, LOWRES + 60)
    pts = np.round(np.stack([u, v], 1) * 16).astype(np.int32)
    cv2.polylines(m, [pts], False, 1, thickness=thick, lineType=cv2.LINE_8, shift=4)
    return m


def to_tensor_batch(wins, device):
    x = np.stack([cv2.resize(w, (IMG, IMG), interpolation=cv2.INTER_LINEAR) for w in wins]).astype(np.float32) / 255.0
    t = torch.from_numpy(x)[:, None].expand(-1, 3, -1, -1)
    m = torch.tensor(PIX_MEAN).view(1, 3, 1, 1)
    s = torch.tensor(PIX_STD).view(1, 3, 1, 1)
    return ((t - m) / s).to(device)


def augment(win, rng):
    """photometric jitter (contrast of the ink, blur, noise); geometry jitter is done by the caller"""
    paper = float(np.median(win))
    f = win.astype(np.float32)
    f = paper - rng.uniform(0.6, 1.6) * (paper - f)
    if rng.random() < 0.3:
        f = cv2.GaussianBlur(f, (0, 0), rng.uniform(0.3, 1.0))
    f = f + rng.normal(0, rng.uniform(0, 4), f.shape)
    return np.clip(f, 0, 255).astype(np.uint8)


def sample_window(it, rng):
    """one training window: image crop + per-line targets and prompts (teacher-forced GT points + jitter)"""
    img, pen = it["img"], it["pen"]
    n, W = pen.shape
    w = int(round(WIN_W * rng.uniform(0.9, 1.1)))                     # ~ +-10 % time scale
    if rng.random() < 0.5:                                            # oversample the quake line's big swings
        a = pen[it["active"]]
        dev = np.abs(a - np.nanmedian(a))
        hot = np.flatnonzero(np.nan_to_num(dev, nan=0) > 0.5 * it["S"])
        cc = int(rng.choice(hot)) if len(hot) else int(rng.integers(it["x0"], it["x1"]))
        c = int(np.clip(cc - rng.integers(0, w), it["x0"], it["x1"] - w))
    else:
        c = int(rng.integers(it["x0"], it["x1"] - w + 1))
    top = int(rng.integers(-40, 41)) if rng.random() < 0.5 else 0      # vertical shift (lines at other rows)
    win = augment(crop(img, top, c, w, int(np.median(img))), rng)
    sx, sy = IMG / w, IMG / WIN_H
    pts = np.zeros((n, len(PROMPT_COLS), 2), np.float32)
    lab = -np.ones((n, len(PROMPT_COLS)), np.float32)
    tg = np.zeros((n, LOWRES, LOWRES), np.uint8)
    for k in range(n):
        tg[k] = target_mask(pen[k], top, c, w)
        cols = [0] if rng.random() < 0.35 else [0] + sorted(rng.choice(np.arange(4, 64), int(rng.integers(1, 4)), replace=False).tolist())
        sd = rng.uniform(0, 2.0)
        for i, j in enumerate(cols):
            y = pen[k, c + j]
            if not np.isfinite(y):
                continue
            y = float(np.clip(y + rng.normal(0, sd), 0, img.shape[0] - 1)) - top
            pts[k, i] = (j * sx, np.clip(y, 0, WIN_H - 1) * sy)
            lab[k, i] = 1
    return win, pts, lab, tg


def seg_loss(logits, tg):
    """BCE (positives up-weighted) + soft dice, per mask"""
    t = tg.float()
    bce = F.binary_cross_entropy_with_logits(logits, t, pos_weight=torch.tensor(3.0, device=logits.device))
    p = torch.sigmoid(logits)
    inter = (p * t).flatten(1).sum(1)
    dice = 1 - (2 * inter + 1) / (p.flatten(1).sum(1) + t.flatten(1).sum(1) + 1)
    return bce + dice.mean(), bce.item(), dice.mean().item()


def trainable_setup(model, lora_blocks=None, r=8):
    for p in model.parameters():
        p.requires_grad_(False)
    add_lora(model, r, 2 * r, lora_blocks_for(model.variant) if lora_blocks is None else tuple(lora_blocks))
    groups = {"lora": [], "head": []}
    for name, p in model.named_parameters():
        if "lora_" in name:
            p.requires_grad_(True); groups["lora"].append(p)
        elif name.startswith(("sam_mask_decoder", "image_encoder.neck", "no_mem_embed")) or \
                name.startswith(("sam_prompt_encoder.point_embeddings", "sam_prompt_encoder.not_a_point_embed")):
            p.requires_grad_(True); groups["head"].append(p)
    return groups


def trainable_state(model):
    return {k: v.detach().cpu() for k, v in model.named_parameters() if v.requires_grad}


def load_finetuned(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model, _ = load_sam21(ck["variant"], "cpu")
    trainable_setup(model, ck.get("lora_blocks"), ck.get("lora_r", 8))
    missing = [k for k in ck["state"] if k not in dict(model.named_parameters())]
    assert not missing, missing[:5]
    model.load_state_dict(ck["state"], strict=False)
    return model.to(device).eval(), ck


def train(args):
    dev = get_device()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    tr_recs, va_recs, val_groups = split_records()
    tr = [load_item(r, g) for r, g in items_of(tr_recs)]
    va = [load_item(r, g) for r, g in items_of(va_recs)]
    print(f"train {len(tr_recs)} records / {len(tr)} images; val groups {val_groups} ({len(va_recs)} records / {len(va)} images)", flush=True)
    model, info = load_sam21(args.variant, dev)
    lora_blocks = lora_blocks_for(args.variant, args.lora)
    groups = trainable_setup(model, lora_blocks, args.lora_r)
    model.to(dev)
    amp = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if (args.amp and dev.type == "cuda") else (lambda: torch.autocast("cpu", enabled=False))
    n_tr = sum(p.numel() for g in groups.values() for p in g)
    print(f"trainable params {n_tr / 1e6:.2f}M (LoRA {sum(p.numel() for p in groups['lora']) / 1e6:.2f}M)", flush=True)
    opt = torch.optim.AdamW([{"params": groups["head"], "lr": 2e-4, "weight_decay": 1e-4},
                             {"params": groups["lora"], "lr": 6e-4, "weight_decay": 0.0}])
    base_lr = [g["lr"] for g in opt.param_groups]
    vrng = np.random.default_rng(123)                       # fixed validation windows (same for every check)
    vwins = [sample_window(va[i % len(va)], vrng) for i in range(0, 4 * len(va), 4)][:40]
    budget = args.minutes * 60.0
    t0, step, hist, best = time.time(), 0, [], (1e9, -1)
    tag = args.tag or f"{args.variant}"
    log = open(OUT / "logs" / f"train_{tag}.log", "a")

    def val_loss():
        model.eval()
        tot = []
        with torch.no_grad():
            for win, pts, lab, tg in vwins:
                emb, hr = model.encode(to_tensor_batch([win], dev))
                n = pts.shape[0]
                lo, _ = model.decode(emb, hr, torch.zeros(n, dtype=torch.long, device=dev),
                                     torch.from_numpy(pts).to(dev), torch.from_numpy(lab).to(dev))
                tot.append(seg_loss(lo[:, 0], torch.from_numpy(tg).to(dev))[0].item())
        model.train()
        return float(np.mean(tot))

    v0 = val_loss()
    print(f"val loss before fine-tuning {v0:.4f}", flush=True)
    next_val = 300.0
    model.train()
    while True:
        el = time.time() - t0
        if el > budget:
            break
        frac = el / budget
        for g, b in zip(opt.param_groups, base_lr):
            g["lr"] = b * min(1.0, (step + 1) / 50) * 0.5 * (1 + math.cos(math.pi * frac))
        batch = [sample_window(tr[int(rng.integers(len(tr)))], rng) for _ in range(args.windows)]
        pts = np.concatenate([b[1] for b in batch])
        lab = np.concatenate([b[2] for b in batch])
        tg = np.concatenate([b[3] for b in batch])
        idx = torch.arange(len(batch), device=dev).repeat_interleave(batch[0][1].shape[0])
        with amp():
            emb, hr = model.encode(to_tensor_batch([b[0] for b in batch], dev))
            lo, _ = model.decode(emb, hr, idx, torch.from_numpy(pts).to(dev), torch.from_numpy(lab).to(dev))
        loss, b_, d_ = seg_loss(lo[:, 0].float(), torch.from_numpy(tg).to(dev))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in groups.values() for p in g], 1.0)
        opt.step()
        hist.append((loss.item(), b_, d_))
        step += 1
        if step % 50 == 0:
            h = np.mean(hist[-50:], 0)
            msg = f"step {step} t {el / 60:.1f}m loss {h[0]:.4f} (bce {h[1]:.4f} dice {h[2]:.4f}) lr {opt.param_groups[0]['lr']:.2e}"
            print(msg, flush=True); log.write(msg + "\n"); log.flush()
        if el > next_val or time.time() - t0 > budget:
            v = val_loss()
            next_val += 300.0
            msg = f"  val loss {v:.4f} at step {step}"
            print(msg, flush=True); log.write(msg + "\n"); log.flush()
            if v < best[0]:
                best = (v, step)
                torch.save({"variant": args.variant, "state": trainable_state(model), "step": step, "val_loss": v,
                            "lora_blocks": lora_blocks, "lora_r": args.lora_r}, OUT / f"ckpt_{tag}_best.pt")
    v = val_loss()
    if v < best[0]:
        best = (v, step)
        torch.save({"variant": args.variant, "state": trainable_state(model), "step": step, "val_loss": v,
                    "lora_blocks": lora_blocks, "lora_r": args.lora_r}, OUT / f"ckpt_{tag}_best.pt")
    torch.save({"variant": args.variant, "state": trainable_state(model), "step": step, "val_loss": v,
                "lora_blocks": lora_blocks, "lora_r": args.lora_r}, OUT / f"ckpt_{tag}_last.pt")
    summ = {"variant": args.variant, "weights": info, "steps": step, "minutes": (time.time() - t0) / 60,
            "val_loss_before": v0, "val_loss_last": v, "best": {"val_loss": best[0], "step": best[1]},
            "train_records": [r.name for r in tr_recs], "val_records": [r.name for r in va_recs], "val_groups": val_groups,
            "trainable_params": n_tr, "lora_blocks": lora_blocks, "lora_r": args.lora_r, "windows_per_step": args.windows, "window": [WIN_H, WIN_W], "step_cols": STEP,
            "loss_curve_every50": [list(np.round(np.mean(hist[i:i + 50], 0), 4)) for i in range(0, len(hist), 50)]}
    (OUT / f"train_{tag}.json").write_text(json.dumps(summ, indent=1))
    print(f"done: {step} steps in {summ['minutes']:.1f} min; val loss {v0:.4f} -> {v:.4f} (best {best})", flush=True)


# =============================================================================================== inference + decode
def viterbi(logp, y_start, lam, device, dmax=None):
    """logp (n, H, T) emission; y_start (n,) fixed start rows; transition -lam * dy^2 -> (n, T) int rows"""
    n, H, T = logp.shape
    lp = torch.from_numpy(np.ascontiguousarray(logp, dtype=np.float32)).to(device)
    yy = torch.arange(H, device=device, dtype=torch.float32)
    d = yy[:, None] - yy[None, :]
    trans = -lam * d * d
    if dmax is not None:
        trans = torch.where(d.abs() <= dmax, trans, torch.full_like(trans, -1e9))
    acc = torch.full((n, H), -1e9, device=device)
    ys = torch.as_tensor(np.clip(np.round(y_start), 0, H - 1).astype(np.int64), device=device)
    acc[torch.arange(n, device=device), ys] = 0.0
    acc = acc + lp[:, :, 0]
    back = torch.empty((T, n, H), dtype=torch.int16, device=device)
    for t in range(1, T):
        best, arg = (acc[:, None, :] + trans[None]).max(-1)
        back[t] = arg.to(torch.int16)
        acc = best + lp[:, :, t]
    back = back.cpu().numpy()
    y = np.empty((n, T), np.int64)
    y[:, -1] = acc.argmax(-1).cpu().numpy()
    ar = np.arange(n)
    for t in range(T - 1, 0, -1):
        y[:, t - 1] = back[t, ar, y[:, t]]
    return y


def refine(prob, y, r=3):
    """sub-pixel: probability-weighted mean of rows within +-r of the DP row"""
    n, H, T = prob.shape
    out = y.astype(np.float64).copy()
    off = np.arange(-r, r + 1)
    for k in range(n):
        rows = np.clip(y[k][None, :] + off[:, None], 0, H - 1)
        w = prob[k][rows, np.arange(T)[None, :]] + 1e-6
        out[k] = (rows * w).sum(0) / w.sum(0)
    return out


def emission(prob):
    return np.log(np.clip(prob, 1e-4, 1.0))


@torch.no_grad()
def segment_image(model, img, y0, x0, x1, tops=None, token="single", lam_chain=3e-4, dev=None):
    """chained windows over [x0, x1) for every line. Each line lives in a band of WIN_H rows starting at tops[k]
    (a whole film render: top 0). Returns the stitched probability in band coordinates (n, WIN_H, W)."""
    H, W = img.shape
    n = len(y0)
    tops = np.zeros(n, int) if tops is None else np.asarray(tops, int)
    fill = int(np.median(img))
    starts = sorted(set(list(range(x0, x1 - WIN_W, STEP)) + [x1 - WIN_W]))
    probb = np.zeros((n, WIN_H, W), np.float32)
    wsum = np.zeros(W, np.float32)
    path = np.full((n, W), np.nan)                       # band coordinates
    path[:, x0] = np.asarray(y0, float) - tops
    u = np.arange(WIN_W)
    wcol = (0.15 + 0.85 * np.sin(np.pi * (u + 0.5) / WIN_W)).astype(np.float32)
    sx, sy = IMG / WIN_W, IMG / WIN_H
    for j, c in enumerate(starts):
        cols = [0] if j == 0 else list(PROMPT_COLS)
        pwin = np.zeros((n, WIN_H, WIN_W), np.float32)
        for top in sorted(set(tops.tolist())):
            ks = np.flatnonzero(tops == top)
            emb, hr = model.encode(to_tensor_batch([crop(img, top, c, WIN_W, fill)], dev))
            pts = np.zeros((len(ks), len(cols), 2), np.float32)
            for i, k in enumerate(ks):
                for q, jj in enumerate(cols):
                    pts[i, q] = (jj * sx, np.clip(path[k, c + jj], 0, WIN_H - 1) * sy)
            lab = np.ones((len(ks), len(cols)), np.float32)
            lo, iou = model.decode(emb, hr, torch.zeros(len(ks), dtype=torch.long, device=dev),
                                   torch.from_numpy(pts).to(dev), torch.from_numpy(lab).to(dev))
            sel = lo[:, 0] if token == "single" else lo[torch.arange(len(ks), device=dev), iou.argmax(1)]
            p = torch.sigmoid(F.interpolate(sel[:, None].float(), (WIN_H, WIN_W), mode="bilinear", align_corners=False))[:, 0]
            pwin[ks] = p.cpu().numpy()
        probb[:, :, c:c + WIN_W] += pwin * wcol[None, None]
        wsum[c:c + WIN_W] += wcol
        if j + 1 < len(starts):                          # chain: decode this window to prompt the next one
            path[:, c:c + WIN_W] = viterbi(emission(pwin), path[:, c], lam_chain, dev)
    probb[:, :, x0:x1] /= np.maximum(wsum[x0:x1], 1e-6)[None, None]
    return probb, starts


def decode_paths(probb, y0, x0, x1, lam, dev, tops=None):
    """final Viterbi over the stitched band probabilities -> (n, W) image rows (NaN outside [x0, x1))"""
    n, _, W = probb.shape
    tops = np.zeros(n, int) if tops is None else np.asarray(tops, int)
    out = np.full((n, W), np.nan)
    for top in sorted(set(tops.tolist())):
        ks = np.flatnonzero(tops == top)
        seg = probb[ks][:, :, x0:x1]
        y = viterbi(emission(seg), np.clip(np.asarray(y0, float)[ks] - top, 0, WIN_H - 1), lam, dev)
        out[ks, x0:x1] = refine(seg, y) + top
    return out


# =============================================================================================== evaluation
def b_cache_path(rec_name, g, split):
    """B for holdout3 comes from out/cache_film_holdout3 when the row-anchor experiment has written it (read only)"""
    if split == "holdout3":
        shared = ROOT / "out" / "cache_film_holdout3" / f"{rec_name}_film_g{g:g}.npz"
        if shared.exists():
            return shared
    return OUT / "cache_b" / f"{rec_name}_film_g{g:g}.npz"


def split_items(split):
    if split == "val":
        return items_of(split_records()[1])
    if split == "holdout3":
        return items_of(records(FILM_HOLD))
    raise ValueError(split)


def cache_b(args):
    """proposals A, B (arbiter.proposals with the GT start at x0, exactly as cache_proposals) for val / holdout3"""
    from arbiter import proposals
    shard, nsh = map(int, args.shard.split("/"))
    (OUT / "cache_b").mkdir(parents=True, exist_ok=True)
    for r, g in split_items(args.arg)[shard::nsh]:
        path = b_cache_path(r.name, g, args.arg)
        if path.exists():
            continue
        d = r / "film"
        meta = json.loads((d / "meta.json").read_text())
        img = cv2.imread(str(d / f"heli_g{g:g}.png"), 0)
        pen = np.load(d / f"gt_g{g:g}.npz")["pen"]
        x0 = meta["margin_x"]
        t = time.time()
        A, B, ink = proposals(img, pen[:, x0], x0, img.shape[1] - x0, meta["spacing"])
        np.savez_compressed(OUT / "cache_b" / path.name, A=A, B=B, ink=ink)
        print(f"{r.name} g{g:g} {time.time() - t:.1f}s", flush=True)


def summarize(rows, keys):
    out = {}
    for g in sorted({r["gain"] for r in rows}):
        rr = [r for r in rows if r["gain"] == g]
        out[f"g{g:g}"] = {"n_images": len(rr), **{k: round(float(np.mean([r[k] for r in rr])), 1) for k in keys}}
    out["all"] = {"n_images": len(rows), **{k: round(float(np.mean([r[k] for r in rows])), 1) for k in keys}}
    return out


def overlay(img, pen_k, est_k, b_k, x0, x1, path, c=None, width=700):
    """crop around the largest swing of the line: GT green, SAM red, B blue (2x)"""
    if c is None:
        dev = np.abs(pen_k[x0:x1] - np.nanmedian(pen_k[x0:x1]))
        c = x0 + int(np.nanargmax(dev))
    c0 = int(np.clip(c - width // 2, x0, x1 - width))
    vis = cv2.cvtColor(img[:, c0:c0 + width], cv2.COLOR_GRAY2BGR)
    for q, col in ((pen_k, (0, 170, 0)), (b_k, (255, 80, 0)), (est_k, (0, 0, 255))):
        if q is None:
            continue
        xs = np.arange(c0, c0 + width)
        pts = np.stack([xs - c0, q[xs]], 1)
        pts = pts[np.isfinite(pts[:, 1])].astype(np.int32)
        cv2.polylines(vis, [pts], False, col, 1, cv2.LINE_AA)
    cv2.imwrite(str(path), cv2.resize(vis, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))


def eval_split(args):
    from track_joint import score
    from arbiter import proposals
    split = args.arg                                   # val | holdout3 | zeroshot-val
    base = split.replace("zeroshot-", "")
    dev = get_device()
    if split.startswith("zeroshot"):
        model, info = load_sam21(args.variant, dev)
        model.eval()
        token, desc = "best", {"model": f"SAM 2.1 {args.variant} zero-shot (no fine-tuning)", **info}
    else:
        model, ck = load_finetuned(args.ckpt, dev)
        token, desc = "single", {"model": f"SAM 2.1 {ck['variant']} fine-tuned", "ckpt": str(Path(args.ckpt).resolve().relative_to(ROOT)),
                                 "step": ck["step"], "val_loss": ck["val_loss"]}
    tag = split + (f"_{args.tag}" if args.tag else "")
    figs = OUT / "figs" / tag
    figs.mkdir(parents=True, exist_ok=True)
    pdir = OUT / "probs" / tag
    if base == "val":
        pdir.mkdir(parents=True, exist_ok=True)
    rows, t_all = [], time.time()
    b_src = set()
    for r, g in split_items(base):
        t = time.time()
        d = r / "film"
        meta = json.loads((d / "meta.json").read_text())
        img = cv2.imread(str(d / f"heli_g{g:g}.png"), 0)
        pen = np.load(d / f"gt_g{g:g}.npz")["pen"]
        S, x0 = meta["spacing"], meta["margin_x"]
        x1, n, act = img.shape[1] - x0, pen.shape[0], meta["active_line"]
        y0 = pen[:, x0]
        probb, _ = segment_image(model, img, y0, x0, x1, token=token, dev=dev)
        P = decode_paths(probb, y0, x0, x1, args.lam, dev)
        bp = b_cache_path(r.name, g, base)
        if not bp.exists():
            A, B, ink = proposals(img, y0, x0, x1, S)
            (OUT / "cache_b").mkdir(parents=True, exist_ok=True)
            bp = OUT / "cache_b" / bp.name
            np.savez_compressed(bp, A=A, B=B, ink=ink)
        b_src.add(str(bp.parent.relative_to(ROOT)))
        z = np.load(bp)
        row = {"record": r.name, "group": r.name[:8], "gain": float(g), "active_line": act}
        for name, est in (("SAM", P), ("B", z["B"]), ("A", z["A"])):
            row[name] = score(est, pen, range(n), x0, x1, S)[0]
            row[name + "_quake"] = score(est, pen, [act], x0, x1, S)[0]
        row["SAM_quake_within_3px"] = round(float(np.nanmean(np.abs(P[act, x0:x1] - pen[act, x0:x1]) <= 3) * 100), 1)
        rows.append(row)
        if base == "val":
            np.savez_compressed(pdir / f"{r.name}_g{g:g}.npz", p=np.round(probb[:, :, x0:x1] * 255).astype(np.uint8), x0=x0)
        (OUT / "paths" / tag).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(OUT / "paths" / tag / f"{r.name}_g{g:g}.npz", SAM=P.astype(np.float32))
        if len(rows) <= 3 or r.name == split_items(base)[0][0].name or g in (4.0, 8.0) and len([q for q in rows if q["gain"] == g]) == 1:
            overlay(img, pen[act], P[act], z["B"][act], x0, x1, figs / f"{r.name}_g{g:g}_quake.png")
        print(f"{r.name} g{g:g}: SAM {row['SAM']:.1f}/{row['SAM_quake']:.1f}  B {row['B']:.1f}/{row['B_quake']:.1f}  "
              f"A {row['A']:.1f}/{row['A_quake']:.1f}  ({time.time() - t:.0f}s)", flush=True)
    keys = ["SAM", "B", "A", "SAM_quake", "B_quake", "A_quake", "SAM_quake_within_3px"]
    summ = summarize(rows, keys)
    res = {"split": split, "method": desc, "decode": {"lam": args.lam, "lam_chain": 3e-4, "window": [WIN_H, WIN_W],
           "step": STEP, "prompt_cols": PROMPT_COLS, "start": "GT pen at x0 (layout start), same as A/B"},
           "b_source": sorted(b_src), "metric": "track_joint.score: % of drawn columns missing or |path - pen| > S/2",
           "summary": summ, "rows": rows, "runtime_min": round((time.time() - t_all) / 60, 1)}
    (OUT / f"eval_{tag}.json").write_text(json.dumps(res, indent=1))
    print("gain   | whole image fail %: SAM / B / A | quake line fail %: SAM / B / A | SAM quake within 3px %")
    for g, v in summ.items():
        print(f"{g:6s} | {v['SAM']:5.1f} / {v['B']:5.1f} / {v['A']:5.1f} | {v['SAM_quake']:5.1f} / {v['B_quake']:5.1f} / {v['A_quake']:5.1f} | {v['SAM_quake_within_3px']:5.1f}  (n={v['n_images']})")


def real_metrics(name, paths):
    """real_paired_eval's metric on its stored layout/alignment for each path set; categories from its results"""
    import real_paired_eval as rpe
    from obspy import UTCDateTime
    cfg = rpe.SCANS[name]
    rp = ROOT / "out" / "real_paired"
    res = json.loads((rp / f"{name}_results.json").read_text())
    z = np.load(rp / name / f"{name}_paths.npz")
    bases = z["bases"]
    lay = res["layout"]
    x0, x1, xa, S = lay["x0"], lay["x1"], lay["xa"], rpe.TARGET_S
    t_on, t_off = UTCDateTime(cfg["on"]), UTCDateTime(cfg["off"])
    t_ref, ref, _ = rpe.reference(cfg, t_on)
    al = res["alignment"]["chosen_dp_fit"]
    a_glob = res["amplitude"]["global_scale_px_per_m"]
    t_lo, t_hi = 0.0, float(t_off - t_on)
    cats = {l["line"]: l["category"] for l in res["lines"]}
    paths = {**paths, "B_recomputed": z["B"]}
    per = {m: {} for m in paths}
    for k in sorted(cats):
        idx, _, _ = rpe.sample_path(z["B"], k, x0, x1, xa, al, t_ref, t_lo, t_hi)
        if idx.size < 1200:
            continue
        tk, rf = t_ref[idx], ref[idx]
        for m, Pm in paths.items():
            _, y, _ = rpe.sample_path(Pm, k, x0, x1, xa, al, t_ref, t_lo, t_hi)
            per[m][k] = rpe.line_metrics(-(y - bases[k]) * al["sign"], rf, tk, S, a_glob)
    keys = ("r_band", "in_0.1S_global", "lost_0.5S_global", "in_0.1S", "lost_0.5S", "rms_err_global_S")
    summ = {}
    for cat in ("quake", "neighbour", "other_active", "quiet"):
        ks = [k for k in per["B_recomputed"] if cats[k] == cat]
        if not ks:
            continue
        summ[cat] = {"lines": ks}
        for m in paths:
            summ[cat][m] = {key: round(float(np.mean([per[m][k][key] for k in ks])), 3) for key in keys}
        for m in ("A", "B", "judge", "DP38"):
            summ[cat][m + "_stored"] = {key: res["summary"][cat][m][key] for key in keys}
    check = {cat: all(abs(v["B_recomputed"][key] - v["B_stored"][key]) < 2e-3 for key in keys) for cat, v in summ.items()}
    return summ, check, per, res


def eval_real(args):
    name = args.arg.upper()
    rp = ROOT / "out" / "real_paired"
    if args.check_only:
        summ, check, _, _ = real_metrics(name, {})
        print(f"[{name}] B metric reproduced from stored paths: {check}")
        return
    img = cv2.imread(str(rp / name / f"{name}_rescaled.png"), 0)       # exactly real_paired_eval's img_s
    z = np.load(rp / name / f"{name}_paths.npz")
    bases, y0 = z["bases"], z["y0"]
    res = json.loads((rp / f"{name}_results.json").read_text())
    x0, x1 = res["layout"]["x0"], res["layout"]["x1"]
    H, W = img.shape
    tops = np.clip(np.round(bases).astype(int) - WIN_H // 2, 0, H - WIN_H)    # each line near the band centre
    dev = get_device()
    model, ck = load_finetuned(args.ckpt, dev)
    t = time.time()
    probb, _ = segment_image(model, img, y0, x0, x1, tops=tops, dev=dev)
    P = decode_paths(probb, y0, x0, x1, args.lam, dev, tops=tops)
    print(f"[{name}] segmented {len(bases)} lines in {time.time() - t:.0f}s", flush=True)
    summ, check, per, res = real_metrics(name, {"SAM": P})
    out = {"scan": name, "ckpt": str(Path(args.ckpt).resolve().relative_to(ROOT)), "lam": args.lam,
           "layout": "out/real_paired/<scan>_rescaled.png + _paths.npz (bases, y0), alignment chosen_dp_fit, global amplitude",
           "band_tops": tops.tolist(), "B_metric_reproduced": check, "summary": summ,
           "per_line_SAM": {str(k): v for k, v in per["SAM"].items()}}
    (OUT / f"real_{name}.json").write_text(json.dumps(out, indent=1))
    np.savez_compressed(OUT / f"real_{name}_paths.npz", SAM=P, tops=tops)
    figs = OUT / "figs" / "real"
    figs.mkdir(parents=True, exist_ok=True)
    km = res["main_shock_line"]
    overlay(img, np.full(W, np.nan), P[km], z["B"][km], x0, x1, figs / f"{name}_quake_line{km}.png",
            c=int(np.nanargmax(np.abs(P[km, x0:x1] - bases[km]))) + x0)
    print(f"[{name}] B metric reproduced from stored paths: {check}")
    for cat, v in summ.items():
        print(f"[{name}] {cat:12s} {v['lines']} | " + " | ".join(
            f"{m}: r {v[m]['r_band']:+.2f} in {v[m]['in_0.1S_global']:.2f} lost {v[m]['lost_0.5S_global']:.2f}"
            for m in ("SAM", "B_stored", "judge_stored", "DP38_stored")), flush=True)


def analyze(args):
    """failure analysis on saved paths: quake-line fail by GT slope, crossing zones, off-image columns, long swaps"""
    from track_joint import touch_cols
    tag, base = args.arg, args.arg.split("_")[0].replace("zeroshot-", "")
    bins = [0, 2, 10, 40, 1e9]
    acc = {}
    for r, g in split_items(base):
        f = OUT / "paths" / tag / f"{r.name}_g{g:g}.npz"
        if not f.exists():
            continue
        d = r / "film"
        meta = json.loads((d / "meta.json").read_text())
        pen = np.load(d / f"gt_g{g:g}.npz")["pen"]
        S, x0 = meta["spacing"], meta["margin_x"]
        x1, k, H = pen.shape[1] - x0, meta["active_line"], WIN_H
        ests = {"SAM": np.load(f)["SAM"], "B": np.load(b_cache_path(r.name, g, base))["B"]}
        gt = pen[k, x0:x1]
        slope = np.abs(np.gradient(gt))
        offimg = (gt < 0) | (gt > H - 1)
        zone = cv2.dilate(touch_cols(pen, k)[x0:x1].astype(np.uint8)[None], np.ones((1, 81), np.uint8))[0] > 0
        a = acc.setdefault(f"g{g:g}", {"n": 0, "cols": 0, "offimg": 0, "zone": 0, "slope_cols": [0] * 4,
                                        **{m: {"fail": 0, "fail_onimg": 0, "zone_fail": 0, "slope_fail": [0] * 4,
                                               "fail_in_runs_ge100": 0} for m in ests}})
        a["n"] += 1; a["cols"] += len(gt); a["offimg"] += int(offimg.sum()); a["zone"] += int(zone.sum())
        sb = np.digitize(slope, bins) - 1
        for b_ in range(4):
            a["slope_cols"][b_] += int((sb == b_).sum())
        for m, E in ests.items():
            e = E[k, x0:x1]
            fail = np.isnan(e) | (np.abs(np.nan_to_num(e, nan=1e9) - gt) > S / 2)
            a[m]["fail"] += int(fail.sum()); a[m]["fail_onimg"] += int((fail & ~offimg).sum())
            a[m]["zone_fail"] += int((fail & zone).sum())
            for b_ in range(4):
                a[m]["slope_fail"][b_] += int((fail & (sb == b_)).sum())
            edges = np.flatnonzero(np.diff(np.r_[0, fail.astype(int), 0]))
            a[m]["fail_in_runs_ge100"] += int(sum(e2 - s2 for s2, e2 in zip(edges[::2], edges[1::2]) if e2 - s2 >= 100))
    out = {}
    for gk, a in acc.items():
        o = {"images": a["n"], "quake_cols_offimage_pct": round(100 * a["offimg"] / a["cols"], 1),
             "touch_zone_cols_pct": round(100 * a["zone"] / a["cols"], 1),
             "slope_bins_px_per_col": ["<2", "2-10", "10-40", ">40"],
             "slope_bin_cols_pct": [round(100 * v / a["cols"], 1) for v in a["slope_cols"]]}
        for m in ("SAM", "B"):
            o[m] = {"quake_fail_pct": round(100 * a[m]["fail"] / a["cols"], 1),
                    "quake_fail_on_image_cols_pct": round(100 * a[m]["fail_onimg"] / max(1, a["cols"] - a["offimg"]), 1),
                    "touch_zone_fail_pct": round(100 * a[m]["zone_fail"] / max(1, a["zone"]), 1),
                    "fail_pct_by_slope_bin": [round(100 * f_ / max(1, c_), 1) for f_, c_ in zip(a[m]["slope_fail"], a["slope_cols"])],
                    "share_of_fails_in_runs_ge100_cols": round(100 * a[m]["fail_in_runs_ge100"] / max(1, a[m]["fail"]), 1)}
        out[gk] = o
    (OUT / f"analysis_{tag}.json").write_text(json.dumps(out, indent=1))
    for gk, o in out.items():
        print(gk, json.dumps(o))


def tune_lam(args):
    """final-DP smoothness on the cached validation probabilities (validation groups only)"""
    from track_joint import score
    dev = get_device()
    tag = args.arg
    res = {}
    lams = [float(v) for v in args.lams.split(",")]
    for r, g in split_items("val"):
        f = OUT / "probs" / tag / f"{r.name}_g{g:g}.npz"
        z = np.load(f)
        d = r / "film"
        meta = json.loads((d / "meta.json").read_text())
        pen = np.load(d / f"gt_g{g:g}.npz")["pen"]
        x0 = int(z["x0"])
        x1 = pen.shape[1] - x0
        probb = np.zeros((pen.shape[0], WIN_H, pen.shape[1]), np.float32)
        probb[:, :, x0:x1] = z["p"].astype(np.float32) / 255.0
        for lam in lams:
            P = decode_paths(probb, pen[:, x0], x0, x1, lam, dev)
            res.setdefault(lam, []).append({"gain": float(g), "SAM": score(P, pen, range(pen.shape[0]), x0, x1, meta["spacing"])[0],
                                            "SAM_quake": score(P, pen, [meta["active_line"]], x0, x1, meta["spacing"])[0]})
    out = {str(l): summarize(v, ["SAM", "SAM_quake"]) for l, v in res.items()}
    (OUT / f"tune_lam_{tag}.json").write_text(json.dumps(out, indent=1))
    for l, v in out.items():
        print(f"lam {l}: " + "  ".join(f"{g} {s['SAM']:.1f}/{s['SAM_quake']:.1f}" for g, s in v.items()))


# =============================================================================================== CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("arg", nargs="?")
    ap.add_argument("--variant", default="tiny")
    ap.add_argument("--minutes", type=float, default=40.0)
    ap.add_argument("--ckpt", default=str(OUT / "ckpt_tiny_best.pt"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--lam", type=float, default=3e-4)
    ap.add_argument("--lams", default="3e-5,1e-4,3e-4,1e-3,3e-3")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--windows", type=int, default=1, help="training windows per step (x 8 line prompts each)")
    ap.add_argument("--lora", default="stages23", choices=["stages23", "all"])
    ap.add_argument("--lora-r", dest="lora_r", type=int, default=8)
    ap.add_argument("--amp", action="store_true", help="bf16 autocast (CUDA only)")
    ap.add_argument("--check-only", dest="check_only", action="store_true", help="real: only re-score the stored B")
    args = ap.parse_args()
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    {"selftest": selftest, "train": train, "cache-b": cache_b, "eval": eval_split, "real": eval_real,
     "tune-lam": tune_lam, "analyze": analyze}[args.cmd](args)


if __name__ == "__main__":
    main()

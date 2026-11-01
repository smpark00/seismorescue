"""SAM 3 (Meta, gated open weights, SAM License) fine-tuned as the same promptable helicorder trace segmenter as the
SAM 2.1 experiment (scripts/ml_sam.py, research/28). Only the model is swapped (research/32).

usage: .venv-sam3/bin/python -I scripts/ml_sam3.py <command> [arg] [options]
  smoke [--layers N]            NO gated weights: random-init SAM 3 tracker built from the default config. Shapes on a real
                                film window, parity of this file's encode/decode glue with transformers' own forward,
                                ml_sam.segment_image + decode_paths on a real film crop, LoRA train steps, checkpoint
                                round trip -> out/ml/sam3/smoke.json (CPU by default; does not touch the GPU)
  bench [--device D --steps N --windows W --lora L --amp --grad-ckpt]
                                NO gated weights: seconds per train step / encoder pass and peak memory on any device
                                (run it on the 5070 Ti or Modal before the weights arrive) -> out/ml/sam3/bench_*.json
  selftest                      gated weights: load the facebook/sam3 tracker, zero-shot point-prompt sanity masks, timing
  train [--minutes M --windows N --lora last16|lastK|all|none --lora-r R --amp --grad-ckpt --tag T]
                                fine-tune on the training groups of out/film_train -> out/ml/sam3/ckpt_<tag>_{best,last}.pt
  eval <val|holdout3|holdout4|holdout5|zeroshot-val> [--ckpt PATH --lam L]   per-line fail % vs A/B -> eval_<split>[_tag].json
  tune-lam <probs tag> [--lams ...]   ml_sam.tune_lam on out/ml/sam3/probs/<tag> (validation groups only)
  analyze <paths tag>           ml_sam.analyze on out/ml/sam3/paths/<tag>
  freeze --ckpt PATH --lam L    record the method chosen on validation (checkpoint sha256, lam, amp). `eval holdout3`
                                refuses to run without it, with any other checkpoint file / lam / amp, with --limit or
                                --random-init, as zero-shot, or a second time (any tag; a start marker is written before
                                the first image). freeze itself refuses once holdout3 or holdout5 has been touched.
  Evaluation (val loss, eval, real) runs in fp32 like ml_sam; --amp (bf16) is meant for the train steps. The point and
  image positional encodings are computed in fp32 even under autocast.
  real <KIP|MAT|ALQ> [--ckpt PATH --lam L]   real paired scan, real_paired_eval's layout/alignment/metrics
                                             (--paths FILE: score SAM paths made elsewhere by real-paths)
  real-paths <MAT|ALQ> [--lines 3,8,...]     research/49 §4: paths only, from out/ml/real_inputs/<name>.npz (public chips)
  --random-init [--layers N] on train / eval: dry run of the whole pipeline with random weights (tag "dryrun"); --limit N
                                evaluates only the first N images. Never a result; refused for holdout3 and holdout5.
  --device cpu|mps|cuda also sets the device ml_sam's own functions use (tune-lam, analyze, Viterbi).

Setup (separate venv; research/seismo_exp/.venv is not touched):
  uv venv .venv-sam3 --python 3.12 && uv pip install --python .venv-sam3/bin/python torch==2.14.1 torchvision \
      numpy==2.5.4 scipy==1.18.1 opencv-python-headless==5.0.0.93 obspy==1.5.1 transformers==5.21.0 safetensors \
      huggingface_hub pillow
  After the Hugging Face request for facebook/sam3 is approved: `hf auth login` (or export HF_TOKEN). The first load
  downloads model.safetensors (3.44 GB, the whole SAM 3 video model; only the tracker part is kept in memory).

Model: the single-image point-prompt (PVS / "SAM 1 task") path of SAM 3 = transformers `Sam3TrackerModel`:
PE ViT trunk (32 layers, width 1024, patch 14, 1008 px -> 72 x 72 tokens, windowed attention 24 with global attention in
layers 7/15/23/31, RoPE; 446.2M params) + tracker FPN neck (strides 3.5/7/14) + SAM 2-style prompt encoder and two-way
mask decoder (4.2M) giving 288 x 288 mask logits. `SAM3Image` below exposes ml_sam.SAM21Image's encode/decode interface,
so ml_sam's windowing, prompting, chaining, Viterbi decoding, scoring and real-scan metrics run unchanged.
Preprocessing follows the official SAM 1-task predictor (sam3/model/utils/sam1_utils.py): plain resize to 1008 x 1008,
mean = std = 0.5; no_mem_embed added to the 72 x 72 map; the first two FPN levels feed the decoder's high-res path.

What changes vs. SAM 2.1. Forced by the model: input 1008 instead of 1024, decoder grid 288 instead of 256 (target line
drawn 3 cells wide = +-2.1 px instead of +-2.4 px), normalisation 0.5/0.5 instead of ImageNet, stride-14 ViT tokens
(upsampled by the neck) instead of Hiera's stride-4 features. Chosen, not forced: LoRA on "the last 16 of 32 blocks"
(separate q/k/v/o, 1.05M) for memory (`all` + --grad-ckpt also fits), and the Modal run's training amount and precision
(4 windows/step for 60 min in TF32 on an H100, ~6x the windows of research/28's 1 window/step for 45 min in fp32 on MPS).
research/34 lists these differences and the controls.
Everything else (data split, window sampler, augmentation, prompts, loss, optimiser, LRs, schedule, validation windows,
decoding, lam grid, metrics) is ml_sam's code.

SAM 3.1 (facebook/sam3.1): not supported here. That repo ships only the official-format `sam3.1_multiplex.pt` (no
transformers weights, model card: no Transformers integration); its tracker uses the Object Multiplex decoder and the
official code needs CUDA. See research/32 for the options.
"""
import argparse
import json
import math
import os
import resource
import sys
import time
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import ml_sam as S  # noqa: E402  (data, windowing, prompting, loss, LoRA, chaining, Viterbi, scoring, real metrics)

OUT3 = ROOT / "out" / "ml" / os.environ.get("SEISMO_SAM3_OUT", "sam3")   # sam3_film2: research/37 (own freeze and test marks)
SAM21_OUT = ROOT / "out" / "ml" / "sam"
HF_ID = "facebook/sam3"

# ---- SAM 3 geometry, applied to ml_sam's module constants so its functions (sample_window, target_mask,
#      to_tensor_batch, segment_image) produce 1008-frame prompts, 288-grid targets and 0.5/0.5-normalised inputs.
IMG3, LOWRES3, MEAN3, STD3 = 1008, 288, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
S.IMG, S.LOWRES, S.PIX_MEAN, S.PIX_STD = IMG3, LOWRES3, MEAN3, STD3
S.OUT = OUT3                     # ml_sam.tune_lam / ml_sam.analyze read and write under out/ml/sam3


def b_cache_path(rec_name, g, split):
    """B proposals for the same images, read only from the caches the SAM 2.1 run used (shared holdout3 cache, then
    out/ml/sam/cache_b); only missing ones are computed, into out/ml/sam3/cache_b"""
    name = f"{rec_name}_film_g{g:g}.npz"
    cands = ([S.TEST[split][1] / name] if split in S.TEST else []) + \
        ([SAM21_OUT / "cache_b" / name] if S.FILM_TRAIN.name == "film_train" else [])   # v1 val caches fit v1 renders only
    for p in cands:
        if p.exists():
            return p
    return OUT3 / "cache_b" / name


S.b_cache_path = b_cache_path    # used by ml_sam.analyze


# =============================================================================================== model
class SAM3Image(nn.Module):
    """SAM 3 tracker PVS head behind ml_sam.SAM21Image's interface:
    encode(x (B,3,1008,1008)) -> emb (B,256,72,72), hr ((B,32,288,288), (B,64,144,144))
    decode(emb, hr, idx (N,), pts (N,P,2) 1008-frame px, labels (N,P) 1/0/-1) -> logits (N,K,288,288), iou (N,K)
    K = 1 (token 0, the single-mask output; training and fine-tuned eval) or 3 (multimask tokens; zero-shot)."""

    def __init__(self, tracker, family="sam3"):
        super().__init__()
        self.m = tracker
        self.variant = family
        self.multimask = False
        tracker.mask_decoder.dynamic_multimask_via_stability = False      # token 0 always, as ml_sam's lo[:, 0]

    def encode(self, x):
        fpn = self.m.vision_encoder(x).fpn_hidden_states          # 288, 144, 72 (and 36 if the neck has 4 levels)
        emb = fpn[2] + self.m.no_memory_embedding.view(1, -1, 1, 1)
        return emb, (self.m.mask_decoder.conv_s0(fpn[0]), self.m.mask_decoder.conv_s1(fpn[1]))

    def decode(self, emb, hr, idx, pts, labels):
        n = pts.shape[0]
        with torch.autocast(pts.device.type, enabled=False):     # Fourier point / image encodings stay fp32 under bf16
            sparse, dense = self.m.prompt_encoder(pts[:, None].float(), labels[:, None].long(), None, None)
            pe = self.m.get_image_wide_positional_embeddings().repeat(n, 1, 1, 1)
        masks, iou, _, _ = self.m.mask_decoder(image_embeddings=emb[idx], image_positional_embeddings=pe,
                                               sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
                                               multimask_output=self.multimask,
                                               high_resolution_features=[hr[0][idx], hr[1][idx]])
        return masks[:, 0], iou[:, 0]


def tracker_config(layers=32):
    """default Sam3TrackerConfig = the released SAM 3 tracker (the default Sam3VideoConfig has 859,921,848 params +
    512 buffer values = the 859,922,360 tensors in facebook/sam3 model.safetensors); 3-level neck as used by the tracker"""
    from transformers import Sam3TrackerConfig
    cfg = Sam3TrackerConfig()
    cfg.vision_config.scale_factors = [4.0, 2.0, 1.0]
    bb = cfg.vision_config.backbone_config
    if layers != bb.num_hidden_layers:                       # reduced depth for smoke tests only, same global pattern
        bb.num_hidden_layers = layers
        bb.global_attn_indexes = [i for i in range(layers) if i % 8 == 7]
    return cfg


def build_random(layers=32, seed=0):
    from transformers import Sam3TrackerModel
    torch.manual_seed(seed)
    return SAM3Image(Sam3TrackerModel(tracker_config(layers)).float())


USED = ("vision_encoder.backbone.", "vision_encoder.neck.fpn_layers.0.", "vision_encoder.neck.fpn_layers.1.",
        "vision_encoder.neck.fpn_layers.2.", "prompt_encoder.", "mask_decoder.", "no_memory_embedding",
        "shared_image_embedding.")


WEIGHTS_HELP = """SAM 3 base weights ({hf_id}, revision {revision}) are not available on this machine.
They are gated by Meta, so this repository does not include them. Reason given: {why}
To get them:
  1. Request access with a Hugging Face account: https://huggingface.co/{hf_id}
  2. .venv/bin/hf auth login
  3. HF_HUB_OFFLINE=0 .venv/bin/hf download {hf_id} config.json model.safetensors --revision {revision}
To run without them, use the recorded paths and clicks: rescue.py --reuse, review_server.py --replay (see the README)."""


def resolve_revision(hf_id, revision=None):
    """pin the base weights to a commit sha (stored in every fine-tuned checkpoint and reused when it is loaded)"""
    try:
        from huggingface_hub import HfApi
        return HfApi().model_info(hf_id, revision=revision).sha
    except Exception as e:                                    # offline: keep what was asked for, say so
        print(f"warning: could not resolve the {hf_id} commit ({type(e).__name__}); using revision={revision}", flush=True)
        return revision


def load_pretrained(hf_id=HF_ID, device="cpu", revision=None):
    """gated weights -> SAM3Image; every parameter of the image path must come from the checkpoint"""
    if "sam3.1" in hf_id:
        raise SystemExit("facebook/sam3.1 has no transformers weights (only the official sam3.1_multiplex.pt); "
                         "see research/32 for the SAM 3.1 options")
    from transformers import Sam3TrackerModel
    revision = resolve_revision(hf_id, revision)
    try:
        tracker, info = Sam3TrackerModel.from_pretrained(hf_id, revision=revision, dtype=torch.float32,
                                                         output_loading_info=True)
    except OSError as e:
        raise SystemExit(WEIGHTS_HELP.format(hf_id=hf_id, revision=revision, why=str(e).splitlines()[0][:200])) from e
    missing = [k for k in info.get("missing_keys", []) if k.startswith(USED)]
    mism = [k for k in info.get("mismatched_keys", [])]
    assert not missing and not mism, f"checkpoint mismatch: missing {missing[:5]} mismatched {mism[:5]}"
    model = SAM3Image(tracker.float())
    levels = len(tracker.config.vision_config.scale_factors)
    return model.to(device), {"weights": hf_id, "revision": revision, "neck_levels_in_config": levels,
                              "missing_keys": list(info.get("missing_keys", [])),
                              "n_unexpected_keys": len(info.get("unexpected_keys", [])),
                              "params_image_path": sum(p.numel() for p in model.parameters())}


def n_layers(model):
    return len(model.m.vision_encoder.backbone.layers)


def lora_blocks_for(scope, n):
    """'last16' (default: the deeper half of the 32-block trunk), 'last<K>', 'all' or 'none'"""
    if scope == "all":
        return tuple(range(n))
    if scope == "none":
        return ()
    k = int(scope.replace("last", ""))
    return tuple(range(max(0, n - k), n))


def add_lora(model, r, alpha, blocks):
    """LoRA (ml_sam.LoRALinear) on q/k/v/o of the given trunk blocks; base weights frozen"""
    for i in blocks:
        att = model.m.vision_encoder.backbone.layers[i].attention
        for nm in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(att, nm, S.LoRALinear(getattr(att, nm), r, alpha))


HEAD = ("m.mask_decoder.", "m.vision_encoder.neck.fpn_layers.0.", "m.vision_encoder.neck.fpn_layers.1.",
        "m.vision_encoder.neck.fpn_layers.2.", "m.no_memory_embedding", "m.prompt_encoder.point_embed.",
        "m.prompt_encoder.not_a_point_embed.")


def trainable_setup(model, lora_blocks, r=8):
    """as ml_sam.trainable_setup: mask decoder (incl. conv_s0/s1), neck, point / not-a-point embeddings and the
    no-memory embedding are trained fully; LoRA r on the chosen trunk blocks; everything else frozen"""
    for p in model.parameters():
        p.requires_grad_(False)
    add_lora(model, r, 2 * r, lora_blocks)
    groups = {"lora": [], "head": []}
    for name, p in model.named_parameters():
        if "lora_" in name:
            p.requires_grad_(True); groups["lora"].append(p)
        elif name.startswith(HEAD):
            p.requires_grad_(True); groups["head"].append(p)
    return groups


def enable_grad_ckpt(model, blocks):
    """activation checkpointing per trunk block (transformers' GradientCheckpointingLayer hook; Sam3 models do not set
    supports_gradient_checkpointing, so it is switched on per layer)"""
    from torch.utils.checkpoint import checkpoint
    for i in blocks:
        layer = model.m.vision_encoder.backbone.layers[i]
        layer.gradient_checkpointing = True
        layer._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)


def load_base(args):
    """gated SAM 3 weights, or (--random-init, pipeline dry runs only) the default config with random weights"""
    if args.random_init:
        return build_random(args.layers), {"weights": "RANDOM INIT (pipeline dry run, not a result)", "layers": args.layers}
    return load_pretrained(args.hf_id, "cpu")


def load_finetuned(path, device, hf_id=None):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if ck.get("random_init"):
        model = build_random(ck["layers"])
    else:
        model, _ = load_pretrained(hf_id or ck["hf_id"], "cpu", ck.get("revision"))
    trainable_setup(model, ck["lora_blocks"], ck["lora_r"])
    own = dict(model.named_parameters())
    bad = [k for k in ck["state"] if k not in own or own[k].shape != ck["state"][k].shape]
    assert not bad, bad[:5]
    model.load_state_dict(ck["state"], strict=False)
    return model.to(device).eval(), ck


def get_device(name="auto"):
    return S.get_device() if name == "auto" else torch.device(name)


def rel_root(p):
    """p relative to ROOT without following symlinks (on Modal out/ is a symlink into /__modal/volumes)"""
    return str(Path(os.path.abspath(p)).relative_to(ROOT))


def amp_ctx(dev, on):
    return torch.autocast(dev.type, dtype=torch.bfloat16) if on else torch.autocast(dev.type, enabled=False)


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()


def peak_mem_gb(dev):
    if dev.type == "cuda":
        return {"cuda_max_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                "cuda_max_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2)}
    if dev.type == "mps":
        return {"mps_current_allocated_gb": round(torch.mps.current_allocated_memory() / 1e9, 2),
                "mps_driver_allocated_gb": round(torch.mps.driver_allocated_memory() / 1e9, 2)}
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"cpu_peak_rss_gb": round(rss / (1e9 if sys.platform == "darwin" else 1e6), 2)}


def param_report(model):
    def cnt(prefix):
        return sum(p.numel() for n, p in model.named_parameters() if n.startswith(prefix))
    return {"total": cnt(""), "trunk": cnt("m.vision_encoder.backbone."), "neck": cnt("m.vision_encoder.neck."),
            "prompt_encoder": cnt("m.prompt_encoder."), "mask_decoder": cnt("m.mask_decoder."),
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "lora": sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)}


def batch_tensors(batch, dev):
    pts = np.concatenate([b[1] for b in batch])
    lab = np.concatenate([b[2] for b in batch])
    tg = np.concatenate([b[3] for b in batch])
    idx = torch.arange(len(batch), device=dev).repeat_interleave(batch[0][1].shape[0])
    return (S.to_tensor_batch([b[0] for b in batch], dev), idx, torch.from_numpy(pts).to(dev),
            torch.from_numpy(lab).to(dev), torch.from_numpy(tg).to(dev))


def train_step(model, opt, groups, batch, dev, amp):
    x, idx, pts, lab, tg = batch_tensors(batch, dev)
    with amp_ctx(dev, amp):
        emb, hr = model.encode(x)
        lo, _ = model.decode(emb, hr, idx, pts, lab)
    loss, b_, d_ = S.seg_loss(lo[:, 0].float(), tg)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_([p for g in groups.values() for p in g], 1.0)
    if not (torch.isfinite(gn) and torch.isfinite(loss)):   # skip: clipping by a non-finite norm writes NaN everywhere
        opt.zero_grad(set_to_none=True)
        return None
    opt.step()
    return loss.item(), b_, d_


def nonfinite_report(model, groups, batch, dev, amp):
    """the first skipped step's batch replayed in the training precision and in strict fp32 (no bf16, no TF32): which
    tensor goes non-finite first, and the batch's prompts / targets (once per run, -> out/ml/sam3/nonfinite_<tag>.json)"""
    x, idx, pts, lab, tg = batch_tensors(batch, dev)
    rep = {"labelled_points_per_mask": (lab > 0).sum(1).tolist(), "target_pixels_per_mask": tg.flatten(1).sum(1).tolist(),
           "pts_min_max": [float(pts.min()), float(pts.max())], "input_finite": bool(torch.isfinite(x).all())}
    names = {id(p): n for n, p in model.named_parameters()}
    tf32 = torch.backends.cuda.matmul.allow_tf32
    train_prec = "bf16" if amp else ("tf32" if tf32 and dev.type == "cuda" else "fp32")
    for name, on, mm_tf32 in ((train_prec, amp, tf32), ("fp32_strict", False, False)):
        model.zero_grad(set_to_none=True)
        torch.backends.cuda.matmul.allow_tf32 = mm_tf32
        try:
            with amp_ctx(dev, on):
                emb, hr = model.encode(x)
                lo, _ = model.decode(emb, hr, idx, pts, lab)
            loss = S.seg_loss(lo[:, 0].float(), tg)[0]
            loss.backward()
        finally:
            torch.backends.cuda.matmul.allow_tf32 = tf32
        bad = [names[id(p)] for g in groups.values() for p in g if p.grad is not None and not torch.isfinite(p.grad).all()]
        rep[name] = {"emb_finite": bool(torch.isfinite(emb).all()), "hr_finite": [bool(torch.isfinite(h).all()) for h in hr],
                     "logits_finite": bool(torch.isfinite(lo).all()), "emb_absmax": float(emb.detach().float().abs().max()),
                     "logits_absmax": float(lo.detach().float().abs().max()), "loss": float(loss.detach()),
                     "n_params_nonfinite_grad": len(bad), "params_nonfinite_grad_first": bad[:10]}
    model.zero_grad(set_to_none=True)
    return rep


def make_opt(groups):
    return torch.optim.AdamW([{"params": groups["head"], "lr": 2e-4, "weight_decay": 1e-4},
                              {"params": groups["lora"], "lr": 6e-4, "weight_decay": 0.0}])


def first_train_item():
    tr_recs, _, _ = S.split_records()
    r, g = S.items_of(tr_recs)[0]
    return S.load_item(r, g)


# =============================================================================================== smoke (no weights)
def smoke(args):
    dev = get_device(args.device if args.device != "auto" else "cpu")
    rep = {"device": str(dev), "torch": torch.__version__, "layers": args.layers, "note": "random init, no gated weights"}
    import transformers
    rep["transformers"] = transformers.__version__
    t0 = time.time()
    model = build_random(args.layers).to(dev).eval()
    rep["build_s"] = round(time.time() - t0, 1)
    rep["params"] = param_report(model)
    rep["config"] = {"image_size": model.m.config.vision_config.backbone_config.image_size,
                     "patch": model.m.config.vision_config.backbone_config.patch_size,
                     "global_attn": model.m.config.vision_config.backbone_config.global_attn_indexes,
                     "window": model.m.config.vision_config.backbone_config.window_size,
                     "neck_scale_factors": model.m.config.vision_config.scale_factors}
    print(json.dumps(rep), flush=True)

    # 1. shapes on a real training window, 8 line prompts x 4 points (unused slots label -1, as ml_sam)
    it = first_train_item()
    rng = np.random.default_rng(0)
    win, pts, lab, tg = S.sample_window(it, rng)
    x = S.to_tensor_batch([win], dev)
    n = pts.shape[0]
    idx = torch.zeros(n, dtype=torch.long, device=dev)
    P, L = torch.from_numpy(pts).to(dev), torch.from_numpy(lab).to(dev)
    with torch.no_grad():
        t = time.time()
        emb, hr = model.encode(x)
        sync(dev)
        t_enc = time.time() - t
        lo, iou = model.decode(emb, hr, idx, P, L)
        model.multimask = True
        lo3, iou3 = model.decode(emb, hr, idx, P, L)
        model.multimask = False
    shp = {"input": list(x.shape), "emb": list(emb.shape), "hr0": list(hr[0].shape), "hr1": list(hr[1].shape),
           "logits_single": list(lo.shape), "iou_single": list(iou.shape), "logits_multi": list(lo3.shape),
           "target": list(tg.shape), "prompt_pts": list(pts.shape), "labels_used": sorted(set(lab.ravel().tolist())),
           "prompt_max_coord": round(float(pts.max()), 1)}
    assert list(emb.shape) == [1, 256, 72, 72] and list(hr[0].shape) == [1, 32, 288, 288] and list(hr[1].shape) == [1, 64, 144, 144]
    assert list(lo.shape) == [n, 1, 288, 288] and list(lo3.shape) == [n, 3, 288, 288] and list(tg.shape) == [n, 288, 288]
    assert pts.max() < IMG3
    rep["shapes"] = shp
    rep["encoder_fwd_s"] = round(t_enc, 2)
    print("shapes ok", json.dumps(shp), f"encoder {t_enc:.1f}s", flush=True)

    # 2. parity: this file's encode/decode vs transformers' Sam3TrackerModel.forward on the same inputs
    with torch.no_grad():
        out1 = model.m(pixel_values=x, input_points=P[None], input_labels=L[None].long(), multimask_output=False)
        out3 = model.m(pixel_values=x, input_points=P[None], input_labels=L[None].long(), multimask_output=True)
    d1 = float((out1.pred_masks[0] - lo).abs().max())
    d3 = float((out3.pred_masks[0] - lo3).abs().max())
    di = float((out3.iou_scores[0] - iou3).abs().max())
    rep["parity_vs_transformers_forward"] = {"single_mask_max_abs_diff": d1, "multimask_max_abs_diff": d3,
                                             "iou_max_abs_diff": di, "logit_scale": float(lo.abs().max())}
    assert d1 < 1e-3 and d3 < 1e-3, rep["parity_vs_transformers_forward"]
    print("parity ok", rep["parity_vs_transformers_forward"], flush=True)

    # 3. ml_sam's chained windows + Viterbi on a real film crop (3 windows)
    d = it["img"]
    x0 = it["x0"]
    x1 = x0 + S.WIN_W + 2 * S.STEP
    y0 = it["pen"][:, x0]
    t = time.time()
    probb, starts = S.segment_image(model, d, y0, x0, x1, token="single", dev=dev)
    Pth = S.decode_paths(probb, y0, x0, x1, 1e-5, dev)
    rep["segment_image"] = {"windows": len(starts), "probb_shape": list(probb.shape), "paths_shape": list(Pth.shape),
                            "paths_finite_frac": float(np.isfinite(Pth[:, x0:x1]).mean()),
                            "prob_range": [float(probb[:, :, x0:x1].min()), float(probb[:, :, x0:x1].max())],
                            "seconds": round(time.time() - t, 1)}
    assert np.isfinite(Pth[:, x0:x1]).all() and probb.shape == (n, S.WIN_H, d.shape[1])
    print("segment_image ok", rep["segment_image"], flush=True)

    # 4. two LoRA train steps (last half of the trunk), grads only where intended
    blocks = lora_blocks_for(f"last{max(1, args.layers // 2)}", args.layers)
    groups = trainable_setup(model, blocks, 8)
    if args.grad_ckpt:
        enable_grad_ckpt(model, blocks)
    model.to(dev).train()
    opt = make_opt(groups)
    frozen_name = "m.vision_encoder.backbone.layers.0.mlp.fc1.weight"
    frozen_before = dict(model.named_parameters())[frozen_name].detach().clone()
    losses, times = [], []
    for _ in range(2):
        t = time.time()
        losses.append((train_step(model, opt, groups, [S.sample_window(it, rng)], dev, args.amp) or (math.nan,))[0])
        sync(dev)
        times.append(round(time.time() - t, 1))
    named = dict(model.named_parameters())
    lora_b = [named[k].detach().abs().max().item() for k in named if k.endswith("lora_b")]
    rep["train"] = {"lora_blocks": list(blocks), "losses": [round(v, 4) for v in losses], "step_s": times,
                    "params": param_report(model), "lora_b_nonzero_after_steps": bool(max(lora_b) > 0),
                    "frozen_unchanged": bool(torch.equal(frozen_before, named[frozen_name].detach())),
                    "grad_ckpt": bool(args.grad_ckpt), **peak_mem_gb(dev)}
    assert all(math.isfinite(v) for v in losses) and rep["train"]["lora_b_nonzero_after_steps"] and rep["train"]["frozen_unchanged"]
    print("train ok", rep["train"], flush=True)

    # 5. checkpoint round trip: trainable state into a fresh model with the same frozen weights -> same output
    model.eval()
    state = {k: v.detach().cpu() for k, v in model.named_parameters() if v.requires_grad}
    fresh = build_random(args.layers).to(dev)
    trainable_setup(fresh, blocks, 8)
    fresh.load_state_dict(state, strict=False)
    fresh.eval()
    with torch.no_grad():
        a = model.decode(*model.encode(x), idx, P, L)[0]
        b = fresh.decode(*fresh.encode(x), idx, P, L)[0]
    rep["ckpt_roundtrip_max_abs_diff"] = float((a - b).abs().max())
    assert rep["ckpt_roundtrip_max_abs_diff"] < 1e-4
    rep["total_s"] = round(time.time() - t0, 1)
    rep.update(peak_mem_gb(dev))
    OUT3.mkdir(parents=True, exist_ok=True)
    (OUT3 / f"smoke_L{args.layers}.json").write_text(json.dumps(rep, indent=1))
    print("ALL SMOKE TESTS PASSED", json.dumps({k: rep[k] for k in ("total_s", "ckpt_roundtrip_max_abs_diff")}), flush=True)


# =============================================================================================== bench (no weights)
def bench(args):
    dev = get_device(args.device)
    model = build_random(args.layers).to(dev)
    blocks = lora_blocks_for(args.lora, n_layers(model))
    groups = trainable_setup(model, blocks, args.lora_r)
    if args.grad_ckpt:
        enable_grad_ckpt(model, blocks)
    model.to(dev)
    opt = make_opt(groups)
    it = first_train_item()
    rng = np.random.default_rng(0)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model.train()
    times = []
    for s in range(args.steps + 1):                          # first step = warm-up, not timed
        batch = [S.sample_window(it, rng) for _ in range(args.windows)]
        t = time.time()
        train_step(model, opt, groups, batch, dev, args.amp)
        sync(dev)
        if s:
            times.append(time.time() - t)
    mem_train = peak_mem_gb(dev)
    model.eval()
    x = S.to_tensor_batch([S.sample_window(it, rng)[0]], dev)
    with torch.no_grad(), amp_ctx(dev, args.amp):
        model.encode(x)
        sync(dev)
        t = time.time()
        for _ in range(3):
            model.encode(x)
        sync(dev)
    enc = (time.time() - t) / 3
    res = {"device": str(dev), "torch": torch.__version__, "layers": args.layers, "lora": args.lora, "lora_blocks": len(blocks),
           "lora_r": args.lora_r, "windows_per_step": args.windows, "prompts_per_window": 8, "amp_bf16": bool(args.amp),
           "tf32": bool(args.tf32), "grad_ckpt": bool(args.grad_ckpt), "params": param_report(model),
           "train_step_s_median": round(float(np.median(times)), 3), "train_step_s_all": [round(v, 3) for v in times],
           "windows_per_hour": round(3600 * args.windows / float(np.median(times))),
           "encoder_fwd_no_grad_s": round(enc, 3), "memory_after_train": mem_train}
    if dev.type == "cuda":
        res["gpu"] = torch.cuda.get_device_name()
    OUT3.mkdir(parents=True, exist_ok=True)
    tag = args.tag or (f"{dev.type}_{args.lora}_w{args.windows}{'_amp' if args.amp else ''}{'_tf32' if args.tf32 else ''}"
                       f"{'_ckpt' if args.grad_ckpt else ''}_L{args.layers}")
    (OUT3 / f"bench_{tag}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1), flush=True)


# =============================================================================================== selftest (weights)
def selftest(args):
    dev = get_device(args.device)
    model, info = load_pretrained(args.hf_id, dev)
    model.eval()
    model.multimask = True
    print("loaded", json.dumps(info)[:600], flush=True)
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
    x = torch.from_numpy(cv2.resize(img, (IMG3, IMG3)).astype(np.float32) / 255.0)[None, None].expand(1, 3, -1, -1)
    x = ((x - 0.5) / 0.5).to(dev)
    names = list(pts)
    p = torch.tensor([[[pts[k][0] * IMG3 / W, pts[k][1] * IMG3 / H]] for k in names], dtype=torch.float32, device=dev)
    res = {}
    with torch.no_grad():
        t = time.time()
        emb, hr = model.encode(x)
        lo, iou = model.decode(emb, hr, torch.zeros(len(names), dtype=torch.long, device=dev), p,
                               torch.ones((len(names), 1), device=dev))
        sync(dev)
        for i, k in enumerate(names):
            ious = []
            for j in range(lo.shape[1]):
                m = F.interpolate(lo[i:i + 1, j:j + 1], (H, W), mode="bilinear", align_corners=False)[0, 0].cpu().numpy() > 0
                ious.append(float((m & truth[k]).sum() / max(1, (m | truth[k]).sum())))
            jb = int(torch.argmax(iou[i]))
            res[k] = {"iou_per_token": [round(v, 3) for v in ious], "picked": jb, "iou_picked": round(ious[jb], 3)}
    print("zero-shot sanity (multimask, best predicted IoU):", json.dumps(res), f"({time.time() - t:.2f}s)", flush=True)
    OUT3.mkdir(parents=True, exist_ok=True)
    (OUT3 / "selftest.json").write_text(json.dumps({"weights": info, "zero_shot": res}, indent=1))
    poor = {k: max(v["iou_per_token"]) for k, v in res.items() if max(v["iou_per_token"]) < 0.8}
    if poor:                                     # non-zero exit: the Modal finetune stops before paying for training
        raise SystemExit(f"selftest: best-token IoU below 0.8 on {poor}; the weights did not load correctly")


# =============================================================================================== train
def train(args):
    dev = get_device(args.device)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    tr_recs, va_recs, val_groups = S.split_records()
    tr = [S.load_item(r, g) for r, g in S.items_of(tr_recs)]
    va = [S.load_item(r, g) for r, g in S.items_of(va_recs)]
    print(f"train {len(tr_recs)} records / {len(tr)} images; val groups {val_groups} ({len(va_recs)} records / {len(va)} images)", flush=True)
    model, info = load_base(args)
    blocks = lora_blocks_for(args.lora, n_layers(model))
    groups = trainable_setup(model, blocks, args.lora_r)
    if args.grad_ckpt:
        enable_grad_ckpt(model, blocks)
    model.to(dev)
    n_tr = sum(p.numel() for g in groups.values() for p in g)
    print(f"trainable params {n_tr / 1e6:.2f}M (LoRA {sum(p.numel() for p in groups['lora']) / 1e6:.2f}M) on {dev}", flush=True)
    opt = make_opt(groups)
    base_lr = [g["lr"] for g in opt.param_groups]
    vrng = np.random.default_rng(123)                       # the same 40 fixed validation windows as ml_sam.train
    vwins = [S.sample_window(va[i % len(va)], vrng) for i in range(0, 4 * len(va), 4)][:40]
    budget = args.minutes * 60.0
    t0, step, hist, best = time.time(), 0, [], (1e9, -1)
    skipped, aborted = [], False
    tag = args.tag or "sam3"
    (OUT3 / "logs").mkdir(parents=True, exist_ok=True)
    log = open(OUT3 / "logs" / f"train_{tag}.log", "a")
    meta = {"family": "sam3", "hf_id": args.hf_id, "revision": info.get("revision"), "lora_blocks": blocks,
            "lora_r": args.lora_r, "random_init": bool(args.random_init), "layers": n_layers(model)}

    def save(path, v):
        torch.save({**meta, "state": S.trainable_state(model), "step": step, "val_loss": v}, path)

    def val_loss():                                         # fp32, as ml_sam.train (bf16 is for the train steps only)
        model.eval()
        tot = []
        with torch.no_grad():
            for win, pts, lab, tg in vwins:
                emb, hr = model.encode(S.to_tensor_batch([win], dev))
                lo, _ = model.decode(emb, hr, torch.zeros(pts.shape[0], dtype=torch.long, device=dev),
                                     torch.from_numpy(pts).to(dev), torch.from_numpy(lab).to(dev))
                tot.append(S.seg_loss(lo[:, 0].float(), torch.from_numpy(tg).to(dev))[0].item())
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
        batch = [S.sample_window(tr[int(rng.integers(len(tr)))], rng) for _ in range(args.windows)]
        r = train_step(model, opt, groups, batch, dev, args.amp)
        step += 1
        if r is None:
            skipped.append(step)
            msg = f"step {step}: non-finite loss/gradient, step skipped ({len(skipped)} so far)"
            if len(skipped) == 1:
                try:                                        # diagnosis only: never let it end a paid run
                    rep = {"step": step, "amp_bf16": bool(args.amp), "tf32": bool(args.tf32),
                           **nonfinite_report(model, groups, batch, dev, args.amp)}
                except Exception as e:
                    rep = {"step": step, "report_error": f"{type(e).__name__}: {e}"}
                (OUT3 / f"nonfinite_{tag}.json").write_text(json.dumps(rep, indent=1))
                msg += f"; replay -> nonfinite_{tag}.json: {json.dumps(rep)}"
            print(msg, flush=True); log.write(msg + "\n"); log.flush()
            recent = sum(s > step - 100 for s in skipped)
            if (len(skipped) >= 20 and len(skipped) > 0.02 * step) or recent >= 10:   # not a rare overflow: stop paying
                aborted = True
                print(f"aborting at step {step}: {len(skipped)} skipped steps ({recent} in the last 100)", flush=True)
                break
        else:
            hist.append(r)
        if step % 50 == 0 and hist:
            h = np.mean(hist[-50:], 0)
            msg = f"step {step} t {el / 60:.1f}m loss {h[0]:.4f} (bce {h[1]:.4f} dice {h[2]:.4f}) lr {opt.param_groups[0]['lr']:.2e} skipped {len(skipped)}"
            print(msg, flush=True); log.write(msg + "\n"); log.flush()
        if el > next_val:
            v = val_loss()
            next_val += 300.0
            msg = f"  val loss {v:.4f} at step {step}"
            print(msg, flush=True); log.write(msg + "\n"); log.flush()
            if v < best[0]:
                best = (v, step)
                save(OUT3 / f"ckpt_{tag}_best.pt", v)
    v = val_loss()
    if v < best[0]:
        best = (v, step)
        save(OUT3 / f"ckpt_{tag}_best.pt", v)
    save(OUT3 / f"ckpt_{tag}_last.pt", v)
    summ = {"family": "sam3", "weights": info, "device": str(dev), "steps": step, "minutes": (time.time() - t0) / 60,
            "val_loss_before": v0, "val_loss_last": v, "best": {"val_loss": best[0], "step": best[1]},
            "train_records": [r.name for r in tr_recs], "val_records": [r.name for r in va_recs], "val_groups": val_groups,
            "trainable_params": n_tr, "lora_blocks": blocks, "lora_r": args.lora_r, "windows_per_step": args.windows,
            "amp_bf16": bool(args.amp), "grad_ckpt": bool(args.grad_ckpt), "window": [S.WIN_H, S.WIN_W], "step_cols": S.STEP,
            "tf32": bool(args.tf32), "skipped_nonfinite_steps": skipped, "aborted": aborted,
            "input": IMG3, "decoder_grid": LOWRES3,
            "loss_curve_every50": [list(np.round(np.mean(hist[i:i + 50], 0), 4)) for i in range(0, len(hist), 50)]}
    (OUT3 / f"train_{tag}.json").write_text(json.dumps(summ, indent=1))
    print(f"done: {step} steps in {summ['minutes']:.1f} min; val loss {v0:.4f} -> {v:.4f} (best {best}); "
          f"skipped {len(skipped)}{' ABORTED' if aborted else ''}", flush=True)
    if aborted or best[1] < 0:                  # non-zero exit: the Modal wrapper must not evaluate this checkpoint
        raise SystemExit(f"train {tag}: {'aborted on non-finite steps' if aborted else 'no finite val loss, no best checkpoint'}")


# =============================================================================================== eval (ml_sam.eval_split with the model swapped)
FROZEN = OUT3 / "frozen_method.json"


def holdout_mark(split):
    return OUT3 / f"{split}_started.json"           # holdout3_started.json, holdout4_ (research/37), holdout5_


def sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def holdout_traces(split="holdout3"):
    """evidence that a test split was already looked at: its start marker or any eval_*<split>* result (any tag)"""
    mark = holdout_mark(split)
    return sorted(p.name for p in OUT3.glob(f"eval_*{split}*.json")) + ([mark.name] if mark.exists() else [])


def check_frozen(args, split="holdout3"):
    """a test split only after `freeze`, only with the frozen checkpoint FILE (sha256), lam and amp, all images, and once
    each. The start marker (write_holdout_mark) follows the model load and precedes the first image, so an interrupted
    run counts too; --force re-runs only the identical frozen method on holdout3 (e.g. after a crash)."""
    if not FROZEN.exists():
        raise SystemExit(f"{split} needs out/ml/sam3/frozen_method.json (run `freeze` after choosing ckpt and lam on val)")
    f = json.loads(FROZEN.read_text())
    if f.get("random_init") or not Path(args.ckpt).exists() or sha256(args.ckpt) != f["ckpt_sha256"] \
            or float(f["lam_final"]) != args.lam or bool(f["amp_bf16"]) != bool(args.amp):
        raise SystemExit(f"{split} must use the frozen method (same checkpoint file, lam, amp): {f}")
    if split != "holdout3" and args.force:             # holdout4 / holdout5 are strictly one-shot
        raise SystemExit(f"--force is not accepted for {split}")
    seen = holdout_traces(split)
    if seen and not args.force:
        raise SystemExit(f"{split} was already started or evaluated {seen}; --force only re-runs the identical frozen method")
    S.check_test_complete(split)                       # every image and every shared B cache present; nothing recomputed
    return {"started_at": None, "argv": sys.argv[1:], "frozen": f, "previous_traces": seen}


def write_holdout_mark(split, payload, overwrite):
    """after the model is loaded, before the first image (an interrupted run counts); exclusive unless --force on holdout3"""
    payload = {**payload, "started_at": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    if overwrite:
        holdout_mark(split).write_text(json.dumps(payload, indent=1))
    else:
        S.write_mark(holdout_mark(split), payload)


def eval_split(args):
    from track_joint import score
    from arbiter import proposals
    split = args.arg                                   # val | holdout3 | holdout4 | holdout5 | zeroshot-val
    base = split.replace("zeroshot-", "")
    if base in S.TEST:
        if split.startswith("zeroshot") or args.limit or args.random_init:
            raise SystemExit(f"{base} is evaluated once, on all images, with the frozen fine-tuned model only")
        mark_payload = check_frozen(args, base)
    dev = get_device(args.device)
    if split.startswith("zeroshot"):
        model, info = load_base(args)
        model = model.to(dev).eval()
        model.multimask = True
        token, desc = "best", {"model": f"SAM 3 tracker ({args.hf_id}) zero-shot, best predicted-IoU multimask token", **info}
    else:
        model, ck = load_finetuned(args.ckpt, dev)
        token, desc = "single", {"model": "SAM 3 tracker fine-tuned" + (" (RANDOM INIT dry run)" if ck.get("random_init") else ""),
                                 "ckpt": rel_root(args.ckpt),
                                 "step": ck["step"], "val_loss": ck["val_loss"], "lora_blocks": len(ck["lora_blocks"])}
    if base in S.TEST:
        write_holdout_mark(base, mark_payload, overwrite=bool(args.force))
    tag = split + (f"_{args.tag}" if args.tag else "")
    figs = OUT3 / "figs" / tag
    figs.mkdir(parents=True, exist_ok=True)
    pdir = OUT3 / "probs" / tag
    if base == "val":
        pdir.mkdir(parents=True, exist_ok=True)
    (OUT3 / "paths" / tag).mkdir(parents=True, exist_ok=True)
    rows, t_all, b_src = [], time.time(), set()
    items = S.split_items(base)
    if args.limit:
        items = items[:args.limit]
        desc["limit_images"] = args.limit
    for r, g in items:
        t = time.time()
        d = r / "film"
        meta = json.loads((d / "meta.json").read_text())
        img = cv2.imread(str(d / f"heli_g{g:g}.png"), 0)
        pen = np.load(d / f"gt_g{g:g}.npz")["pen"]
        S_, x0 = meta["spacing"], meta["margin_x"]
        x1, n, act = img.shape[1] - x0, pen.shape[0], meta["active_line"]
        y0 = pen[:, x0]
        with amp_ctx(dev, args.amp):
            probb, _ = S.segment_image(model, img, y0, x0, x1, token=token, dev=dev)
        P = S.decode_paths(probb, y0, x0, x1, args.lam, dev)
        bp = b_cache_path(r.name, g, base)
        if not bp.exists():
            A, B, ink = proposals(img, y0, x0, x1, S_)
            bp.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(bp, A=A, B=B, ink=ink)
        b_src.add(str(bp.parent.relative_to(ROOT)))
        z = np.load(bp)
        row = {"record": r.name, "group": r.name[:8], "gain": float(g), "active_line": act}
        for name, est in (("SAM", P), ("B", z["B"]), ("A", z["A"])):
            row[name] = score(est, pen, range(n), x0, x1, S_)[0]
            row[name + "_quake"] = score(est, pen, [act], x0, x1, S_)[0]
        row["SAM_quake_within_3px"] = round(float(np.nanmean(np.abs(P[act, x0:x1] - pen[act, x0:x1]) <= 3) * 100), 1)
        rows.append(row)
        if base == "val":
            np.savez_compressed(pdir / f"{r.name}_g{g:g}.npz", p=np.round(probb[:, :, x0:x1] * 255).astype(np.uint8), x0=x0)
        np.savez_compressed(OUT3 / "paths" / tag / f"{r.name}_g{g:g}.npz", SAM=P.astype(np.float32))
        if len(rows) <= 3 or r.name == items[0][0].name or g in (4.0, 8.0) and len([q for q in rows if q["gain"] == g]) == 1:
            S.overlay(img, pen[act], P[act], z["B"][act], x0, x1, figs / f"{r.name}_g{g:g}_quake.png")
        print(f"{r.name} g{g:g}: SAM3 {row['SAM']:.1f}/{row['SAM_quake']:.1f}  B {row['B']:.1f}/{row['B_quake']:.1f}  "
              f"A {row['A']:.1f}/{row['A_quake']:.1f}  ({time.time() - t:.0f}s)", flush=True)
    keys = ["SAM", "B", "A", "SAM_quake", "B_quake", "A_quake", "SAM_quake_within_3px"]
    summ = S.summarize(rows, keys)
    res = {"split": split, "method": desc, "decode": {"lam": args.lam, "lam_chain": 3e-4, "window": [S.WIN_H, S.WIN_W],
           "step": S.STEP, "prompt_cols": S.PROMPT_COLS, "input": IMG3, "decoder_grid": LOWRES3, "amp_bf16": bool(args.amp),
           "start": "GT pen at x0 (layout start), same as A/B"}, "b_source": sorted(b_src),
           "metric": "track_joint.score: % of drawn columns missing or |path - pen| > S/2", "note": "column 'SAM' = SAM 3",
           "summary": summ, "rows": rows, "runtime_min": round((time.time() - t_all) / 60, 1)}
    (OUT3 / f"eval_{tag}.json").write_text(json.dumps(res, indent=1))
    print("gain   | whole image fail %: SAM3 / B / A | quake line fail %: SAM3 / B / A | SAM3 quake within 3px %")
    for g, v in summ.items():
        print(f"{g:6s} | {v['SAM']:5.1f} / {v['B']:5.1f} / {v['A']:5.1f} | {v['SAM_quake']:5.1f} / {v['B_quake']:5.1f} / "
              f"{v['A_quake']:5.1f} | {v['SAM_quake_within_3px']:5.1f}  (n={v['n_images']})")


def freeze(args):
    """record the method chosen on validation. Never after holdout3 was touched; re-freezing before that needs --force"""
    seen = [t for sp in S.TEST for t in holdout_traces(sp)]     # no re-freeze once any test split was touched
    if seen:
        raise SystemExit(f"a test split was already started or evaluated {seen}: the method can no longer be changed")
    if FROZEN.exists() and not args.force:
        raise SystemExit("out/ml/sam3/frozen_method.json exists (use --force to replace it before any holdout3 run)")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    if ck.get("random_init"):
        raise SystemExit("a --random-init dry-run checkpoint cannot be frozen")
    out = {"frozen_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "ckpt": rel_root(args.ckpt),
           "ckpt_sha256": sha256(args.ckpt), "hf_id": ck["hf_id"], "revision": ck.get("revision"), "random_init": False,
           "ckpt_step": ck["step"], "ckpt_val_loss": ck["val_loss"], "lam_final": args.lam, "lam_chain": 3e-4,
           "amp_bf16": bool(args.amp), "window": [S.WIN_H, S.WIN_W], "step": S.STEP, "prompt_cols": list(S.PROMPT_COLS),
           "lam_choice": args.note or "chosen on the validation groups only (tune-lam)"}
    FROZEN.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def eval_real(args):
    name = args.arg.upper()
    rp = ROOT / "out" / "real_paired"
    img = cv2.imread(str(rp / name / f"{name}_rescaled.png"), 0)       # exactly real_paired_eval's img_s
    z = np.load(rp / name / f"{name}_paths.npz")
    bases, y0 = z["bases"], z["y0"]
    res = json.loads((rp / f"{name}_results.json").read_text())
    x0, x1 = res["layout"]["x0"], res["layout"]["x1"]
    H, W = img.shape
    tops = np.clip(np.round(bases).astype(int) - S.WIN_H // 2, 0, H - S.WIN_H)    # each line near the band centre
    if args.paths:                                                      # made by real_paths on another machine
        zp = np.load(args.paths)
        assert np.array_equal(zp["tops"], tops), "paths come from a different layout"
        assert str(zp["ckpt_sha256"]) == sha256(args.ckpt), "paths come from a different checkpoint"
        assert float(zp["lam"]) == args.lam, "paths come from a different lam"
        P = zp["SAM"].astype(float)
    else:
        dev = get_device(args.device)
        model, ck = load_finetuned(args.ckpt, dev)
        t = time.time()
        with amp_ctx(dev, args.amp):
            probb, _ = S.segment_image(model, img, y0, x0, x1, tops=tops, dev=dev)
        P = S.decode_paths(probb, y0, x0, x1, args.lam, dev, tops=tops)
        print(f"[{name}] segmented {len(bases)} lines in {time.time() - t:.0f}s", flush=True)
    summ, check, per, res = S.real_metrics(name, {"SAM": P})
    out = {"scan": name, "model": "SAM 3 tracker fine-tuned (column 'SAM')",
           "ckpt": rel_root(args.ckpt), "lam": args.lam, "amp_bf16": bool(args.amp),
           "layout": "out/real_paired/<scan>_rescaled.png + _paths.npz (bases, y0), alignment chosen_dp_fit, global amplitude",
           "band_tops": tops.tolist(), "B_metric_reproduced": check, "summary": summ,
           "per_line_SAM": {str(k): v for k, v in per["SAM"].items()}}
    (OUT3 / f"real_{name}.json").write_text(json.dumps(out, indent=1))
    np.savez_compressed(OUT3 / f"real_{name}_paths.npz", SAM=P, tops=tops)
    figs = OUT3 / "figs" / "real"                                       # KIP figures stay local (out/ is not published)
    figs.mkdir(parents=True, exist_ok=True)
    km = res["main_shock_line"]
    S.overlay(img, np.full(W, np.nan), P[km], z["B"][km], x0, x1, figs / f"{name}_quake_line{km}.png",
              c=int(np.nanargmax(np.abs(P[km, x0:x1] - bases[km]))) + x0)
    print(f"[{name}] B metric reproduced from stored paths: {check}")
    for cat, v in summ.items():
        print(f"[{name}] {cat:12s} {v['lines']} | " + " | ".join(
            f"{m}: r {v[m]['r_band']:+.2f} in {v[m]['in_0.1S_global']:.2f} lost {v[m]['lost_0.5S_global']:.2f}"
            for m in ("SAM", "B_stored", "judge_stored", "DP38_stored")), flush=True)


# =============================================================================================== CLI
def real_paths(args):
    """same tops, windows and decoding as eval_real, without the reference (the digital pairs stay on our own machines);
    --lines: only these lines (the device gate, research/49 §4). Output next to the input, ckpt hash and lam included."""
    name = args.arg.upper()
    d = ROOT / "out" / "ml" / "real_inputs"
    z = np.load(d / f"{name}.npz")
    img, bases, y0, x0, x1 = z["img"], z["bases"], z["y0"], int(z["x0"]), int(z["x1"])
    H, W = img.shape
    tops = np.clip(np.round(bases).astype(int) - S.WIN_H // 2, 0, H - S.WIN_H)
    ks = np.array([int(k) for k in args.lines.split(",")] if args.lines else range(len(bases)))
    dev = get_device(args.device)
    model, _ = load_finetuned(args.ckpt, dev)
    t = time.time()
    with amp_ctx(dev, args.amp):
        probb, _ = S.segment_image(model, img, y0[ks], x0, x1, tops=tops[ks], dev=dev)
    P = S.decode_paths(probb, y0[ks], x0, x1, args.lam, dev, tops=tops[ks])
    out = d / f"{name}_paths{'_lines' if args.lines else ''}.npz"
    np.savez_compressed(out, SAM=P, tops=tops, lines=ks, ckpt_sha256=sha256(args.ckpt), lam=args.lam, device=str(dev),
                        gpu=torch.cuda.get_device_name(0) if dev.type == "cuda" else "", seconds=time.time() - t)
    print(f"[{name}] {len(ks)} lines in {time.time() - t:.0f}s on {dev} -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("arg", nargs="?")
    ap.add_argument("--hf-id", dest="hf_id", default=HF_ID)
    ap.add_argument("--device", default="auto", help="auto (mps > cuda > cpu, as ml_sam) | cpu | mps | cuda")
    ap.add_argument("--layers", type=int, default=32, help="smoke/bench only: trunk depth of the random-init model")
    ap.add_argument("--minutes", type=float, default=60.0)
    ap.add_argument("--ckpt", default=str(OUT3 / "ckpt_sam3_best.pt"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--lam", type=float, default=3e-4)
    ap.add_argument("--lams", default="0,3e-6,1e-5,3e-5,1e-4,3e-4,1e-3,3e-3")
    ap.add_argument("--windows", type=int, default=1, help="training windows per step (x 8 line prompts each)")
    ap.add_argument("--lora", default="last16", help="last16 | last<K> | all | none")
    ap.add_argument("--lora-r", dest="lora_r", type=int, default=8)
    ap.add_argument("--amp", action="store_true",
                    help="bf16 autocast (CUDA; accepted on MPS/CPU, but on this Mac's CPU it was ~100x slower than fp32)")
    ap.add_argument("--tf32", action="store_true",
                    help="CUDA: TF32 matmuls/convs for fp32 train steps (Modal default: bf16 gave non-finite trunk "
                         "gradients in the dry run while the same batch in fp32 was finite, out/logs/sam3_modal_dryrun2.log)")
    ap.add_argument("--grad-ckpt", dest="grad_ckpt", action="store_true", help="activation checkpointing on LoRA blocks")
    ap.add_argument("--steps", type=int, default=5, help="bench: timed steps")
    ap.add_argument("--note", default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--random-init", dest="random_init", action="store_true",
                    help="train/eval dry run of the whole pipeline without the gated weights (never a result)")
    ap.add_argument("--limit", type=int, default=0, help="eval: only the first N images (dry runs; refused for the test splits)")
    ap.add_argument("--paths", default="", help="real: score this real-paths output instead of tracking here")
    ap.add_argument("--lines", default="", help="real-paths: comma-separated lines (default all)")
    args = ap.parse_args()
    if args.device != "auto":                  # ml_sam's own device choice (tune-lam / analyze / Viterbi) follows --device
        S.get_device = lambda: torch.device(args.device)
    if args.tf32:
        if args.cmd not in ("train", "bench", "smoke"):   # evaluation stays fp32 matmuls (the frozen method has no tf32 field)
            raise SystemExit("--tf32 is for train/bench only")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    if args.random_init and not args.tag:
        args.tag = "dryrun"
    if args.limit and args.cmd in ("tune-lam", "analyze"):   # dry runs: only the images a --limit eval produced
        full = S.split_items
        S.split_items = lambda split: full(split)[:args.limit]
    OUT3.mkdir(parents=True, exist_ok=True)
    (OUT3 / "logs").mkdir(parents=True, exist_ok=True)
    {"smoke": smoke, "bench": bench, "selftest": selftest, "train": train, "eval": eval_split, "real": eval_real, "real-paths": real_paths,
     "freeze": freeze, "tune-lam": S.tune_lam, "analyze": S.analyze}[args.cmd](args)


if __name__ == "__main__":
    main()

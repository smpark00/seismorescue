"""Modal app: scripts/ml_sam3.py (SAM 3 tracker fine-tune, research/32) on one H100 or A100-80GB.
Modal runs so far: bench (bf16) passed and the bf16 pipeline dry run hit non-finite gradients (10-28); see --prec below.

One-time setup (after `pip install modal` and `modal setup`, and after facebook/sam3 access is approved on Hugging Face):
    modal secret create huggingface HF_TOKEN=<read token of the approved HF account>
    modal volume create seismo-sam                                    # same volume layout as ml_sam_modal.py
    modal volume put seismo-sam out/film_train /film_train            # ~0.35 GB synthetic renders + GT
    modal volume put seismo-sam out/ml/sam/cache_b /ml/sam/cache_b    # B proposals of the 60 val images (read only)
    modal volume put seismo-sam out/film_holdout3 /film_holdout3      # only for the one final holdout run
    modal volume put seismo-sam out/cache_film_holdout3 /cache_film_holdout3
    modal volume put seismo-sam out/film_holdout5 /film_holdout5      # research/34 fresh test split (same renderer)
    modal volume put seismo-sam out/cache_film_holdout5 /cache_film_holdout5
Runs (GPU type from the SEISMO_GPU env var: H100 (default) | A100-80GB | L40S; --prec tf32 (default) | bf16 for the
train steps: the bf16 dry run on 10-28 gave non-finite trunk gradients that fp32 did not, out/logs/sam3_modal_dryrun2.log):
    modal run scripts/ml_sam3_modal.py --mode bench                   # ~5 min: step time + peak memory, random weights
    modal run scripts/ml_sam3_modal.py --mode pipeline-dryrun         # ~10 min: train/eval/tune-lam with random weights
    modal run scripts/ml_sam3_modal.py --mode train --minutes 60 --windows 4   # fine-tune + val eval + zero-shot + lam grid
    mkdir -p out/ml/sam3_modal && (cd out/ml/sam3_modal && modal volume get seismo-sam ml/sam3 . && mv sam3/* . && rmdir sam3)
        # read tune_lam_val_<tag>.json, pick lam on val (`get ... /ml/sam3 out/ml/sam3_modal` fails: 'Is a directory')
    modal run scripts/ml_sam3_modal.py --mode holdout --lam <chosen>  # freeze, then holdout3 exactly once
    modal run scripts/ml_sam3_modal.py --mode holdout --lam <chosen> --split holdout5   # same freeze, holdout5 once
    modal run scripts/ml_sam3_modal.py --mode real-paths --name MAT --lam 1e-5 --lines 3,8,13,14,15,22   # research/49 §4 gate
    modal run scripts/ml_sam3_modal.py --mode real-paths --name ALQ --lam 1e-5   # public chip paths; scored locally
The real KIP scan is NOT uploaded (KIP is local-only by licence), and neither are the digital pairs: run
`ml_sam3.py real KIP|MAT --ckpt ...` locally with the downloaded checkpoint. real-paths reads only out/ml/real_inputs/<name>.npz,
made from a public CC0 chip. Weights are fetched from Hugging Face inside the container with the secret and cached on
the volume (HF_HOME), so the 3.44 GB download happens once. No Meta API is called.
Budget (Modal list prices read 2026-10-27, research/26 4.1; check again before running): H100 $3.95/h, A100-80GB
$2.50/h, plus CPU/memory. Plan ~2-2.5 GPU-hours in total -> roughly $8-10 on H100, $6-8 on A100-80GB (research/32).
"""
import os
from pathlib import Path

import modal

LOCAL = Path(__file__).resolve().parent
REMOTE = "/root/seismo"                     # ml_sam3.py resolves ROOT = <script dir>/.., so out/ = the volume
GPU = os.environ.get("SEISMO_GPU", "H100")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("torch==2.14.1", "torchvision==0.29.1", "numpy==2.5.4", "scipy==1.18.1",
                 "opencv-python-headless==5.0.0.93", "transformers==5.21.0", "huggingface_hub", "safetensors")
    .env({"HF_HOME": f"{REMOTE}/out/hf_cache", "PYTHONUNBUFFERED": "1"})
    .add_local_file(LOCAL / "ml_sam3.py", f"{REMOTE}/scripts/ml_sam3.py")
    .add_local_file(LOCAL / "ml_sam.py", f"{REMOTE}/scripts/ml_sam.py")
    .add_local_file(LOCAL / "track_joint.py", f"{REMOTE}/scripts/track_joint.py")
    .add_local_file(LOCAL / "arbiter.py", f"{REMOTE}/scripts/arbiter.py")
)
vol = modal.Volume.from_name("seismo-sam", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")
app = modal.App("seismo-sam3-finetune", image=image)


PREC = {"tf32": ["--tf32"], "bf16": ["--amp"]}   # train-step precision; eval is always fp32 (ml_sam3 eval/real)
DATA = {"film": {},                              # training renders -> ml_sam / ml_sam3 environment (research/35)
        "film2": {"SEISMO_FILM_TRAIN": "film2_train", "SEISMO_SAM3_OUT": "sam3_film2",
                  "SEISMO_VAL_GROUPS": "20090103,20110311,20120831,20230510"}}   # = the film2 validation dates (research/31)


def _out(data):
    return DATA[data].get("SEISMO_SAM3_OUT", "sam3")


def _run(*argv, data="film"):
    import subprocess
    import sys
    cmd = [sys.executable, "-I", f"{REMOTE}/scripts/ml_sam3.py", *map(str, argv), "--device", "cuda"]
    print("+", " ".join(cmd), DATA[data] or "", flush=True)
    subprocess.run(cmd, check=True, env={**os.environ, **DATA[data]})


@app.function(gpu=GPU, volumes={f"{REMOTE}/out": vol}, timeout=3600)
def bench(windows: int = 4, prec: str = "tf32"):
    for extra in ([], ["--grad-ckpt"]):                  # same settings as the planned run, with / without checkpointing
        _run("bench", "--steps", 10, "--windows", windows, "--lora", "last16", *PREC[prec], *extra)
    _run("bench", "--steps", 5, "--windows", windows, "--lora", "all", *PREC[prec], "--grad-ckpt")
    vol.commit()


@app.function(gpu=GPU, volumes={f"{REMOTE}/out": vol}, timeout=3600)
def pipeline_dryrun(prec: str = "tf32"):
    """random weights, 4-layer trunk: exercises train -> eval -> tune-lam -> analyze on the volume layout"""
    _run("train", "--random-init", "--layers", 4, "--minutes", 2, "--windows", 2, *PREC[prec])
    _run("eval", "val", "--random-init", "--ckpt", f"{REMOTE}/out/ml/sam3/ckpt_dryrun_best.pt", "--limit", 3, "--lam", 1e-5)
    _run("tune-lam", "val_dryrun", "--limit", 3, "--lams", "1e-5,3e-4")
    _run("analyze", "val_dryrun", "--limit", 3)
    vol.commit()


@app.function(gpu=GPU, volumes={f"{REMOTE}/out": vol}, secrets=[hf_secret], timeout=4 * 3600)
def finetune(minutes: float = 60, windows: int = 4, lora: str = "last16", lora_r: int = 8, grad_ckpt: bool = False,
             tag: str = "sam3_modal", prec: str = "tf32", data: str = "film"):
    _run("selftest", data=data)                          # downloads/caches the gated weights, zero-shot sanity masks
    vol.commit()
    try:                                                 # an aborted train exits non-zero: validate never runs on it
        _run("train", "--minutes", minutes, "--windows", windows, "--lora", lora, "--lora-r", lora_r, *PREC[prec],
             "--tag", tag, *(["--grad-ckpt"] if grad_ckpt else []), data=data)
    finally:
        vol.commit()                                     # logs, summary and nonfinite_<tag>.json reach the volume
    return tag


@app.function(gpu=GPU, volumes={f"{REMOTE}/out": vol}, secrets=[hf_secret], timeout=3 * 3600)
def validate(tag: str, lam: float = 3e-4, data: str = "film"):
    ckpt = f"{REMOTE}/out/ml/{_out(data)}/ckpt_{tag}_best.pt"
    _run("eval", "val", "--ckpt", ckpt, "--lam", lam, "--tag", tag, data=data)   # evaluation in fp32, as ml_sam
    _run("tune-lam", f"val_{tag}", data=data)            # final-Viterbi smoothness on validation only
    _run("eval", "zeroshot-val", "--lam", lam, data=data)   # SAM 3 without fine-tuning, same decoding
    vol.commit()


@app.function(gpu=GPU, volumes={f"{REMOTE}/out": vol}, secrets=[hf_secret], timeout=2 * 3600)
def holdout(tag: str, lam: float, split: str = "holdout3", data: str = "film", freeze: bool = True):
    ckpt = f"{REMOTE}/out/ml/{_out(data)}/ckpt_{tag}_best.pt"
    if freeze:                                           # the first test run freezes; later test splits reuse it
        _run("freeze", "--ckpt", ckpt, "--lam", lam, "--note", f"lam chosen on validation (tune_lam_val_{tag}.json)",
             data=data)
    try:
        _run("eval", split, "--ckpt", ckpt, "--lam", lam, data=data)   # fp32; refuses unless frozen / if touched
        _run("analyze", split, data=data)
    finally:
        vol.commit()                                     # the start marker reaches the volume even after a crash


@app.function(gpu=GPU, volumes={f"{REMOTE}/out": vol}, secrets=[hf_secret], timeout=1800)
def real_paths(name: str, lam: float, lines: str = ""):
    assert name in ("MAT", "ALQ"), "public chips only"
    ckpt = f"{REMOTE}/out/ml/sam3_film2/ckpt_sam3_film2_best.pt"
    try:
        _run("real-paths", name, "--ckpt", ckpt, "--lam", lam, *(["--lines", lines] if lines else []), data="film2")
    finally:
        vol.commit()


@app.local_entrypoint()
def main(mode: str = "bench", minutes: float = 60, windows: int = 4, lora: str = "last16", lora_r: int = 8,
         grad_ckpt: bool = False, lam: float = -1.0, tag: str = "", prec: str = "tf32", split: str = "holdout3",
         data: str = "film", freeze: str = "auto", name: str = "", lines: str = ""):
    if prec not in PREC or data not in DATA:
        raise SystemExit(f"--prec must be one of {sorted(PREC)}, --data one of {sorted(DATA)}")
    tag = tag or {"film": "sam3_modal", "film2": "sam3_film2"}[data]
    if mode == "bench":
        bench.remote(windows, prec)
    elif mode == "pipeline-dryrun":
        pipeline_dryrun.remote(prec)
    elif mode == "train":
        tag = finetune.remote(minutes, windows, lora, lora_r, grad_ckpt, tag, prec, data)
        validate.remote(tag, 3e-4, data)
    elif mode == "holdout":                              # once, after the checkpoint and lam are chosen on validation
        if lam < 0:                                      # 0 is a valid grid value
            raise SystemExit("pass --lam chosen from tune_lam_val_<tag>.json")
        if split not in ("holdout3", "holdout5"):
            raise SystemExit("--split must be holdout3 or holdout5")
        if freeze == "auto" and data != "film":          # film2: the first test run must say --freeze yes, later ones no
            raise SystemExit("--data film2 holdout needs --freeze yes (first test split) or --freeze no")
        holdout.remote(tag, lam, split, data, split == "holdout3" if freeze == "auto" else freeze == "yes")
    elif mode == "real-paths":
        real_paths.remote(name, lam, lines)
        return
    else:
        raise SystemExit(f"unknown mode {mode}")
    o = _out(data)
    print(f"done: mkdir -p out/ml/{o}_modal && (cd out/ml/{o}_modal && modal volume get seismo-sam ml/{o} . && mv {o}/* . && rmdir {o})")

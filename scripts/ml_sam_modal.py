"""Modal app: the same SAM 2.1 trace-segmenter fine-tune as scripts/ml_sam.py, on one A100/H100 with a larger model and
more steps (docs/RESULTS.md). NOT RUN YET: written before a Modal account exists; Modal is not installed locally.

Setup (once, after `pip install modal` and `modal setup`):
    modal volume create seismo-sam
    modal volume put seismo-sam out/film_train /film_train            # ~0.4 GB of synthetic renders + GT
    modal volume put seismo-sam out/ml/sam/cache_b /ml/sam/cache_b    # B proposals for the val images (optional)
    modal volume put seismo-sam out/film_holdout3 /film_holdout3      # only needed for the one final --holdout run
    modal volume put seismo-sam out/cache_film_holdout3 /cache_film_holdout3
Run (fine-tune + validation; the holdout is not touched unless --holdout is passed, and then only once):
    modal run scripts/ml_sam_modal.py --variant base_plus --minutes 120 --windows 4
    modal volume get seismo-sam /ml/sam out/ml/sam_modal              # checkpoints, logs, eval json
Budget: H100 is roughly $4/h on Modal (2025 list price; check before running), so 2 h training + ~20 min evaluation
is ~$10 of the $100 credit. Only model weights are downloaded inside the container (dl.fbaipublicfiles.com,
Apache-2.0); no data leaves the volume and no Meta API is called.
"""
from pathlib import Path

import modal

LOCAL = Path(__file__).resolve().parent
REMOTE = "/root/seismo"                     # ml_sam.py resolves ROOT = <script dir>/.., so out/ = the volume

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("torch==2.5.1", "numpy==2.1.3", "scipy==1.14.1", "opencv-python-headless==4.10.0.84")
    .add_local_file(LOCAL / "ml_sam.py", f"{REMOTE}/scripts/ml_sam.py")
    .add_local_file(LOCAL / "track_joint.py", f"{REMOTE}/scripts/track_joint.py")
    .add_local_file(LOCAL / "arbiter.py", f"{REMOTE}/scripts/arbiter.py")
)
vol = modal.Volume.from_name("seismo-sam", create_if_missing=True)
app = modal.App("seismo-sam-finetune", image=image)

WEIGHT_FILES = {"tiny": "sam2.1_hiera_tiny.pt", "small": "sam2.1_hiera_small.pt",
                "base_plus": "sam2.1_hiera_base_plus.pt", "large": "sam2.1_hiera_large.pt"}


def _weights(variant):
    import urllib.request
    d = Path(REMOTE) / "out" / "ml" / "sam" / "weights" / f"sam2.1_hiera_{variant}"
    f = d / WEIGHT_FILES[variant]
    if not f.exists():
        d.mkdir(parents=True, exist_ok=True)
        url = f"https://dl.fbaipublicfiles.com/segment_anything_2/092824/{WEIGHT_FILES[variant]}"
        urllib.request.urlretrieve(url, f)
        vol.commit()
    return f


def _run(*argv):
    import subprocess
    import sys
    cmd = [sys.executable, "-I", f"{REMOTE}/scripts/ml_sam.py", *map(str, argv)]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


@app.function(gpu="H100", volumes={f"{REMOTE}/out": vol}, timeout=4 * 3600)
def finetune(variant: str = "base_plus", minutes: float = 120, windows: int = 4, lora: str = "all", lora_r: int = 16):
    _weights(variant)
    tag = f"{variant}_modal"
    _run("train", "--variant", variant, "--minutes", minutes, "--windows", windows, "--lora", lora,
         "--lora-r", lora_r, "--amp", "--tag", tag)
    vol.commit()
    return tag


@app.function(gpu="H100", volumes={f"{REMOTE}/out": vol}, timeout=2 * 3600)
def evaluate(tag: str, split: str = "val", lam: float = 3e-4):
    ckpt = f"{REMOTE}/out/ml/sam/ckpt_{tag}_best.pt"
    _run("eval", split, "--ckpt", ckpt, "--lam", lam, "--tag", tag)
    if split == "val":                       # choose the final-DP smoothness on validation only
        _run("tune-lam", f"val_{tag}")
    vol.commit()


@app.local_entrypoint()
def main(variant: str = "base_plus", minutes: float = 120, windows: int = 4, lora: str = "all", lora_r: int = 16,
         holdout: bool = False, lam: float = 3e-4):
    tag = finetune.remote(variant, minutes, windows, lora, lora_r)
    evaluate.remote(tag, "val", lam)
    if holdout:                              # once, after the method (and lam) are frozen
        evaluate.remote(tag, "holdout3", lam)
    print(f"done: modal volume get seismo-sam /ml/sam out/ml/sam_modal   (tag {tag})")

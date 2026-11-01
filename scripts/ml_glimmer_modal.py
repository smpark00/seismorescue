"""Modal app: Muse Glimmer QLoRA pilot, stage 1 (research/38). Question: can Glimmer be QLoRA-trained on our
line-identity task at all, and at what cost? 4-bit load -> zero-shot forced choice with and without the ink -> a few
QLoRA steps -> adapter save -> both scores again -> adapter reload. The no-ink control (glimmer_task.compose ink=False)
keeps the red paths on a flat background: how much do the path shapes alone answer? (path-only rules score high on
this data, see manifest.json "baselines"). The task and data files are defined in glimmer_task.py.

Setup (Modal CLI 1.6.3; a Modal secret named `huggingface` holds the Hugging Face token):
    modal run scripts/ml_glimmer_modal.py --mode download        # CPU: 59.6 GB bf16 weights -> volume 'glimmer'
    .venv/bin/python -I scripts/glimmer_verifier_data.py         # local: synthetic film2 crops -> out/glimmer_verifier
    modal volume put glimmer out/glimmer_verifier /data/glimmer_verifier
    modal run scripts/ml_glimmer_modal.py --mode check           # CPU: imports, NF4 layout, data, tokens per example
Stage 1 (GPU from SEISMO_GPU, default H100, with 4 cores and 16 GiB requested; the 36 min function timeout caps it at
about $2.56 = 36 min x (H100 $3.95/h + 4 x $0.0473/core-h + 16 x $0.008/GiB-h) while usage stays within the requests.
The $3 plan cap also holds if 60 GiB of mmapped weights and 2 min of startup are billed: 38 min x $4.62/h = $2.93):
    modal run --detach scripts/ml_glimmer_modal.py --mode stage1   # stops if out/stage1/report.json exists (--force)
    modal volume get glimmer /out/stage1 out/ml/                 # -> out/ml/stage1: report.json, scores, adapter
Data rule: only synthetic renders go to Modal; no Meta API call. Delete the volume when the pilot ends
(`modal volume delete glimmer`; storage is $0.09/GiB-month, about $5/month for the weights).
"""
import os
import time
from pathlib import Path

import modal

LOCAL = Path(__file__).resolve().parent
REMOTE = "/root/seismo"
GPU = os.environ.get("SEISMO_GPU", "H100")            # rep["gpu"] records the card ("H100!" is not in the modal 1.6.3 source)
HF_ID = "meta-models/Muse-Glimmer-30B"
REV = "cec94fcfd6512f2cfe708138c32533b346ea4f69"      # main on 2026-10-30 (lastModified 2026-10-22), pinned
WEIGHTS = f"/vol/hf/Muse-Glimmer-30B@{REV[:8]}"
DATA = "/vol/data/glimmer_verifier"
# transformers matches a skip name only as a module-name prefix or suffix (should_convert_module): a bare "vision_tower"
# matches nothing. NF4 = Linear4bit modules per part after the 4-bit load: 52 language layers x 8, none elsewhere.
SKIP_4BIT = ["model.vision_tower", "model.vision_adapter", "model.vision_projection", "lm_head"]   # stay bf16
NF4 = {"language_model": 416, "vision_tower": 0, "vision_adapter": 0, "vision_projection": 0, "lm_head": 0, "other": 0}
LORA_TARGETS = r".*language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
CPU, MEM = 4.0, 16384                                 # stage1 cores and MiB (requests)
H100_PER_H, CORE_PER_H, GIB_PER_H = 3.95, 0.0473, 0.008   # `modal billing rates`, 2026-10-30
USD_PER_H = H100_PER_H + CPU * CORE_PER_H + MEM / 1024 * GIB_PER_H   # stage1 all-in: $4.27/h, $2.56 per 36 min

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("torch==2.14.1", "torchvision==0.29.1", "numpy==2.5.4", "opencv-python-headless==5.0.0.93",
                 "transformers==5.21.0", "peft==0.21.2", "bitsandbytes==0.50.2", "accelerate==1.15.0",
                 "huggingface_hub==2.3.0", "safetensors==0.8.0")
    .env({"PYTHONUNBUFFERED": "1", "HF_HUB_DISABLE_TELEMETRY": "1"})
    .add_local_file(LOCAL / "glimmer_task.py", f"{REMOTE}/scripts/glimmer_task.py")
)
vol = modal.Volume.from_name("glimmer", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")
app = modal.App("seismo-glimmer-qlora", image=image)


@app.function(volumes={"/vol": vol}, secrets=[hf_secret], cpu=4.0, memory=8192, timeout=3600)
def download():
    from huggingface_hub import HfApi, snapshot_download
    info = HfApi().model_info(HF_ID, revision=REV, files_metadata=True)
    want = {s.rfilename: s.size for s in info.siblings}
    t = time.time()
    snapshot_download(HF_ID, revision=REV, local_dir=WEIGHTS, max_workers=8)
    got = {f: (Path(WEIGHTS) / f).stat().st_size for f in want if (Path(WEIGHTS) / f).exists()}
    vol.commit()
    bad = {f: [want[f], got.get(f)] for f in want if got.get(f) != want[f]}
    return {"seconds": round(time.time() - t), "files": len(got), "GB": round(sum(got.values()) / 1e9, 2), "mismatch": bad}


def _env():
    import sys
    sys.path.insert(0, f"{REMOTE}/scripts")
    import glimmer_task
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(WEIGHTS)
    tok = proc.tokenizer
    ids = {a: tok(a, add_special_tokens=False)["input_ids"] for a in glimmer_task.ANSWERS}
    assert all(len(v) == 1 for v in ids.values()), ids
    return glimmer_task, proc, tok, {a: v[0] for a, v in ids.items()}


def _encode(gt, proc, ex, answer=None, ink=True):
    """prompt ends right before the answer letter (no reasoning turn); with an answer, labels supervise only the
    assistant tail ' to=user<|message|>A<|eot|>'. The template already starts with the BOS token. ink=False encodes
    the no-ink composite (same size, so the same tokens)."""
    import torch
    img = gt.compose(ex["crop"], ex["cands"][0], ex["cands"][1], ink)
    prefix = proc.apply_chat_template(gt.messages(), add_generation_prompt=True, tokenize=False)
    if answer is None:
        enc = proc(text=[prefix + " to=user<|message|>"], images=[img], return_tensors="pt", add_special_tokens=False)
        labels = None
    else:
        full = proc.apply_chat_template(gt.messages(answer), tokenize=False)
        assert full.startswith(prefix), "chat template: prompt is not a prefix of the training text"
        enc = proc(text=[full], images=[img], return_tensors="pt", add_special_tokens=False)
        p_ids = proc(text=[prefix], images=[img], return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        n = len(p_ids)
        assert enc["input_ids"][0, :n].tolist() == p_ids.tolist(), "prompt tokens differ inside the training text"
        labels = enc["input_ids"].clone()
        labels[:, :n] = -100
    bos = proc.tokenizer.bos_token_id
    assert enc["input_ids"][0, 0] == bos and enc["input_ids"][0, 1] != bos, "BOS missing or doubled"
    return enc, labels


def _parts(names):
    """count module names per model part (the keys of NF4)"""
    n = dict.fromkeys(NF4, 0)
    for name in names:
        n[next((p for p in NF4 if p in name.split(".")), "other")] += 1
    return n


def _nf4(model):
    """T4/T7 probe: bnb Linear4bit modules per part. A dtype probe cannot see NF4: with bnb_4bit_quant_storage=bf16
    the packed 4-bit weight reports bf16 too"""
    import bitsandbytes as bnb
    return _parts(n for n, m in model.named_modules() if isinstance(m, bnb.nn.Linear4bit))


@app.function(volumes={"/vol": vol}, cpu=2.0, memory=8192, timeout=900)
def check():
    """CPU only: library versions, NF4 layout of a weightless model, data fields and counts, processor from the
    volume, token counts of real examples"""
    import importlib
    import math
    from collections import Counter
    import torch
    from transformers import AutoConfig, MuseGlimmerForConditionalGeneration
    from transformers.quantizers.base import HfQuantizer
    from transformers.quantizers.quantizers_utils import should_convert_module
    gt, proc, tok, ans = _env()
    # str(): torch.__version__ is a TorchVersion, which would not unpickle in the torch-less local Modal client
    vers = {m: str(importlib.import_module(m).__version__) for m in ("torch", "transformers", "peft", "bitsandbytes", "accelerate")}
    with torch.device("meta"):                           # the bnb quantizer's skip list and name test, no weights
        meta = MuseGlimmerForConditionalGeneration(AutoConfig.from_pretrained(WEIGHTS))
    skip = HfQuantizer.get_modules_to_not_convert(meta, SKIP_4BIT, meta._keep_in_fp32_modules)
    nf4 = _parts(n for n, m in meta.named_modules() if type(m) is torch.nn.Linear and should_convert_module(n, skip))
    assert nf4 == NF4, (nf4, NF4)
    data, counts = {s: gt.load_split(DATA, s) for s in ("val", "train")}, {}
    for s, subs in (("val", ("v1a", "v1b")), ("train", ("train",))):
        ms = [e["meta"] for e in data[s]]
        assert len({m["id"] for m in ms}) == len(ms), f"{s}: duplicate ids"
        bad = [m["id"] for e, m in zip(data[s], ms) if m["type"] not in ("cross", "kiss") or m["subset"] not in subs
               or not isinstance(m["P"], (int, float)) or not math.isfinite(m["P"]) or m["label"] != e["label"]]
        assert not bad, (s, len(bad), bad[:5])
        sm = _summary([{"type": m["type"], "P": m["P"], "label": e["label"], "pred": (None, 0, 1)[i % 3]}
                       for i, (e, m) in enumerate(zip(data[s], ms))])          # dummy predictions on the real fields
        assert sm["all"]["n"] == len(ms) and sm["ties"] == len(ms[::3]), sm
        counts[s] = {"n": len(ms), "A_true": sum(e["label"] == 0 for e in data[s]),
                     **Counter(f"{m['subset']}/{m['type']}" for m in ms),
                     "by_P": {b: v and v["n"] for b, v in sm["by_P"].items()}}
    val = [e for e in data["val"] if e["meta"]["subset"] == "v1a"]
    lens = []
    for ex in val[:8]:
        enc, _ = _encode(gt, proc, ex)
        enc0, _ = _encode(gt, proc, ex, ink=False)
        assert torch.equal(enc0["input_ids"], enc["input_ids"]) and not torch.equal(enc0["pixel_values"], enc["pixel_values"])
        enc2, lab = _encode(gt, proc, ex, gt.ANSWERS[ex["label"]])
        assert enc2["input_ids"][0, -2] == ans[gt.ANSWERS[ex["label"]]], "answer letter not at -2 (stage1 letter loss)"
        sup = tok.convert_ids_to_tokens(enc2["input_ids"][0][lab[0] != -100].tolist())
        lens.append(enc["input_ids"].shape[1])
    return {"versions": vers, "answer_ids": ans, "nf4": nf4, "counts": counts, "seq_len": lens, "supervised_tail": sup,
            "n_v1a": len(val), "pixel_values": list(enc["pixel_values"].shape), "grid": enc["image_grid_thw"].tolist()}


def _load(adapter=None):
    import torch
    from transformers import BitsAndBytesConfig, MuseGlimmerForConditionalGeneration
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_storage=torch.bfloat16,
                             llm_int8_skip_modules=SKIP_4BIT)
    model = MuseGlimmerForConditionalGeneration.from_pretrained(WEIGHTS, quantization_config=bnb, dtype=torch.bfloat16,
                                                                device_map={"": 0})
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    return model.eval()


def _score(model, gt, proc, exs, ans, deadline, ink=True):
    """forced choice: logit of 'A' vs 'B' as the first answer token; pred None on a tie (bf16 logits can be equal) or NaN"""
    import torch
    rows = []
    with torch.no_grad():
        for ex in exs:
            if time.time() > deadline:
                break
            enc, _ = _encode(gt, proc, ex, ink=ink)
            enc = {k: (v.to("cuda", torch.bfloat16) if k == "pixel_values" else v.to("cuda")) for k, v in enc.items()}
            lg = model(**enc).logits[0, -1].float()
            la, lb = lg[ans["A"]].item(), lg[ans["B"]].item()
            m = ex["meta"]
            rows.append({"id": m["id"], "type": m["type"], "P": m["P"], "label": ex["label"], "la": la, "lb": lb,
                         "pred": int(lb > la) if la < lb or lb < la else None, "top5": lg.topk(5).indices.tolist(),
                         "seq": enc["input_ids"].shape[1]})
    return rows


def _summary(rows):
    """a tie (pred None) scores 1/2, as in the data builder's path-only baselines; pred_A_frac counts decided rows"""
    import math
    def acc(rs):
        n = len(rs)
        if not n:
            return None
        k = sum(0.5 if r["pred"] is None else float(r["pred"] == r["label"]) for r in rs)
        p, z = k / n, 1.96                               # Wilson 95% interval
        c = (p + z * z / (2 * n)) / (1 + z * z / n)
        h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        return {"n": n, "acc": round(p, 4), "ci95": [round(c - h, 4), round(c + h, 4)]}
    bins = {"P<2": lambda P: P < 2, "2-5": lambda P: 2 <= P < 5, "5-10": lambda P: 5 <= P < 10, "P>=10": lambda P: P >= 10}
    dec = [r for r in rows if r["pred"] is not None]
    return {"all": acc(rows), "ties": len(rows) - len(dec),
            "pred_A_frac": round(sum(r["pred"] == 0 for r in dec) / max(1, len(dec)), 4),
            "by_type": {t: acc([r for r in rows if r["type"] == t]) for t in sorted({r["type"] for r in rows})},
            "by_P": {b: acc([r for r in rows if f(r["P"])]) for b, f in bins.items()}}


@app.function(gpu=GPU, volumes={"/vol": vol}, cpu=CPU, memory=MEM, timeout=36 * 60)
def stage1(n_val: int = 200, steps: int = 20, accum: int = 4, lr: float = 2e-4, rank: int = 16, budget_min: float = 30.0,
           tag: str = "stage1", force: bool = False):
    import gc
    import importlib
    import json
    import math
    import numpy as np
    import torch
    t0 = time.time()
    deadline = t0 + budget_min * 60                       # optional phases are skipped past this; timeout is the hard cap
    hard = t0 + 35 * 60                                   # the reload (T7) must end by here, 1 min before the timeout
    out = Path(f"/vol/out/{tag}")
    if (out / "report.json").exists() and not force:
        raise SystemExit(f"{out}/report.json exists: fetch it first, or rerun with --force to overwrite it")
    out.mkdir(parents=True, exist_ok=True)
    rep = {"gpu": torch.cuda.get_device_name(0), "rev": REV, "args": dict(n_val=n_val, steps=steps, accum=accum, lr=lr,
           rank=rank), "versions": {m: str(importlib.import_module(m).__version__) for m in
                                    ("torch", "transformers", "peft", "bitsandbytes", "accelerate")}}

    def save(**kw):
        rep.update(kw, elapsed_s=round(time.time() - t0))
        (out / "report.json").write_text(json.dumps(rep, indent=1))
        vol.commit()
        print(json.dumps({k: rep[k] for k in kw}, default=str)[:1500], flush=True)

    def phase(key, name, model, ink, until=None):
        """forced choice on v1a -> <name>.jsonl, summary in rep[key]; stops at `until` (default: the soft deadline)"""
        t = time.time()
        rows = _score(model, gt, proc, val, ans, until or deadline, ink)
        (out / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        n = max(1, len(rows))
        save(**{key: {**_summary(rows), "s_per_ex": round((time.time() - t) / n, 3),
                      "AB_in_top5": round(sum(ans["A"] in r["top5"] and ans["B"] in r["top5"] for r in rows) / n, 3),
                      "seq_mean": round(sum(r["seq"] for r in rows) / n, 1)}})
        return rows

    gt, proc, tok, ans = _env()
    vals = gt.load_split(DATA, "val")
    val = [e for e in vals if e["meta"]["subset"] == "v1a"][:n_val]
    n_v1b = sum(e["meta"]["subset"] == "v1b" for e in vals)              # stage 2's final evaluation set
    train = gt.load_split(DATA, "train")
    gb = lambda: round(torch.cuda.max_memory_allocated() / 2**30, 2)

    # 1. 4-bit load (T4): Linear4bit in the language model only, the vision path and lm_head stay bf16
    t = time.time()
    model = _load()
    nf4 = _nf4(model)
    save(load={"s": round(time.time() - t), "peak_GiB": gb(), "linear4bit": nf4})
    if nf4 != NF4:
        raise SystemExit(f"T4: Linear4bit per part {nf4}, want {NF4}")

    # 2. smoke generation with reasoning allowed (one example): the vision path works, what does it say?
    if time.time() < deadline:
        ex = val[0]
        img = gt.compose(ex["crop"], ex["cands"][0], ex["cands"][1])
        prompt = proc.apply_chat_template(gt.messages(), add_generation_prompt=True, tokenize=False)
        enc = proc(text=[prompt], images=[img], return_tensors="pt", add_special_tokens=False)
        enc = {k: (v.to("cuda", torch.bfloat16) if k == "pixel_values" else v.to("cuda")) for k, v in enc.items()}
        t = time.time()
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=384, do_sample=False)
        text = tok.decode(g[0, enc["input_ids"].shape[1]:], skip_special_tokens=False)
        save(smoke={"s": round(time.time() - t, 1), "new_tokens": int(g.shape[1] - enc["input_ids"].shape[1]),
                    "label": gt.ANSWERS[ex["label"]], "text": text[-1500:]})

    # 3. zero-shot forced choice on v1a (T5), then the no-ink control on the same examples
    phase("zeroshot", "zeroshot_v1a", model, True)
    phase("zeroshot_noink", "zeroshot_noink_v1a", model, False)

    # 4. QLoRA steps (bf16 base kept frozen, LoRA on the language projections only)
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(0)                                  # LoRA init and dropout
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.05, bias="none",
                                             target_modules=LORA_TARGETS))
    params = [p for p in model.parameters() if p.requires_grad]
    n_lora = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    rng = np.random.default_rng(0)
    order = rng.permutation(len(train))
    model.train()
    losses, letter_losses, times, mem = [], [], [], {}
    for s in range(steps):
        if time.time() > deadline:
            break
        torch.cuda.synchronize()
        t = time.time()
        for g_ in opt.param_groups:
            g_["lr"] = lr * min(1.0, (s + 1) / 3)          # 3-step warmup
        tot = tot_letter = 0.0
        for m in range(accum):
            ex = train[order[(s * accum + m) % len(order)]]
            enc, lab = _encode(gt, proc, ex, gt.ANSWERS[ex["label"]])
            enc = {k: (v.to("cuda", torch.bfloat16) if k == "pixel_values" else v.to("cuda")) for k, v in enc.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = model(**enc, labels=lab.to("cuda"))
            loss = o.loss / accum
            loss.backward()
            tot += loss.item()
            # the letter alone (4 of the 5 supervised tokens are fixed format tokens): it sits at -2 before <|eot|>,
            # so the logits at -3 predict it (check() asserts the position)
            tot_letter += torch.nn.functional.cross_entropy(o.logits[0, -3:-2].detach().float(),
                                                            enc["input_ids"][0, -2:-1]).item() / accum
        if not math.isfinite(tot):
            save(train={"aborted": f"non-finite loss at step {s}", "losses": losses, "letter_losses": letter_losses})
            raise SystemExit(f"non-finite loss at step {s}")
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        times.append(time.time() - t)
        losses.append(round(tot, 4))
        letter_losses.append(round(tot_letter, 4))
        if s in (0, 1):
            mem[f"after_step{s + 1}_GiB"] = gb()          # Adam state appears after step 1 (Glimmer docs)
        print(f"step {s + 1}/{steps} loss {tot:.4f} letter {tot_letter:.4f} {times[-1]:.1f}s", flush=True)
    model.save_pretrained(out / "adapter")                # committed by save(train=...) right below
    sec_ex = (float(np.mean(times[1:]) if len(times) > 1 else times[0]) / accum) if times else None
    # stage 2 at this function's all-in rate: two loads (train, eval), n x epochs examples, v1b with and without ink
    usd = lambda sec: round(sec / 3600 * USD_PER_H, 2)
    load_s, s_score = rep["load"]["s"], rep["zeroshot"]["s_per_ex"]
    est = {f"{n}x{ep}": {"train": usd(n * ep * sec_ex), "total": usd(2 * load_s + n * ep * sec_ex + 2 * n_v1b * s_score)}
           for n, ep in ((2000, 2), (3000, 2))} if times else None
    save(train={"steps": len(losses), "losses": losses, "letter_losses": letter_losses,
                "step_s": [round(x, 1) for x in times], "s_per_example": round(sec_ex, 3) if times else None,
                "lora_params": n_lora, "peak_GiB": gb(), **mem, "est_stage2_usd": est, "usd_per_h": round(USD_PER_H, 4),
                "n_v1b": n_v1b})

    # 5. the same v1a after the steps, with and without ink: a sanity check that the adapter changes the answers, not
    # a result. T7 has priority: its 8 reference rows come first, the full pass leaves room for the reload if one can
    # still fit, and the no-ink pass runs only if it and the reload both fit
    model.eval()
    ref = _score(model, gt, proc, val[:8], ans, hard)
    (out / "ref_v1a.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ref))
    save(ref_n=len(ref))                                   # the only post-training scores if the full pass gets no time
    t7_fits = lambda: time.time() + load_s + 60 < hard
    until = min(deadline, hard - load_s - 120) if t7_fits() else deadline
    post = phase("post_steps_sanity", "post_v1a", model, True, until)
    reload_s = 1.3 * load_s + 90                           # a fresh 4-bit load plus 8 examples, with margin
    if post and time.time() + len(val) * rep["post_steps_sanity"]["s_per_ex"] + reload_s < hard:
        phase("post_steps_noink", "post_noink_v1a", model, False)

    # 6. adapter round trip (T7): reload the saved adapter into a fresh 4-bit base, same logits? The gate has no margin
    # because a timeout loses nothing here: every phase is already committed. No steps: nothing trained to round-trip
    if losses and ref and t7_fits():
        model = opt = params = o = loss = enc = lab = None   # o, loss, enc, lab keep the last step's graph (~13 GB)
        gc.collect()
        torch.cuda.empty_cache()
        m2 = _load(str(out / "adapter"))
        nf4 = _nf4(m2)
        if nf4 != NF4:
            save(reload={"linear4bit": nf4})
            raise SystemExit(f"T7: Linear4bit per part {nf4}, want {NF4}")
        ids = [r["id"] for r in ref]
        again = {r["id"]: r for r in _score(m2, gt, proc, [e for e in val if e["meta"]["id"] in ids], ans, hard)}
        pairs = [(a, again[a["id"]]) for a in ref if a["id"] in again]
        (out / "reload_v1a.jsonl").write_text("".join(json.dumps(r) + "\n" for r in again.values()))
        worst = lambda xs: max(xs, key=lambda x: (x != x, x))   # a NaN must not hide: max(0.0, nan) is 0.0
        d = [worst((abs(a["la"] - b["la"]), abs(a["lb"] - b["lb"]))) for a, b in pairs]
        save(reload={"n": len(d), "max_abs_logit_diff": round(worst(d), 4) if d else None,
                     "same_pred": sum(a["pred"] == b["pred"] for a, b in pairs), "linear4bit": nf4})
        assert list(again) == ids, f"reload scored {list(again)}, want {ids}"
    save(done=True)
    return rep


@app.local_entrypoint()
def main(mode: str = "check", n_val: int = 200, steps: int = 20, accum: int = 4, lr: float = 2e-4, rank: int = 16,
         force: bool = False):
    import json
    if mode == "download":
        print(json.dumps(download.remote(), indent=1))
    elif mode == "check":
        print(json.dumps(check.remote(), indent=1))
    elif mode == "stage1":
        rep = stage1.remote(n_val, steps, accum, lr, rank, force=force)
        print(json.dumps(rep, indent=1, default=str)[:6000])
        print("results: modal volume get glimmer /out/stage1 out/ml/")
    else:
        raise SystemExit(f"unknown mode {mode}")

#!/usr/bin/env python
"""Headless FLUX image generation for the shlomi/ pipeline.

Designed for long unattended runs (nohup / background). Resumable (skips images
already on disk) and reproducible (deterministic per-image seeds). Reads prompts
from shlomi/data/prompts.json via the local utils.py and writes undegraded images
to shlomi/data/synthetic_clean/{class}/ plus a manifest.csv.

Run in the background (survives disconnect):
    cd ~/my_new_project/Plates/genai-project
    nohup /home/benshise/my_new_project/.venv/bin/python shlomi/generate_images.py \
        --gpu 6 --size 512 --steps 20 > shlomi/gen.log 2>&1 &
    tail -f shlomi/gen.log        # watch progress (Ctrl+C stops watching, NOT the job)

Stop it:   ps aux | grep generate_images   then   kill <pid>

Tip: for a much faster run, FLUX.1-schnell needs no license and ~4 steps:
    ... shlomi/generate_images.py --model black-forest-labs/FLUX.1-schnell --steps 4 --guidance 0
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", default="6", help="CUDA device index to use")
ap.add_argument("--size", type=int, default=512,
                help="generate+save resolution (step 03 degrades to ~128 anyway, so 512 is plenty)")
ap.add_argument("--steps", type=int, default=20, help="num_inference_steps")
ap.add_argument("--guidance", type=float, default=3.5, help="guidance_scale (use 0 for FLUX.1-schnell)")
ap.add_argument("--images-per-prompt", type=int, default=1)
ap.add_argument("--model", default="black-forest-labs/FLUX.1-dev")
args = ap.parse_args()

# Must be set BEFORE torch initialises CUDA.
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import torch                       # noqa: E402
from PIL import Image              # noqa: E402  (kept for parity; save uses PIL via the pipe output)
from diffusers import FluxPipeline  # noqa: E402

# Import the local shlomi/utils.py (this file lives in shlomi/).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import utils                       # noqa: E402


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    prompts = utils.load_prompts()
    total = sum(len(prompts[c]) for c in utils.CLASS_NAMES) * args.images_per_prompt

    log(f"loading {args.model} on GPU {args.gpu} (fp16 + cpu offload) ...")
    pipe = FluxPipeline.from_pretrained(args.model, torch_dtype=torch.float16)
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)
    log(f"loaded. target = {total} images @ {args.size}px, {args.steps} steps.")

    def generate(prompt: str, seed: int):
        gen = torch.Generator("cpu").manual_seed(seed)
        return pipe(
            prompt,
            height=args.size, width=args.size,
            guidance_scale=args.guidance,
            num_inference_steps=args.steps,
            max_sequence_length=512,
            generator=gen,
        ).images[0]

    t0 = time.time()
    gidx = made = done = 0
    for cls in utils.CLASS_NAMES:
        out_dir = utils.class_dir(utils.CLEAN_DIR, cls)
        for p_i, prompt in enumerate(prompts[cls]):
            for k in range(args.images_per_prompt):
                seed = utils.SEED + gidx
                gidx += 1
                done += 1
                fp = out_dir / f"{cls}_{p_i:04d}_{k}.jpg"
                if fp.exists():
                    continue
                generate(prompt, seed).save(fp, quality=95)
                made += 1
                if made % 10 == 0:
                    rate = (time.time() - t0) / made
                    eta_min = rate * (total - done) / 60
                    log(f"{done}/{total} done ({made} new this run) | "
                        f"{rate:.1f}s/img | ETA ~{eta_min:.0f} min")

    # Manifest (rebuilt from the same deterministic loop; robust to resumes).
    utils.CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    gidx = rows = 0
    with open(utils.MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filepath", "class", "seed", "model", "prompt"])
        for cls in utils.CLASS_NAMES:
            for p_i, prompt in enumerate(prompts[cls]):
                for k in range(args.images_per_prompt):
                    seed = utils.SEED + gidx
                    gidx += 1
                    fp = utils.CLEAN_DIR / cls / f"{cls}_{p_i:04d}_{k}.jpg"
                    if fp.exists():
                        w.writerow([str(fp.relative_to(utils.ROOT_DIR)), cls, seed, args.model, prompt])
                        rows += 1

    log(f"DONE. {made} new images this run | {rows} total on disk | "
        f"elapsed {(time.time() - t0) / 60:.0f} min | manifest -> {utils.MANIFEST_PATH}")


if __name__ == "__main__":
    main()

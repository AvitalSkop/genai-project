# CLAUDE.md — Plate Status Detection

GenAI course final project. A classifier that decides whether a restaurant table's plate should be cleared, from a single low-quality security-camera image.

## Source of truth
`Project_Plan_Plate_Status_Detection_3.md` is the full spec. **Read it before building anything.** This file holds only the rules and conventions that must hold in every session; the plan holds the detail.

## Critical domain rules (easy to get wrong — get these right)
- **Input is ONE still image of ONE plate.** Not video, not a sequence of frames. The crop is done by an upstream model and is out of scope — assume the input is already a tight, cropped single plate.
- **Five classes, defined by the plate's visible state** (not by cutlery, not by whether a diner is present):
  - `clean` — pristine, fresh, **unused** plate; no food, no crumbs, no residue → **do not clear**
  - `empty` — **used** plate eaten bare; only crumbs/sauce/residue → **clear**
  - `finished_leftovers` — small leftovers, or garbage like a napkin/paper → **clear**
  - `full` — a **moderate-to-full** serving of food (merges the old `semi_full` + `full`) → **do not clear**
  - `unclassified` — **too degraded to identify** the plate's state → **uncertain** (abstain; see below)
- **Binary decision rule (non-monotonic by design):** `clear = {empty, finished_leftovers}`, `do not clear = {clean, full}`, and `unclassified → uncertain` (abstain — no auto-action; safe fallback is *do not clear*). A `clean` plate has the least on it yet is *do not clear* (freshly set, diner about to eat), so "clear" is a band in the middle of the food-amount axis, not a threshold — this non-monotonic boundary is the main reason the fine-grained model exists.
- **`unclassified` has no prompts of its own.** In step 01 it borrows random prompts from the other classes (so the underlying images are normal plates); in `03_degrade_and_augment` those images are corrupted so heavily the food state is unreadable. The class only becomes meaningful *after* degradation, so it's inherently tied to the degraded condition.
- **`clean` vs `empty` is the subtlest pair — watch it.** Pristine-unused vs eaten-bare-with-crumbs can blur together under heavy degradation; expect that confusion in the matrix and treat it as error-analysis material, not a bug.
- **Cutlery is a nuisance attribute, not a label signal.** Vary it randomly across all classes in prompts so the model keys on food amount, never on cutlery presence.
- **`real_restaurant_cctv/` images are calibration/inspiration ONLY — never training or test data.** Use them to (a) inform diffusion prompts and (b) measure realistic degradation parameters (resolution, noise, blur, JPEG quality). The training/eval dataset is 100% synthetic.

## Hard technical gotchas
- **SDXL-Turbo ignores negative prompts.** At its intended `guidance_scale=0.0` there is no classifier-free guidance, so a negative prompt does nothing. Either run Turbo with strong positive prompts + manual culling, or switch to SD 1.5 / SDXL-base at `guidance_scale≈7`, `steps≈25` where negatives work. See Project_Plan_Plate_Status_Detection_3.md §4.3.
- **GPU-heavy work runs on Colab, not locally.** Image generation (SDXL) and fine-tuning (ViT/ResNet50/DINOv2) are written here but executed in Colab. Don't try to run them on this machine.
- **Never put "security camera" / "CCTV" / "surveillance" in a diffusion prompt.** Those phrases make the model burn a fake HUD overlay (timestamp, "REC", camera id) into the image — a spurious cue the classifier could learn instead of food amount. Keep diffusion outputs clean; the realistic CCTV degradation is added in `03_degrade_and_augment`, never in the prompt. This also keeps the with/without-degradation ablation valid.

## Conventions
- Code comments and docstrings in **English**.
- **Fixed seeds everywhere** (`torch.manual_seed(42)` etc.) — runs must be reproducible.
- **One shared train/val/test split** lives in `utils.py` and is imported by every notebook. Never re-split ad hoc.
- **Keep both undegraded and degraded image copies on disk** (`data/synthetic_clean/` = undegraded, `data/synthetic_degraded/`) so the with/without-degradation ablation can run. Note: "clean" in the folder name means *undegraded*, not the `clean` plate-state class — e.g. `data/synthetic_clean/clean/` holds the undegraded images of clean plates.
- Notebooks must run top-to-bottom on a fresh Colab. `requirements.txt` stays pinned and complete.

## Repo layout (target)
```
slides/  code/  data/  results/  visuals/  README.md  requirements.txt
```
Code notebooks are numbered in execution order: `01_generate_prompts` → `02_generate_images` → `03_degrade_and_augment` → `04_train_models` → `05_evaluate` → `06_gradio_app`. See Project_Plan_Plate_Status_Detection_3.md §8 for the full tree.

## How to build
- **Incrementally, one numbered step at a time.** Validate each before moving on. Do not attempt to build the whole project in one pass.
- Before scaling image generation, generate **10 per class** and confirm the four *content* states (`clean`, `empty`, `finished_leftovers`, `full`) are visually distinct after degradation — pay special attention to `clean` vs `empty`, the subtlest pair. (`unclassified` is the exception: it's *meant* to be unidentifiable after its heavier corruption.) Only then scale to the full set.

## Current state
Pre-implementation. Artifacts present: this file, `Project_Plan_Plate_Status_Detection_3.md`, and the `real_restaurant_cctv/` reference images. No code, `requirements.txt`, or build tooling exists yet — once it does, add the real run/setup commands here.

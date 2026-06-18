# Plate Status Detection — Master Project Plan

A complete playbook for the GenAI course final project. Built from the course rubric (`1023_ProjectRequirements`) and the sample project (`2041_GenAI_SampleProject`), adapted for your specific task and constraints (2–3 weeks left, Colab compute, proposal already submitted).

---

## 1. Project at a Glance

**Use case.** A restaurant security camera produces low-quality images of dining tables. Given a **single cropped still image of one plate** (cropping is handled by an upstream model — out of scope per the lecturer), classify the plate's current status so the system can decide whether staff should clear the plate.

**Input is one still image — not video, not a sequence of frames.** The model makes its decision from a single photograph of a single plate. No temporal information is used or assumed.

**Formal task.** Image classification, 5 fine-grained states (ordered by how much is on the plate) with a derived binary decision. Note the binary action is **non-monotonic** in food amount — see the note under the table:

| Class | What's on the plate | Final action |
|---|---|---|
| `clean` | A pristine, fresh, **unused** plate — no food, no crumbs, no residue at all | Do not clear |
| `empty` | A **used** plate eaten bare — only crumbs, sauce smears, or residue, no food | Clear |
| `finished_leftovers` | Small leftover food, or garbage such as a napkin or paper on the plate | Clear |
| `semi_full` | A moderate amount of food — more than leftovers, but the plate is not full | Do not clear |
| `full` | The plate is full of food | Do not clear |

**Binary decision rule:** `clear` = {`empty`, `finished_leftovers`} · `do not clear` = {`clean`, `semi_full`, `full`}. (If you intend a different mapping, this is the one line to change.)

> **The decision boundary is non-monotonic by design.** A `clean` plate has the *least* on it yet maps to **do not clear** (freshly set — the diner is about to eat), while `empty`/`finished_leftovers` map to **clear**, and `semi_full`/`full` go back to **do not clear**. So "clear" is a band in the middle of the food-amount axis, not a simple threshold. This is exactly why the fine-grained 5-class model earns its keep: a binary classifier keyed on "how full is the plate" cannot represent this boundary, but predicting the state first and then applying the rule can. Call this out in the report — it is a clean justification for the fine-grained taxonomy.

> **Why these five work well here.** Each label depends only on the **plate's visible state** — food amount plus any garbage (napkin, paper), and whether the plate is pristine vs used. That is fully visible in a single cropped still image, so there's no reliance on anything outside the frame (no diner, no cutlery position, no previous frame). This is also a friendlier target for the diffusion model: "empty vs some leftovers vs half-full vs full" produces visually distinct images that survive heavy degradation far better than subtle cues would — which makes the synthetic data and the degraded-image evaluation more reliable. The one genuinely subtle pair is **`clean` vs `empty`** — a pristine plate vs a plate eaten bare with faint crumbs/residue — which heavy degradation can blur together; expect (and discuss) that confusion in the matrix.

**Why this is novel** (the two angles that justify the project to graders):

1. **Domain-realistic synthetic data.** We don't just generate clean diffusion images — we degrade them to match the noise, blur, low resolution, and lighting of actual restaurant CCTV footage. This is the same paradigm as the "Vision Through Mud" past student project.
2. **Fine-grained state model, binary decision.** Going beyond binary classification gives the system more interpretable behavior and a much richer error analysis — the confusion matrix tells you which plate states are confused with which, not just one number. It also lets the system represent a **non-monotonic** decision — a `clean` plate and a `full` plate both mean *do not clear*, for opposite reasons — that a single binary food-amount classifier could not.

---

## 2. Locked Decisions

| Decision | Choice | Reasoning |
|---|---|---|
| Framework | PyTorch + Hugging Face `transformers` + `Trainer` | Matches course canon (see sample project pages 5–13) |
| Compute | Google Colab (free or Pro) | Your stated constraint |
| Diffusion model | SDXL-Turbo | ~1 s/image, good quality, fits Colab |
| Image size | 224 × 224 | Matches ViT/ResNet50 pretrained input |
| Backbones to compare | `google/vit-base-patch16-224`, `microsoft/resnet-50`, `facebook/dinov2-base` | Course canon, three architectures (transformer, conv, self-supervised) |
| Training regime | Two-pass per model: (a) frozen backbone + trained head, then (b) full fine-tuning | Lets you talk about feature extraction vs fine-tuning in the report |
| Dataset target | ~1,250–1,500 synthetic images (~250–300 per class, 5 classes) | Comfortably fits a Colab session |
| Real photos (50) | Used as prompt inspiration + as the source for degradation parameter calibration | Your choice; see "optional bonus" in §10 |
| Demo interface | Gradio app, deployed to Hugging Face Spaces | Free, one-click, matches spec |

---

## 3. The Novelty Story (How to Frame It)

The proposal slide describes the task. The interim and final must answer: **what's the contribution?** The course rubric is explicit that novelty is mandatory. Here's the one-paragraph version to keep on hand:

> Existing image classifiers for tableware are trained on clean, high-resolution food photographs that look nothing like what a restaurant security camera produces. We propose a two-part contribution: (1) a synthetic data generation pipeline that uses SDXL-Turbo plus a CCTV-style degradation module — calibrated to match real footage — to produce training data that survives the synthetic-to-real gap, and (2) a fine-grained 5-class plate-state taxonomy that lets the deployed system give richer signals to restaurant staff than a binary clear/don't-clear decision while preserving a simple lookup rule for the binary case — including a non-monotonic boundary (a `clean` freshly-set plate and a `full` plate both map to *do not clear*) that a plain binary classifier cannot express.

Hammer this paragraph in slide 1 of both interim and final.

---

## 4. Dataset Generation Pipeline

### 4.1 Class taxonomy and balance
Target ~250–300 images per class, so roughly:
- `clean`: 280
- `empty`: 280
- `finished_leftovers`: 280
- `semi_full`: 280
- `full`: 280

That's ~1,400 images total.

Keep classes balanced — you'll thank yourself when reading the confusion matrix, especially along the subtle `clean`/`empty` boundary.

### 4.2 Prompt generation (LLM step)
Use Claude or GPT to generate ~60–80 prompts per class. Use an **attribute-based prompt template** (the rubric explicitly praises attribute-based generation):

> "A {plate_color} {plate_shape} plate on a {table_surface}, viewed from {camera_angle}, with {plate_contents}, {cutlery_state}, under {lighting}."

Vary attributes per class:
- **Plate color:** white porcelain, off-white, beige, blue-rim
- **Plate shape:** round dinner plate, oval, square, pasta bowl, deep dish
- **Table surface:** dark wood, white tablecloth, marble, checkered cloth
- **Camera angle:** top-down or slight overhead (a ceiling camera almost always looks down on the table — keep this dominant)
- **Plate contents per class** (the plate's *visible state* is what defines the label):
  - `clean`: "a pristine, freshly set, completely unused clean plate — no food, no crumbs, no sauce, no residue of any kind"
  - `empty`: "a used plate that has been eaten bare — empty of food but with light crumbs, sauce smears, or residue left behind"
  - `finished_leftovers`: "a small amount of leftover food, or a crumpled napkin / piece of paper / wrapper on the plate"
  - `semi_full`: "a moderate amount of food covering part of the plate — clearly more than scraps, but the plate is not full"
  - `full`: "a full plate piled with a complete portion of food"
- **Cutlery state** (vary freely as a *nuisance attribute*, not a class signal): fork and knife on the plate, cutlery beside the plate, or no cutlery visible. Mixing this across all classes stops the model from cheating by keying on cutlery instead of food amount.
- **Lighting:** dim restaurant lighting, warm tungsten, fluorescent overhead

**Framing matters.** Because the deployed input is a *single cropped plate*, your training images must also be tight single-plate shots — not wide table scenes. Append to every prompt: "close-up of a single plate filling the frame, top-down view." This keeps the synthetic data in the same visual domain as the real cropped inputs, and reduces multi-plate contamination.

**Mind the `clean` vs `empty` boundary.** These two are the hardest pair to keep visually separable, especially after degradation. In prompts, push `clean` toward *obviously pristine and unused* (bright, spotless, freshly set) and `empty` toward *obviously used* (visible crumbs, sauce streaks, smudges). If after the 10-per-class pilot you still can't tell them apart in the degraded thumbnails, that's a signal to strengthen these prompts or to note the limitation up front.

Save prompts to `data/prompts.json` keyed by class. Inspect 5 prompts per class manually to catch garbage before generating images.

### 4.3 Image generation (diffusion step)
SDXL-Turbo on Colab. Key settings:
```python
pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")
# Turbo recommended settings:
# num_inference_steps=1-4, guidance_scale=0.0
```

**Important catch about negative prompts with Turbo.** SDXL-Turbo is meant to run at `guidance_scale=0.0`, and at that setting classifier-free guidance is off — which means **the negative prompt is silently ignored**. Two real options:
- **Stay on Turbo (fastest):** drop the negative prompt, lean on a strong positive prompt ("a single plate filling the frame, no other objects"), and cull bad images by hand. With ~1,400 images this is a bit over an hour of clicking.
- **Switch to SD 1.5 or SDXL base (slower, but negative prompts work):** run at `guidance_scale≈7` with `num_inference_steps≈25`. Then a negative prompt like "multiple plates, hands, face, table edge, watermark, text" actively suppresses contamination. On Colab that's several seconds per image instead of one — still fine for ~1,400 images if you let it run.

For your timeline, start with Turbo + manual culling, and only fall back to SD 1.5 + CFG if you see heavy multi-plate or hands-in-frame contamination.

**Do not put "security camera" / "CCTV" / "surveillance" in the prompt.** It makes the model render a fake HUD overlay (timestamp, "REC", camera id) into the image — a spurious cue the classifier could latch onto instead of food amount. Keep the diffusion output a clean photo; the realistic CCTV look is added by the degradation pipeline in §4.4, **not** the prompt. (If overlays still slip in on a CFG model, suppress them with a negative prompt such as "text, watermark, timestamp, on-screen display, camera UI".)

Save with UUID filenames into `data/synthetic_clean/{class}/`. Inspect a sample grid per class — discard obvious failures manually (this is fine and the dog-breed example does it implicitly).

### 4.4 Degradation pipeline (NOVELTY #1 — this is your contribution)
Before training, pass each image through a CCTV-style degradation. Use the 50 real online photos to calibrate parameters:

1. Measure each real photo's resolution distribution → pick a typical low resolution (e.g., 96×96 or 128×128).
2. Estimate the noise floor (standard deviation of pixel values in flat regions).
3. Estimate motion blur extent (if any visible).
4. Estimate JPEG quality (if metadata available, or by visual estimate).

Then apply, per image, a randomized pipeline using `albumentations` or `torchvision.transforms.v2`:
```python
A.Compose([
    A.Downscale(scale_min=0.25, scale_max=0.5, p=1.0),    # res drop
    A.GaussNoise(var_limit=(10, 50), p=0.8),               # sensor noise
    A.MotionBlur(blur_limit=(3, 7), p=0.4),                # motion blur
    A.ImageCompression(quality_lower=20, quality_upper=60, p=0.8),  # JPEG artifacts
    A.RandomBrightnessContrast(p=0.6),                     # poor lighting
    A.Perspective(scale=(0.05, 0.1), p=0.5),               # CCTV angle
    A.Resize(224, 224),                                    # back to model input size
])
```

Save degraded versions to `data/synthetic_degraded/{class}/` — keep both the undegraded copies (`data/synthetic_clean/`) and the degraded copies so you can run the ablation in §6. (Terminology: "clean" in `synthetic_clean/` means *undegraded* — unrelated to the `clean` plate-state class, so `data/synthetic_clean/clean/` holds the undegraded images of clean plates.)

### 4.5 Standard augmentations (for training only)
On top of degradation, apply standard augmentations during training: `RandomHorizontalFlip`, `RandomRotation(15)`, `ColorJitter`. These are independent of the degradation pipeline.

### 4.6 Train / val / test split
- 70% train, 15% val, 15% test — stratified by class.
- Set seeds (e.g., `torch.manual_seed(42)`) so splits are reproducible.
- Document the split logic in `code/utils.py` so the same split is used by every notebook.

---

## 5. Modeling Pipeline

All three models share the same HF training loop and hyperparameters, only the `model_id` changes — this mirrors the course example exactly.

### 5.1 Hyperparameters (from course example, lightly adjusted)
```python
TrainingArguments(
    per_device_train_batch_size=32,    # 32 not 64; SDXL outputs are larger memory-wise
    per_device_eval_batch_size=64,
    num_train_epochs=10,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="best",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=True,
    logging_steps=10,
    dataloader_num_workers=4,
)
```

### 5.2 Two regimes per model
1. **Feature extraction:** freeze the backbone, train only the classifier head. ~1–2 epochs is enough; this is your "off-the-shelf" baseline.
2. **Full fine-tuning:** unfreeze everything. 10 epochs.

Report both. The course rubric explicitly asks for "few fine-tuned models vs. few off-the-shelf models" — the frozen runs cover the off-the-shelf side.

### 5.3 What goes in `code/04_train_models.ipynb`
- Load preprocessed dataset
- Loop over 3 model IDs × 2 regimes = 6 runs
- For each run: save best checkpoint, training curves (loss + acc per epoch), val confusion matrix
- Save all results to `results/{model}_{regime}.json`

---

## 6. Evaluation Plan

### 6.1 Metrics (all reported)
- **Top-1 accuracy** — headline number
- **Macro-F1** — robust to mild class imbalance
- **Per-class precision/recall** — interpretation aid
- **Confusion matrix** — required by spec, must appear in final slides
- **Binary-derived metrics** — accuracy and F1 of the clear/don't-clear decision after the rule is applied

### 6.2 Required comparisons (the "results" slide of the final)
1. **Model-vs-model:** ViT vs ResNet50 vs DINOv2, both regimes — single table.
2. **Training curves:** loss and accuracy across train/val for each model — single figure with subplots, prevents the overfitting question.
3. **Confusion matrices:** one per model, side-by-side.

### 6.3 Ablation (this is the differentiator — do this and you score noticeably higher)
Train the best model **with** and **without** the degradation augmentation. Evaluate both on:
- A clean-synthetic test split
- A degraded-synthetic test split
- (Optional bonus) the 50 real photos held out

Expected result: the "with degradation" version takes a tiny hit on clean synthetic but does dramatically better on the degraded set. This is your money chart.

### 6.4 Error analysis (required per the rubric)
Pick the 5–10 worst-misclassified examples from your best model. For each, show:
- The image
- Predicted class, true class, confidence
- A one-sentence hypothesis for why it failed (e.g., "model confused `semi_full` with `full` because the food covered most of the plate and heavy downscaling erased the bare rim that would have signalled a partial portion")

Put this in `results/error_analysis.csv` and as a slide in the final.

---

## 7. Demo Interface (Gradio)

`code/06_gradio_app.py`:
- Single image upload widget
- On submit:
  - Run the image through the best model
  - Show 5 class probabilities as a horizontal bar chart
  - Show the binary decision (CLEAR / DO NOT CLEAR) in large text with the confidence percentage
  - Show the input image with the degradation pipeline applied next to it (transparency about what the model "saw")
- Deploy to Hugging Face Spaces (free tier handles this easily)

Include the HF Spaces URL in your README and on the final slide — it's a live demo the lecturer can poke at during the defense.

---

## 8. GitHub Repository Structure

Follow the rubric exactly (the GitHub Repository Requirements slide of `1023_ProjectRequirements`).

```
plate-status-detection/
├── README.md
├── requirements.txt
├── .gitignore
├── slides/
│   ├── proposal.pptx
│   ├── proposal.pdf
│   ├── interim.pptx
│   ├── interim.pdf
│   ├── final.pptx
│   └── final.pdf
├── code/
│   ├── 01_generate_prompts.ipynb
│   ├── 02_generate_images.ipynb
│   ├── 03_degrade_and_augment.ipynb
│   ├── 04_train_models.ipynb
│   ├── 05_evaluate.ipynb
│   ├── 06_gradio_app.py
│   └── utils.py
├── data/
│   ├── prompts.json
│   ├── class_taxonomy.md
│   ├── synthetic_clean/         # one subfolder per class
│   ├── synthetic_degraded/      # one subfolder per class
│   └── splits/                  # train/val/test JSON manifests
├── results/
│   ├── vit_full.json
│   ├── vit_frozen.json
│   ├── resnet50_full.json
│   ├── resnet50_frozen.json
│   ├── dinov2_full.json
│   ├── dinov2_frozen.json
│   ├── confusion_matrices/
│   ├── ablation_with_vs_without_degradation.csv
│   └── error_analysis.csv
└── visuals/
    ├── visual_abstract.png
    ├── sample_images_per_class.png
    ├── degradation_before_after.png
    ├── training_curves.png
    ├── confusion_matrices.png
    └── ablation_chart.png
```

### README sections (required by rubric — page 11 of requirements PDF)
1. Project motivation
2. Problem statement
3. Visual abstract (an image — make a clean one)
4. Datasets used or collected
5. Data augmentation and generation methods
6. Input/Output examples
7. Models and pipelines used
8. Training process and parameters
9. Metrics
10. Results
11. Repository structure
12. Team members

Code comments in English (the spec says so). Inline docstrings on every function.

---

## 9. Interim Deliverable — 5-Slide Outline

The rubric's interim presentation maps directly onto five slides. Each slide answers specific questions.

### Slide 1 — Project Review
- One-line motivation
- Task statement (5-class plate state classification → binary clear decision)
- What's novel (the §3 paragraph, compressed)
- Any changes from the proposal

### Slide 2 — Previous Work
A table reviewing **3 recent, highly-cited papers**. Verify each on Google Scholar before citing. Suggested starting points (check current citations and titles):
- Azizi et al., "Synthetic Data from Diffusion Models Improves ImageNet Classification" (2023) — most directly relevant, justifies your synthetic-data approach.
- He et al., "Is Synthetic Data from Generative Models Ready for Image Recognition?" (ICLR 2023) — examines limits of synthetic-only training.
- Sariyildiz et al., "Fake it till you make it: Learning(s) from a synthetic ImageNet clone" (CVPR 2023) — strong methodology comparable to yours.
- Hendrycks & Dietterich, "Benchmarking Neural Network Robustness to Common Corruptions and Perturbations" (ICLR 2019) — the foundational paper on the degradation angle.

Table columns: Title / Year, Task, Methods, Data, Results, Relation to your project.

### Slide 3 — Dataset
- Class taxonomy table (the one from §1)
- Prompt template + 1 example per class
- Generation pipeline diagram: LLM prompts → SDXL-Turbo → degradation → augmentation → training set
- EDA: class distribution histogram, image resolution stats, brightness/contrast stats, a sample grid showing a few examples per class (undegraded + degraded side by side)

### Slide 4 — Baseline Solution and Results
- Pretrained ViT, frozen backbone, classifier head only, 2 epochs
- Initial top-1 accuracy and macro-F1
- Initial confusion matrix
- 3–5 example errors with short hypotheses

### Slide 5 — Plan (as a table)
| Item | Due | Outcome |
|---|---|---|
| Full fine-tuning of ViT, ResNet50, DINOv2 | Week 2 day 9 | Full comparison results |
| Degradation ablation | Day 12 | "With vs without" chart |
| Gradio app + deploy to HF Spaces | Day 14 | Live demo link |
| Final report and visual abstract | Day 18 | Final slides + README |
| Final presentation prep | Day 20 | Practice run |

---

## 10. Final Deliverable — 5-Slide Outline

### Slide 1 — Refined project definition
- Single sentence: motivation + task + the two novelty angles
- The visual abstract (do not skip — graders love this)

### Slide 2 — Project achievements and novelty
- "We did X, Y, Z" — bullet your concrete outputs (~1,400 synthetic images across 5 classes, 3 backbones × 2 regimes compared, degradation ablation, Gradio demo)
- The novelty paragraph one more time, sharpened

### Slide 3 — Methodology review
- Diagram: prompts → diffusion → degradation → augmentation → 3 backbones → evaluation
- Hyperparameters table (your modified course-canon config)
- Training process notes (epochs, eval strategy, best-checkpoint loading)

### Slide 4 — Results
- Headline accuracy/F1 table: 3 models × 2 regimes
- Training curves grid (3 subplots, one per model)
- Confusion matrices (3 side-by-side)
- **The ablation chart** (with vs without degradation, on clean vs degraded test) — this is the showcase
- Worst-error gallery (5 examples)

### Slide 5 — Conclusion
- Did we hit the goal? Yes/partly/no, with numbers
- What we learned (specific insights, e.g., "DINOv2 features transferred better than ImageNet ResNet50 features under degradation — consistent with the self-supervised pretraining literature")
- Future work: collect real labeled CCTV data, evaluate on it, deploy to a pilot restaurant

### What to NOT do on the final slides
- No "wall of text" slides — every bullet should be ≤ 12 words.
- No screenshots of code (results only).
- No claim of generalization to "all plates" — you trained on synthetic.

---

## 11. Sprint Timeline (2–3 weeks)

### Week 1 — Dataset + interim deliverable
- **Day 1:** Lock taxonomy. Write LLM prompt template. Generate first 50 prompts and inspect quality.
- **Day 2:** Generate the rest (~280/class). Generate all ~1,400 undegraded synthetic images on Colab.
- **Day 3:** Build the degradation pipeline. Calibrate parameters by inspecting your 50 real photos. Apply to all synthetic images.
- **Day 4:** Build the train/val/test split. Write `utils.py` with the data loaders.
- **Day 5:** Run baseline ViT with frozen backbone. Get first metrics. Plot confusion matrix.
- **Day 6:** EDA notebook (class dist, image stats, sample grids). Read your 3 papers.
- **Day 7:** Write interim slides. Polish baseline results.

### Week 2 — Full comparison + final deliverable
- **Day 8:** Full fine-tune ViT and ResNet50. Save curves.
- **Day 9:** Full fine-tune DINOv2. Save curves. Submit interim.
- **Day 10:** Evaluate all 6 runs. Build confusion matrices. Pick the best model.
- **Day 11:** Ablation: train best model without degradation. Evaluate on clean + degraded.
- **Day 12:** Error analysis: collect 5–10 worst errors with hypotheses.
- **Day 13:** Build Gradio app skeleton, test locally.
- **Day 14:** Deploy to HF Spaces. Make screenshots / short screen recording.

### Week 3 (or compressed buffer days)
- **Day 15:** Visual abstract. Final figures (training curves, confusion matrices, ablation chart, error gallery).
- **Day 16:** Write final slides. Iterate slide 1 (motivation/novelty) until it's a single tight paragraph.
- **Day 17:** Polish the README — every section the rubric lists.
- **Day 18:** Clean up the repo: remove dead code, add `requirements.txt`, ensure notebooks run top-to-bottom on a fresh Colab.
- **Day 19:** Defense practice. Time the presentation to 10 minutes.
- **Day 20:** Submit.

---

## 12. What "Perfect" Looks Like — Top-Mark Differentiators

These are the things that separate a 90-grade project from a 100-grade one. Each costs little once the core work is done.

1. **A real visual abstract.** Not a screenshot of slide content — a clean diagram showing the pipeline (prompts → diffusion → degradation → classifier → decision). Use any tool — even draw.io is fine. Put it at the top of the README and on slide 1 of the final.
2. **The ablation chart.** With vs without degradation, on clean vs degraded test data. This single chart, more than anything else, demonstrates that your novelty contribution actually works.
3. **Confusion matrices that you talk about.** Don't just dump them; for each confusion matrix, pick ONE off-diagonal cell and explain in one sentence why those two classes are confused. Graders love this.
4. **DINOv2 commentary.** DINOv2 was trained self-supervised on natural images — comment on whether its features transfer better/worse than ImageNet-supervised ResNet50, and offer a hypothesis. This shows you understand what the models actually are.
5. **Reproducibility.** Seeds locked everywhere, notebooks run top-to-bottom on a fresh Colab, `requirements.txt` pinned. The README should let a stranger reproduce your work.
6. **Saliency maps (optional bonus).** One slide with Grad-CAM or attention-rollout visualizations showing where ViT looks — does it focus on the food, the plate edge, the cutlery? This is the kind of thing that gets called out in defense.
7. **Live demo.** A working HF Spaces URL beats every screenshot. Make sure it loads in under 30 seconds during the defense.
8. **Optional bonus — small real-world sanity check.** If you have time at the end, run your best model on the 50 real photos (the ones you used as inspiration). Even without ground truth labels you can eyeball it and report something like "the model's predictions on real-world photos were consistent with human intuition in 38/50 cases." This recovers most of the credit of having a real test set without doing the work of properly labeling one.

---

## 13. Quick Checklist Before Submission

- [ ] Repo has all six folders (slides, code, data, results, visuals + README)
- [ ] README hits all 12 sections from the rubric
- [ ] Inline code comments are in English
- [ ] Visual abstract exists and is on slide 1 of the final
- [ ] All 6 training runs have results files
- [ ] At least 3 papers cited in the interim
- [ ] EDA in the interim (class dist + sample grids + image stats)
- [ ] Confusion matrices for all 3 final models
- [ ] Ablation chart (with vs without degradation)
- [ ] Error analysis with 5+ examples
- [ ] Gradio app deployed and URL in README
- [ ] Notebooks run top-to-bottom on a fresh Colab
- [ ] `requirements.txt` is pinned and complete
- [ ] Seeds are fixed in every notebook

If every box is ticked, you're not just submitting — you're submitting a defensible, top-grade project.

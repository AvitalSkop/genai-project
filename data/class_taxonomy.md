# Plate-state class taxonomy

Five classes. The first four are *content* classes ordered by how much is on the
plate (low → high); `unclassified` is a special class for images too degraded to read.
The label is defined by the plate's **visible state** only — never by cutlery or by
whether a diner is present. The binary action is **non-monotonic** by design.

| Idx | Class | Visible state | Action |
|---|---|---|---|
| 0 | `clean` | Pristine, **unused** plate — no food, no crumbs, no residue | **do not clear** |
| 1 | `empty` | **Used** plate eaten bare — only crumbs / sauce smears / residue | **clear** |
| 2 | `finished_leftovers` | Small leftovers, or garbage (napkin / paper / wrapper) | **clear** |
| 3 | `full` | A **moderate-to-full** serving of food (merges old `semi_full` + `full`) | **do not clear** |
| 4 | `unclassified` | **Too degraded to identify** — can't tell whether there is food | **uncertain** (abstain) |

## Binary decision rule
- `clear` = { `empty`, `finished_leftovers` }
- `do not clear` = { `clean`, `full` }
- `unclassified` → **uncertain** — abstain / flag for human review; the safe fallback
  if a hard binary is forced is *do not clear* (never auto-clear a plate you can't see).

**Non-monotonic on purpose.** A `clean` plate has the *least* on it yet maps to
*do not clear* (a freshly set plate — the diner is about to eat), so "clear" is a band
in the middle of the food-amount axis, not a simple threshold. This is the main reason
for the fine-grained model: a plain binary "how full is the plate" classifier cannot
represent this boundary.

## The `unclassified` class
`unclassified` has **no prompts of its own**. In `01_generate_prompts` it borrows random
prompts from the four content classes, so its raw images are ordinary plates. In
`03_degrade_and_augment` those images are corrupted far more heavily than usual
(aggressive downscale, blur, noise, JPEG) until the plate state is unreadable — that
corruption is what *defines* the class. Consequence: `unclassified` is only meaningful in
the degraded condition (relevant to the with/without-degradation ablation).

## Notes for data generation
- **Cutlery is a nuisance attribute**, varied randomly across all content classes so the
  model keys on food amount, not on cutlery presence.
- The **`clean` vs `empty`** pair is the subtlest and the one to watch under heavy
  degradation (pristine vs eaten-bare-with-crumbs).
- Class indices above are the canonical order used everywhere — see
  [`code/utils.py`](../code/utils.py) (`CLASS_NAMES`).

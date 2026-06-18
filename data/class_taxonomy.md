# Plate-state class taxonomy

Five fine-grained classes, ordered by how much is on the plate (low → high).
The label is defined by the plate's **visible state** only — never by cutlery
or by whether a diner is present. The binary action is **non-monotonic** by design.

| Idx | Class | Visible state | Binary action |
|---|---|---|---|
| 0 | `clean` | Pristine, **unused** plate — no food, no crumbs, no residue | **do not clear** |
| 1 | `empty` | **Used** plate eaten bare — only crumbs / sauce smears / residue | **clear** |
| 2 | `finished_leftovers` | Small leftovers, or garbage (napkin / paper / wrapper) | **clear** |
| 3 | `semi_full` | Moderate food — more than scraps, plate not full | **do not clear** |
| 4 | `full` | Full plate of food | **do not clear** |

## Binary decision rule
- `clear` = { `empty`, `finished_leftovers` }
- `do not clear` = { `clean`, `semi_full`, `full` }

**Non-monotonic on purpose.** A `clean` plate has the *least* on it yet maps to
*do not clear* (a freshly set plate — the diner is about to eat), so "clear" is a
band in the middle of the food-amount axis, not a simple threshold. This is the
main reason for the fine-grained 5-class model: a plain binary "how full is the
plate" classifier cannot represent this boundary.

## Notes for data generation
- **Cutlery is a nuisance attribute**, varied randomly across all classes so the
  model keys on food amount, not on cutlery presence.
- The **`clean` vs `empty`** pair is the subtlest and the one to watch under heavy
  degradation (pristine vs eaten-bare-with-crumbs).
- Class indices above are the canonical order used everywhere — see
  [`code/utils.py`](../code/utils.py) (`CLASS_NAMES`).

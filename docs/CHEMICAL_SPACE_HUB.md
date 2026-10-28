# Chemical Space Selection Hub

**Last updated:** 2026-10-28

SMILES2Select helps researchers explore, compare and deliberately sample
chemical space. Chemical eligibility, drug-likeness classification, structural
alerts, chemical-space location, reference novelty and final sampling remain
separate layers.

The seven-step wizard and separate Hub remain available. The Hub foundations are
exposed in the Python API and CLI through reference libraries, exact overlap detection,
fingerprint novelty, selection strategies, final/reserve counts and English
method cards.

Important methodological rule: PCA and UMAP coordinates are for
visualization. Their two-dimensional distance is never used as the molecular
similarity or novelty metric.

## Desktop Hub navigation

Use **Configure** to show **Chemical ranking**, **Model (experimental)**,
**Contextual (experimental)** or **Tolerance (experimental)**. Switching this
selector only changes the visible form; it does not apply a method or change the
final library.

The top area keeps **Requested N**, the applied method, final count and difference
from the request visible while the forms scroll. Use **Create selection** for
chemical ranking, or the chosen method's preview and adopt buttons for a proposal.
**Inspect applied criteria...**, **Undo**, **Redo** and both export actions remain
in this area. Export is blocked during computation, with unapplied criteria or
with no final molecules; a count mismatch requires confirmation.

Expand **Quotas / objectives** for shared quotas and chemical property objectives.
**Refresh map** and **Display...** sit beside the map; display settings include
projection, color, reference overlay and local Papyrus evidence. Inspecting a
point shows **Identity / properties**, **Selection / decision** and
**Evidence / provenance**, with the existing **Shortlist**, **Select**, **Exclude**
and **Pin** actions.

The view selector offers **Chemical space**, **Pareto** and **Table**. The read-only
Table exposes all evaluable Hub candidates, including chemical screening failures,
without a 5,000-row cap; this does not make every row eligible for model selection.
Stable **Record ID** links row inspection to both plots and the inspector, even
when external molecule IDs repeat. **Final for export** states actual membership,
separately from chemical and selection status. Sorting or inspecting changes no
selection decisions; a green outline ring marks inspection, while gold squares
continue to mark final molecules.

Use **Compare molecules...** in the persistent action row for a separate window
that stays open while you inspect records. **Keep current molecule as A** keeps
a view reference only; B follows inspection by **Record ID**. Both columns show
2D structures, identity, properties, current basket states and applied evidence.
Keeping A does not select or pin a molecule. The applied-method banner follows
the applied selection; pending model, contextual and tolerance evidence remains
in the main inspector.

See the [GUI guide](GUI_GUIDE.md) for comparison, preview, adoption and export details.

## CLI example

```text
smiles2select candidates.csv --smiles-column SMILES --id-column ID \
  --reference ecbd.csv --reference-smiles-column SMILES \
  --reference-id-column ID --exclude-reference-duplicates \
  --selection-strategy reference_aware_diversity \
  --final-count 3000 --reserve-count 500 \
  --database session.sqlite --excel session.xlsx
```

The report records the reference libraries, exact search mode, Morgan
fingerprint settings, nearest reference, maximum similarity and
`reference_novelty = 1 - max_reference_similarity`.

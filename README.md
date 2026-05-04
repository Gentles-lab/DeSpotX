# DeSpotX

DeSpotX, a deep generative model for single-cell spatial transcriptomics decontamination.

![DeSpotX overview](DeSpotXOverview.png)

## Quickstart with example data

A 4,237-cell × 550-gene MERFISH brain subset with a 10% ambient-RNA spike-in
ships in `example_data/` for end-to-end testing:

- `merfish_brain_example.h5ad` — 25 cell types in `obs.cell_type_manual`,
  spatial coordinates in `obsm["spatial"]`.
- `merfish_brain_example_ground_truth.npz` — keys `spiked_counts`,
  `contam_counts`, `eps_per_cell` (per-cell ground truth for evaluation).
  True signal = `spiked_counts − contam_counts`.

Run DeSpotX:

```bash
python scripts/run_despotx.py \
    --h5ad example_data/merfish_brain_example.h5ad \
    --cell-type-col cell_type_manual \
    --spatial-key spatial \
    --out example_output/ \
    --device cpu \
    --epochs 5
```

Evaluate against ground truth:

```python
import numpy as np
import anndata as ad
import pandas as pd

gt = np.load("example_data/merfish_brain_example_ground_truth.npz")
true = (gt["spiked_counts"] - gt["contam_counts"]).clip(min=0)
adj = ad.read_h5ad("example_output/adjusted.h5ad").X.toarray()
pred_eps = pd.read_csv("example_output/contamination.tsv", sep="\t")["contamination"].values

print("retention :", adj.sum() / true.sum())
print("epsilon MAE:", np.mean(np.abs(pred_eps - gt["eps_per_cell"])))
```

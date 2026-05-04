# DeSpotX

DeSpotX, a deep generative model for single-cell spatial transcriptomics decontamination.

![DeSpotX overview](DeSpotXOverview.png)

## Install

DeSpotX requires Python ≥ 3.9. Install the dependencies:

```bash
pip install torch torch_geometric anndata numpy pandas scipy matplotlib
```

`matplotlib` is only needed for the optional training-loss figure;
the model trains and runs without it.

Clone this repository — no pip-install of the package itself is required;
`scripts/run_despotx.py` adds the parent directory to `sys.path` at runtime:

```bash
git clone https://github.com/<org>/DeSpotX.git
cd DeSpotX
```

## Quickstart with example data

A ~12,000-cell × 550-gene MERFISH brain subset (raw, single tissue
sub-region, 25 cell types) ships in `example_data/`:

- `merfish_brain_example.h5ad` — cell-type labels in
  `obs.cell_type_manual`, spatial coordinates in `obsm["spatial"]`.

Run DeSpotX:

```bash
python scripts/run_despotx.py \
    --h5ad example_data/merfish_brain_example.h5ad \
    --cell-type-col cell_type_manual \
    --spatial-key spatial \
    --out example_output/ \
    --device cuda
```

After decontamination, UMAP shows tighter, better-separated clusters:

![Raw vs DeSpotX-decontaminated UMAP](example_data/umap_raw_vs_despotx.png)

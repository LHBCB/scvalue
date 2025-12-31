# scValue

<p align="center">
    <img width="100%" src="scValue_workflow.png" alt="scValue workflow">
</p>

`scValue` is a Python package for **value-based subsampling (“sketching”)** of large scRNA-seq datasets for downstream machine / deep learning tasks.

scValue assigns each cell a **data value (DV)** using **out-of-bag (OOB)** estimates from a random forest classifier. Intuitively, a cell gets a higher DV if it is more helpful for correctly distinguishing its cell type in OOB predictions. scValue then builds a subsample by:
1. **Computing DV for each cell** (OOB-based).
2. **Allocating sketch size across cell types** (DV-weighted or proportional).
3. **Selecting cells within each cell type** using a DV-guided strategy (binning or top-pick).

Implemented using: `Python 3.10`, `scikit-learn 1.5.2`, `SciPy 1.14.0`, `NumPy 1.26.4`, `pandas 1.5.3`, `joblib 1.4.2`.  
Experiments in the paper used: `Scanpy 1.10.2` and `AnnData 0.10.8`.

---
## Installation
Install from PyPI:
``` python
pip install scvalue
```

## Quick start
Load modules
```python
import scanpy as sc
from scvalue import SCValue
```
### Read and preprocess an input scRNA-seq dataset 
The example [mTC dataset](https://datasets.cellxgene.cziscience.com/6c253303-8255-4004-8464-5e90dd5846cb.h5ad) (mouse T cells; 89,429 cells) is available in `.h5ad` format. Download it (example with `wget`):
```
wget -O mTC.h5ad "https://datasets.cellxgene.cziscience.com/6c253303-8255-4004-8464-5e90dd5846cb.h5ad"
```
It is already log-normalised. Here we compute PCA on the top 3,000 HVGs. By default, scValue expects features from `adata.obsm["X_pca"]` (`use_rep="X_pca"`).
```{python}
adata = sc.read_h5ad('mTC.h5ad')

sc.pp.highly_variable_genes(adata, n_top_genes=3000)
sc.pp.pca(adata, n_comps=50) # stores PCs in adata.obsm["X_pca"]
```

### Subsample by scValue
Subsample 10% of cells and return an AnnData sketch:
```{python}
scv = SCValue(
    adata=adata,
    sketch_size=0.1,          # float in (0, 1] or an integer cell count
    use_rep="X_pca",          # default: "X_pca"; use "X" to use adata.X
    cell_type_key="cell_type",
    n_trees=100,
    type_weighting="inv_cv",  # "inv_cv" (default), "sqrt_n_sd", or "proportional"
    strategy="FB",            # "FB" (default), "MTB", or "TP"
    write_dv=False,           # if True, writes dv_values.csv and dv_summary.csv
    seed=42,
)

adata_sub = scv.value()       # also available as scv.adata_sub
```
After running:
- DV is stored in `adata.obs["dv"]` (and included in `adata_sub.obs["dv"]`).
- `scv.dv` stores a DataFrame of cell type + DV for the full dataset.
- `scv.adata_sub` is the subsampled `AnnData`.

The sketch (`adata_sub`) can be visualised with the usual Scanpy workflow (neighbours/UMAP/etc.). The function `plot_umap` in `reproducibility/eval_time_hd_gini.py` can be used to reproduce the mTC demonstration in our paper. 

All datasets used in our study are publicly available, as detailed in the [data availability](https://academic.oup.com/bib/article/26/3/bbaf279/8162443#523154998) section.

## Parameters
### Core inputs
- `sketch_size` (required):
    - `float` in (0, 1] = fraction of cells to keep, or
    - `int` = exact number of cells to keep.
- `use_rep` (default `"X_pca"`): feature matrix used to fit the Random Forest and compute DV.
    - `"X_pca"` expects `adata.obsm["X_pca"]` (e.g., from `sc.pp.pca`).
    - `"X"` uses `adata.X` (note: sparse matrices may be densified internally).
- `cell_type_key` (default `"cell_type"`): column name in `adata.obs` containing cell type labels.
### Step 1: data value computation
- `n_trees` (default `100`): number of trees in random forest
- `seed` (default `42`): random seed for reproducibility.
### Step 2: sketch size determination
- `type_weighting` for each cell type
    - `"inv_cv"` (default): weights ∝ mean(DV) * sqrt(N) / std(DV)
    - `"sqrt_n_sd"`: weights ∝ sqrt(N) * std(DV)
    - `"proportional"`: proportional to the original cell-type frequencies
### Step 3: value-guided cell selection
- `strategy`
    - `"FB"` (full binning, default): DV is binned into 0.1 intervals from 0.0 to 1.0; selects top-DV cells within each bin.
    - `"MTB"` (mean-threshold binning): same bins as FB, but only bins above the type mean DV are considered.
    - `"TP"` (top-pick): no binning; simply selects the highest-DV cells per type.
### Saving DV
- `write_dv` (default `False`): if `True`, writes:
    - `dv_values.csv` (per-cell DV + type)
    - `dv_summary.csv` (per-type DV summary statistics)

## Reference
Li Huang, Weikang Gong, Dongsheng Chen, scValue: value-based subsampling of large-scale single-cell transcriptomic data for machine and deep learning tasks, Briefings in Bioinformatics, Volume 26, Issue 3, May 2025, bbaf279, https://doi.org/10.1093/bib/bbaf279

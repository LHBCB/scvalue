from __future__ import annotations

import warnings
from numbers import Integral, Real
from typing import Optional, Union, List, Any

import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# -----------------------------
# Constants / options
# -----------------------------
TYPE_WEIGHTING_OPTIONS = {"inv_cv", "sqrt_n_sd", "proportional"}
STRATEGY_OPTIONS = {"FB", "MTB", "TP"}
CELL_TYPE_NUMERIC_KEY = "cell_type_numeric"
DV_KEY = "dv"


class SCValue:
    """
    Data value-based subsampling of large-scale single-cell transcriptomic (scRNA-seq) data.

    This class is instantiated with an AnnData object, which contains large scRNA-seq data 
    and cell type information. Three steps are taken to obtain a subsample (sketch) of the data:
    Step 1: Data value computation
        Computation of each cell's data values (DV) in terms of out-of-bag (OOB) estimate by 
        fitting a random forest (RF) model. Here, DV of a cell is defined as how helpful (high DV)
        or harmful (low DV) the cell is in learning the differences of cell types in RF.
    Step 2: Sketch size determinination
        DV-weighted sampling is carried out to determine the number of cells to be subsampled in 
        each cell type, aiming to enhance the cell type balanceness in the sketch. Sketch allocation
        across cell types can be DV-weighted or proportional to the original type distribution 
        (controlled by `type_weighting`).
    Step 3: Value-guided cell selection
        Three cell selection strategies are available: full binning (FB), mean-threshold binning (MTB),
        or top-pick (TP). 
    
    Parameters
    ----------
    sketch_size : Union[int, float], default=None
        The number or percentage of cells to be subsampled from AnnData.
    
    use_rep : str, default='X_pca'
        The cell representations for fitting the random forest and computing DVs.

    cell_type_key : str, default='cell_type'
        The name of the cell type column in AnnData.obs.
    
    n_trees : int, default=100
        The number of trees in the random forest.

    type_weighting : {'inv_cv', 'sqrt_n_sd', 'proportional'}, default='inv_cv'
        How to allocate sketch budget across cell types.
        1) 'inv_cv': weights proportional to mean(DV) * sqrt(N) / std(DV).
        2) 'sqrt_n_sd': weights proportional to sqrt(N) * std(DV)
           (Eq. 2 after substituting SE = SD/sqrt(N)).
        3) 'proportional': proportional to the original cell type distribution.

    strategy : {'FB', 'MTB', 'TP'}, default='FB'
        The cell selection strategies after determination of the cell number in each type.
        1)  'FB' : full binning
            For cells of each type, their DVs are divided into bins with 0.1 intervals ranging
            from 0.0 to 1.0. Within each bin, cells of the highest DVs are selected to construct
            the sketch.
        2)  'MTB' : 'mean-threshold binning'
            The same binning as FB is performed; only those bins above the DV average are considered.
        3)  'TP' : top-pick
            No binning is involved. Simply select cells with the highest DVs in each type.
    
    write_dv : bool, default=False
        Whether to write the computed DVs for all cells in the original AnnData to disk.
    
    seed : int, default=42
        To initialize a random number generator (RNG) in random forest. Ensure reproducibility.

    Attributes
    ----------
    rf_model : class
        The instantiated RandomForestSCValue class, containing n_trees and using all available CPU cores.

    dv : pandas.core.frame.DataFrame
        The pandas DataFrame holding DVs and cell types for all cells in the original AnnData.

    adata_sub : AnnData
        The sketch of the original AnnData.
    """

    class SketchWarning(UserWarning):
        pass

    def __init__(
        self,
        adata: Any,
        sketch_size: Optional[Union[int, float]] = None,
        use_rep: str = "X_pca",
        cell_type_key: str = "cell_type",
        n_trees: int = 100,
        type_weighting: str = "inv_cv",
        strategy: str = "FB",
        write_dv: bool = False,
        seed: int = 42,
    ):
        if adata is None:
            raise ValueError("Anndata object is required.")

        # --- Validation ---
        n_obs = adata.n_obs
        if sketch_size is None:
            raise ValueError("Sketch size should be specified.")
        elif isinstance(sketch_size, float):
            if not (0 < sketch_size <= 1):
                raise ValueError("Sketch percentage should be between 0 and 1.")
            self.sketch_size = int(n_obs * sketch_size)
        elif isinstance(sketch_size, int):
            self.sketch_size = min(n_obs, sketch_size)
        else:
            raise ValueError("Sketch size should be an integer or a float.")

        if use_rep != "X" and use_rep not in adata.obsm:
            raise ValueError(f"{use_rep!r} does not exist in adata.obsm.")
        if cell_type_key not in adata.obs.columns:
            raise ValueError(f"{cell_type_key!r} does not exist in adata.obs.")

        if type_weighting not in TYPE_WEIGHTING_OPTIONS:
            raise ValueError(f"type_weighting must be one of {sorted(TYPE_WEIGHTING_OPTIONS)}")
        if strategy not in STRATEGY_OPTIONS:
            raise ValueError(f"strategy must be one of {sorted(STRATEGY_OPTIONS)}")

        # --- Initialization ---
        self.adata = adata
        self.use_rep = use_rep
        self.cell_type_key = cell_type_key
        self.n_trees = n_trees
        self.strategy = strategy
        self.type_weighting = type_weighting
        self.write_dv = write_dv
        self.seed = seed

        # Initialize RF with seed directly to avoid global side-effects
        self.rf_model = self.RFValue(
            n_estimators=self.n_trees, 
            n_jobs=-1, 
            random_state=self.seed
        )
        
        self.dv: Optional[pd.DataFrame] = None
        self.adata_sub: Optional[Any] = None

    def hamilton(self, exact_alloc: pd.Series, size: int) -> pd.Series:
        """Applies the Largest Remainder Method (Hamilton method)."""
        int_alloc = np.floor(exact_alloc).astype(int)
        remaining = int(size - int_alloc.sum())
        
        if remaining > 0:
            remainders = exact_alloc - int_alloc
            indices = remainders.nlargest(remaining).index
            int_alloc.loc[indices] += 1
            
        return int_alloc

    def get_prop(self, df: pd.DataFrame, col: str, size: int) -> pd.Series:
        counts = df[col].value_counts()
        exact_alloc = counts / counts.sum() * size
        return self.hamilton(exact_alloc, size)

    def get_weighted_prop(self, df: pd.DataFrame, size: int) -> pd.Series:
        exact_alloc = df["weighted_prop"] * size
        return self.hamilton(exact_alloc, size)

    def train_rf(self) -> pd.DataFrame:
        """Fits Random Forest and calculates Data Values."""
        # 1. Prepare Data (Use local variable to avoid modifying self.adata.X in-place)
        if self.use_rep == "X":
            print("Use counts as representations for computing dv")
            X = self.adata.X
            if scipy.sparse.issparse(X):
                X = X.toarray()
        else:
            print(f"Use {self.use_rep} as representations for computing dv")
            X = self.adata.obsm[self.use_rep]

        # 2. Prepare Labels (pd.factorize is faster than manual dict comprehension)
        codes, _ = pd.factorize(self.adata.obs[self.cell_type_key], sort=True)
        self.adata.obs[CELL_TYPE_NUMERIC_KEY] = codes
        
        # 3. Fit & Compute
        self.rf_model.fit(X, codes)
        dv_values = self.rf_model.compute_rf_dv(X, codes)
        
        return pd.DataFrame(dv_values, columns=[DV_KEY], index=self.adata.obs_names)

    def get_dv_bins(self, threshold: float) -> List[float]:
        """Returns bin edges for stratification."""
        # Standardize bin generation
        base_bins = np.linspace(0.0, 1.0, 11).round(1).tolist()
        if threshold <= 0:
            return base_bins
        return [0.0, float(threshold)] + [b for b in base_bins if b > threshold]

    def reallocate(self, df: pd.DataFrame, excess: int) -> pd.DataFrame:
        """Iteratively reallocates excess sample slots to cell types that have room."""
        # Vectorized initial clamp
        df["sample_count"] = np.minimum(df["sample_count"], df["count"])
        
        while excess > 0:
            valid_mask = df["sample_count"] < df["count"]
            valid_df = df[valid_mask].copy()
            
            if valid_df.empty:
                break

            total_weight = valid_df["weighted_prop"].sum()
            # Avoid div by zero if weights vanish
            alloc_prop = valid_df["weighted_prop"] / (total_weight if total_weight > 0 else 1)
            
            # Proportional allocation of excess
            alloc = np.floor(alloc_prop * excess).astype(int)

            # Handle edge case where excess is small and spread thin (alloc becomes all 0)
            if alloc.max() == 0:
                top_idx = alloc_prop.nlargest(excess).index
                alloc.loc[top_idx] = 1
            
            # Ensure allocating no more than receivable
            receivable = valid_df["count"] - valid_df["sample_count"]
            alloc = np.minimum(alloc, receivable)
            
            # Apply update
            df.loc[valid_df.index, "sample_count"] += alloc
            excess -= int(alloc.sum())
        
        df["sample_count"] = df["sample_count"].astype(int)
        return df

    def value(self):
        warnings.filterwarnings("default", category=self.SketchWarning)
        print(f"Start: sketch {self.sketch_size}")
        
        # --- Step 1: Data value computation ---
        dv_df = self.train_rf()
        self.adata.obs[DV_KEY] = dv_df[DV_KEY]
        
        # Sort upfront to simplify Top-Pick logic later
        obs_df = self.adata.obs.copy()
        obs_df.sort_values(by=DV_KEY, ascending=False, inplace=True)
        
        info_df = obs_df.groupby(self.cell_type_key, observed=True)[DV_KEY].describe()
        
        self.dv = self.adata.obs[[self.cell_type_key, DV_KEY]].copy()
        if self.write_dv:
            self.dv.to_csv("dv_values.csv")
            info_df.to_csv("dv_summary.csv")

        # --- Step 2: Sketch size determination ---
        if self.type_weighting == "proportional":
            print("Perform proportional sampling for each cell type")
            sample_counts = self.get_prop(obs_df, self.cell_type_key, self.sketch_size)
        else:
            print(f"Perform weighted sampling ({self.type_weighting})")
            
            # Use numpy errstate for cleaner safe division
            with np.errstate(divide='ignore', invalid='ignore'):
                if self.type_weighting == "inv_cv":
                    # Matches original logic: Mean * Sqrt(N) / Std
                    cv_adj = (info_df["std"] / info_df["mean"]) * np.sqrt(info_df["count"])
                    expected = info_df["count"] / cv_adj
                elif self.type_weighting == "sqrt_n_sd":
                    # Matches paper logic: Sqrt(N) * Std
                    expected = np.sqrt(info_df["count"]) * info_df["std"]

            # Handle Infs/NaNs (Std=0 or Mean=0)
            expected = expected.replace([np.inf, -np.inf], np.nan)
            
            if self.type_weighting == "inv_cv":
                mask = (info_df["mean"] == 0) | (info_df["std"] == 0) | expected.isna()
                expected.loc[mask] = info_df.loc[mask, "count"]
            else: # sqrt_n_sd
                mask = (info_df["std"] == 0) | expected.isna()
                expected.loc[mask] = np.sqrt(info_df.loc[mask, "count"])

            info_df["expected_count"] = expected
            info_df["weighted_prop"] = expected / expected.sum()
            info_df["sample_count"] = self.get_weighted_prop(info_df, self.sketch_size)
            
            # Handle Over-allocation
            info_df["excess"] = (info_df["sample_count"] - info_df["count"]).clip(lower=0)
            excess = int(info_df["excess"].sum())
            
            if excess > 0:
                warnings.warn(
                    "Sketch size for some types exceeds original size. Reallocating...", 
                    self.SketchWarning
                )
                info_df = self.reallocate(info_df, excess)
            
            sample_counts = info_df["sample_count"]

        # --- Step 3: Value-guided cell selection ---
        sampled_indices = []
        
        grouped_obs = obs_df.groupby(self.cell_type_key, observed=True)

        if self.strategy == "TP":
            print(f"Get top dv samples for each cell type (strategy: {self.strategy})")
            for cell_type, sub_df in grouped_obs:
                target_size = sample_counts.get(cell_type, 0)
                if target_size > 0:
                    print(f"Sketch: {cell_type}\n\tSize: {target_size}")
                    sampled_indices.extend(sub_df.index[:target_size])
        else:
            print(f"Get bin-based stratified samples for each cell type (strategy: {self.strategy})")
            for cell_type, sub_df in grouped_obs:
                target_size = sample_counts.get(cell_type, 0)
                if target_size <= 0:
                    continue
                
                # Determine Bins
                if self.strategy == "FB":
                    bins = self.get_dv_bins(0)
                else:  # MTB
                    mean_val = np.round(info_df.loc[cell_type, "mean"], 1)
                    bins = self.get_dv_bins(mean_val)

                print(f"Sketch: {cell_type}\n\tSize: {target_size}\n\tBins: {bins}")

                sub_df = sub_df.copy()
                sub_df["dv_interval"] = pd.cut(sub_df[DV_KEY], bins=bins, include_lowest=True)
                
                bin_counts = self.get_prop(sub_df, "dv_interval", target_size)
                bin_counts = bin_counts[bin_counts > 0]

                if not bin_counts.empty:
                    # Select top cells within each bin
                    bin_groups = sub_df.groupby("dv_interval", observed=True)
                    for interval, b_size in bin_counts.items():
                        if interval in bin_groups.groups:
                            idx = bin_groups.get_group(interval).index[:b_size]
                            sampled_indices.extend(idx)

        print(f"Done: sketch {self.sketch_size}")
        self.adata_sub = self.adata[sampled_indices].copy()
        return self.adata_sub

    class RFValue(RandomForestClassifier):
        """
        Extended RandomForestClassifier to compute Data Value (average OOB accuracy).
        Uses deterministic random states to reconstruct OOB indices.

        References:
            [1] L. Breiman, "Random Forests", Machine Learning, 45(1), 5-32, 2001.
            [2] Y. Kwon and J. Zou, "Data-oob: Out-of-bag estimate as a simple and efficient 
                data value", International Conference on Machine Learning, 18135-18152, 2023.
        """        
        def compute_rf_dv(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
            """
            Computes Data Value for each sample based on OOB predictions.
            """
            n_samples = X.shape[0]
            # Accumulators (float for precision)
            numerator = np.zeros(n_samples, dtype=np.float64)
            denominator = np.zeros(n_samples, dtype=np.int32)

            if not hasattr(self, "estimators_samples_"):
                raise RuntimeError("estimators_samples_ is missing; was bootstrap disabled?")
    
            for tree, inbag in zip(self.estimators_, self.estimators_samples_):
                counts = np.bincount(inbag, minlength=n_samples)
                oob_mask = counts == 0
                
                if not np.any(oob_mask):
                    continue
                
                # Predict on OOB
                pred = tree.predict(X[oob_mask])
                numerator[oob_mask] += (pred == y[oob_mask])
                denominator[oob_mask] += 1

            # Compute average, handle samples that were never OOB (division by zero)
            dv = np.full(n_samples, np.nan, dtype=np.float64)
            np.divide(numerator, denominator, out=dv, where=denominator > 0)
            
            return dv

# Helper to replicate internal sklearn logic for bootstrap size
def _get_n_samples_bootstrap(n_samples, max_samples):
    if max_samples is None:
        return n_samples
    if isinstance(max_samples, Integral):
        return max_samples
    if isinstance(max_samples, Real):
        return int(n_samples * max_samples)
    return n_samples
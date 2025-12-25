import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from typing import Sequence, Optional, Tuple, Union, List
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns


class Spatial_DEG:
    """
    Standardize -> PCA -> plots & utilities.

    Notes
    -----
    - X: shape (n_samples, n_features)
    - feature_names: length n_features (gene/cluster names etc.)
    """

    def __init__(
        self,
        n_components: Optional[int] = None,   # None -> full PCA
        standardize: bool = True,
        random_state: Optional[int] = None,
    ):
        self.n_components = n_components
        self.standardize = standardize
        self.random_state = random_state

        self.scaler_: Optional[StandardScaler] = None
        self.pca_: Optional[PCA] = None

        self.X_std_: Optional[np.ndarray] = None
        self.X_pca_: Optional[np.ndarray] = None
        self.feature_names_: Optional[List[str]] = None

    # -----------------------
    # Fit
    # -----------------------
    def fit(self, X: Union[np.ndarray, pd.DataFrame], feature_names: Optional[Sequence[str]] = None):
        if isinstance(X, pd.DataFrame):
            X_mat = X.values
            if feature_names is None:
                feature_names = list(X.columns)
        else:
            X_mat = np.asarray(X)

        n_features = X_mat.shape[1]
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]
        self.feature_names_ = list(feature_names)

        # standardize
        if self.standardize:
            self.scaler_ = StandardScaler()
            self.X_std_ = self.scaler_.fit_transform(X_mat)
        else:
            self.scaler_ = None
            self.X_std_ = X_mat

        # PCA
        self.pca_ = PCA(n_components=self.n_components, random_state=self.random_state)
        self.X_pca_ = self.pca_.fit_transform(self.X_std_)

        return self

    # -----------------------
    # Properties
    # -----------------------
    @property
    def explained_variance_ratio_(self) -> np.ndarray:
        self._check_fitted()
        return self.pca_.explained_variance_ratio_

    @property
    def cumulative_variance_ratio_(self) -> np.ndarray:
        return np.cumsum(self.explained_variance_ratio_)

    @property
    def components_(self) -> np.ndarray:
        self._check_fitted()
        return self.pca_.components_

    @property
    def loadings_(self) -> np.ndarray:
        """
        loadings (a.k.a contributions in your code): shape (n_features, n_components)
        = components_.T
        """
        self._check_fitted()
        return self.pca_.components_.T

    def get_loadings_df(self, pcs: Optional[Sequence[int]] = None) -> pd.DataFrame:
        self._check_fitted()
        L = self.loadings_
        n_comp = L.shape[1]

        if pcs is None:
            pcs = list(range(n_comp))
        cols = [f"PC{i+1}" for i in pcs]
        df = pd.DataFrame(L[:, pcs], index=self.feature_names_, columns=cols)
        return df

    # -----------------------
    # Plots
    # -----------------------
    def plot_explained_variance_bar(
        self,
        max_pc: int = 10,
        figsize=(7, 4),
        color="steelblue",
        show_values: bool = True,
    ):
        """
        Bar plot of explained variance ratio for each PC (non-cumulative).
    
        Parameters
        ----------
        max_pc : int
            Number of PCs to display (e.g. 10)
        show_values : bool
            Annotate bars with values
        """
        self._check_fitted()
    
        evr = self.explained_variance_ratio_
        max_pc = min(max_pc, len(evr))
    
        x = np.arange(1, max_pc + 1)
        y = evr[:max_pc]
    
        plt.figure(figsize=figsize)
        bars = plt.bar(x, y, color=color)
    
        if show_values:
            for b, v in zip(bars, y):
                plt.text(
                    b.get_x() + b.get_width() / 2,
                    v,
                    f"{v:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )
    
        plt.xlabel("Principal Component", fontsize=14)
        plt.ylabel("Explained Variance Ratio", fontsize=14)
        plt.xticks(x)
        plt.ylim(0, max(y) * 1.15)
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()


    def plot_loading_violin(
        self,
        pcs: Sequence[int] = (0, 1, 2),
        figsize: Tuple[int, int] = (5, 6),
        cut: float = 0,
    ):
        """
        Violin plot of loadings for selected PCs.
        """
        df = self.get_loadings_df(pcs=pcs)

        melted = df.reset_index().melt(
            id_vars="index",
            var_name="PC",
            value_name="loading"
        )

        plt.figure(figsize=figsize)
        sns.violinplot(data=melted, x="PC", y="loading", cut=cut)
        plt.axhline(0, color="gray", linestyle="--", linewidth=1)
        plt.ylabel("Loading", fontsize=18)
        plt.xlabel("Principal Component", fontsize=18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.tight_layout()
        plt.show()

    def plot_signed_top_contributions(
        self,
        pc: int = 0,                # 0-based
        top_pos: int = 10,
        top_neg: int = 10,
        figsize: Tuple[int, int] = (6, 12),
        pos_color: str = "#C0392B",
        neg_color: str = "#5DADE2",
        xlabel: Optional[str] = None,
        title: Optional[str] = None,
        ytick_fontsize: int = 25,
        xtick_fontsize: int = 14,
    ):
        """
        Signed horizontal bar plot: top positive and top negative (by abs) contributors.
        """
        self._check_fitted()

        pc_contrib = self.loadings_[:, pc]  # (n_features,)
        
        # Prepare candidates: positives descending, negatives ascending (more negative first)
        pos_cand = np.where(pc_contrib > 0)[0]
        neg_cand = np.where(pc_contrib < 0)[0]
        
        pos_sorted = pos_cand[np.argsort(pc_contrib[pos_cand])[::-1]]
        neg_sorted = neg_cand[np.argsort(pc_contrib[neg_cand])]
        
        # --- first, set the desired number --
        want_pos = top_pos
        want_neg = top_neg
        
        take_pos = min(want_pos, len(pos_sorted))
        take_neg = min(want_neg, len(neg_sorted))
        
        # --- if positives are insufficient, fill with negatives ---
        if take_pos < want_pos:
            extra = want_pos - take_pos
            can_add = len(neg_sorted) - take_neg
            add = min(extra, can_add)
            take_neg += add
        
       # --- if negatives are insufficient, fill with positives (re-check after the above padding) ---
        if take_neg < want_neg:
            extra = want_neg - take_neg
            can_add = len(pos_sorted) - take_pos
            add = min(extra, can_add)
            take_pos += add
        
        # --- final choise ---
        pos_idx = pos_sorted[:take_pos]
        neg_idx = neg_sorted[:take_neg]
        
        pos_vals = pc_contrib[pos_idx]
        pos_labels = [self.feature_names_[i] for i in pos_idx]
        
        neg_vals = pc_contrib[neg_idx]
        neg_labels = [self.feature_names_[i] for i in neg_idx]
        
        values = np.concatenate([neg_vals, pos_vals])
        labels = neg_labels + pos_labels
        colors = [neg_color] * len(neg_vals) + [pos_color] * len(pos_vals)

        plt.figure(figsize=figsize, facecolor="white")
        plt.barh(range(len(values)), values, color=colors)
        plt.xticks(fontsize=xtick_fontsize)
        plt.yticks(range(len(values)), labels, fontsize=ytick_fontsize)
        plt.gca().invert_yaxis()

        if xlabel is None:
            xlabel = f"PC{pc+1} loading"
        if title is None:
            title = f"Top signed contributors to PC{pc+1}"

        plt.xlabel(xlabel, fontsize=18)
        plt.title(title, fontsize=18)
        plt.tight_layout()
        plt.show()

    def plot_signed_top_contributions_multi(
        self,
        pcs: Sequence[int] = (0, 1, 2),
        top_pos: int = 10,
        top_neg: int = 10,
        figsize: Tuple[int, int] = (6, 12),
    ):
        """
        Convenience: plot for PC1, PC2, PC3... in a loop.
        """
        for pc in pcs:
            self.plot_signed_top_contributions(
                pc=pc,
                top_pos=top_pos,
                top_neg=top_neg,
                figsize=figsize,
            )

    def get_top_genes_signed(
        self,
        pcs: Sequence[int] = (0, 1, 2),
        top_pos: int = 10,
        top_neg: int = 10,
        sort_by: str = "value",  # "value" or "abs"
    ) -> pd.DataFrame:

        self._check_fitted()
        if sort_by not in ("value", "abs"):
            raise ValueError("sort_by must be 'value' or 'abs'")
    
        rows = []
        for pc in pcs:
            v = self.loadings_[:, pc]
    
            # positive
            pos_idx = np.where(v > 0)[0]
            if pos_idx.size > 0:
                order = np.argsort(v[pos_idx])
                pos_sel = pos_idx[order][-top_pos:][::-1]
                for r, j in enumerate(pos_sel, start=1):
                    rows.append([f"PC{pc+1}", "pos", r, self.feature_names_[j], float(v[j])])
    
            # negative
            neg_idx = np.where(v < 0)[0]
            if neg_idx.size > 0:
                if sort_by == "abs":
                    order = np.argsort(np.abs(v[neg_idx]))
                else:
                    order = np.argsort(v[neg_idx])
                neg_sel = neg_idx[order][:top_neg]
                
                if sort_by == "abs":
                    neg_sel = neg_sel[::-1]
                for r, j in enumerate(neg_sel, start=1):
                    rows.append([f"PC{pc+1}", "neg", r, self.feature_names_[j], float(v[j])])
    
        return pd.DataFrame(rows, columns=["PC", "sign", "rank", "gene", "loading"])

            
    
    def fit_umap_on_loadings(
        self,
        n_components: int = 2,
        n_neighbors: int = 10,
        min_dist: float = 0.1,
        metric: str = "euclidean",
        random_state: int = 777,
    ):
        """
        Fit UMAP on PCA loadings (features = PCs, rows = genes/features).
    
        Returns
        -------
        umap_coords : pd.DataFrame
            columns = ["UMAP1", "UMAP2"] (or more if n_components>2),
            index = feature_names
        """
        self._check_fitted()
    
        import umap.umap_ as umap 
    
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
        )
    
        X = self.loadings_  # shape: (n_features, n_pcs)
        emb = reducer.fit_transform(X)
    
        cols = [f"UMAP{i+1}" for i in range(emb.shape[1])]
        umap_df = pd.DataFrame(emb, columns=cols, index=self.feature_names_)
    
        self.umap_model_ = reducer
        self.umap_loadings_ = umap_df
    
        return umap_df


    
    def plot_umap_loadings_interactive(
        self,
        umap_df: Optional[pd.DataFrame] = None,
        title: str = "UMAP of PCA Loadings (Gene Embedding)",
        width: int = 650,
        height: int = 650,
        marker_size: int = 7,
    ):
        """
        Interactive scatter (hover shows gene/feature name) using Plotly.
        """
        self._check_fitted()
    
        import plotly.express as px
    
        if umap_df is None:
            # Use it if fit_umap_on_loadings has already been called
            if hasattr(self, "umap_loadings_"):
                umap_df = self.umap_loadings_.copy()
            else:
                umap_df = self.fit_umap_on_loadings()
    
        df_plot = umap_df.reset_index().rename(columns={"index": "gene"})
    
        fig = px.scatter(
            df_plot,
            x="UMAP1",
            y="UMAP2",
            hover_name="gene",
            title=title,
        )

        fig.update_traces(marker=dict(size=marker_size, opacity=0.85))
        fig.update_layout(width=width, height=height, plot_bgcolor="white")
        fig.update_xaxes(showline=True, linecolor="black", tickfont=dict(size=16), title_font=dict(size=18), linewidth=1, mirror=True)
        fig.update_yaxes(showline=True, linecolor="black", tickfont=dict(size=16), title_font=dict(size=18), linewidth=1, mirror=True)
        fig.show()
    
        #return fig


    # -----------------------
    # internal
    # -----------------------
    def _check_fitted(self):
        if self.pca_ is None or self.feature_names_ is None:
            raise RuntimeError("PCAReport is not fitted. Call .fit(X) first.")

            

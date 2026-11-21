"""
Save / load the preprocessed grid (stlearn grid + SKNY ``calculate_distance``).

Building the grid and running SKNY take a few minutes for a whole Xenium
sample, and both only depend on the input data, so they can be computed once
and reloaded::

    grid = calculate_distance(grid, pos_marker_ls=["CDH1", "EPCAM"])
    save_preprocessed(grid, "grid_skny.h5ad")
    ...
    grid = load_preprocessed("grid_skny.h5ad")   # grid.shortest restored too

The round trip is exact: ``grid.X``, ``obs`` / ``var`` (values and dtypes,
including interval columns such as ``region`` and object columns such as
``region_10``), ``uns`` images, ``obsm`` and ``grid.shortest`` come back
identical.
"""
from __future__ import annotations

import json
from typing import Any, Dict

import anndata as ad
import numpy as np
import pandas as pd

_META = "_scomv_io"


def _split_frame(df: pd.DataFrame, prefix: str, store: Dict[str, Any], meta: Dict[str, Any]) -> pd.DataFrame:
    """Move columns that h5ad cannot round-trip exactly into ``store``; return the rest."""
    keep = {}
    cols_meta = []
    for col in df.columns:
        s = df[col]
        key = f"{prefix}{len(cols_meta)}"
        if isinstance(s.dtype, pd.IntervalDtype):
            arr = s.array
            store[key + "_left"] = np.asarray(arr.left)
            store[key + "_right"] = np.asarray(arr.right)
            store[key + "_na"] = np.asarray(s.isna())
            cols_meta.append({"name": col, "kind": "interval", "key": key, "closed": arr.closed,
                              "subtype": str(s.dtype.subtype)})
        elif isinstance(s.dtype, pd.CategoricalDtype) and isinstance(s.cat.categories, pd.IntervalIndex):
            cats = s.cat.categories
            store[key + "_codes"] = np.asarray(s.cat.codes)
            store[key + "_left"] = np.asarray(cats.left)
            store[key + "_right"] = np.asarray(cats.right)
            cols_meta.append({"name": col, "kind": "interval_categorical", "key": key, "closed": cats.closed,
                              "ordered": bool(s.cat.ordered)})
        elif s.dtype == object:
            na = np.asarray(s.isna())
            store[key + "_values"] = np.asarray(["" if m else str(v) for v, m in zip(s, na)], dtype=object)
            store[key + "_na"] = na
            cols_meta.append({"name": col, "kind": "object", "key": key})
        else:
            keep[col] = s
            cols_meta.append({"name": col, "kind": "native"})
    meta[prefix] = cols_meta
    return pd.DataFrame(keep, index=df.index)


def _restore_frame(df_native: pd.DataFrame, prefix: str, store: Dict[str, Any], meta: Dict[str, Any], index) -> pd.DataFrame:
    out = {}
    for c in meta[prefix]:
        name, kind = c["name"], c["kind"]
        if kind == "native":
            out[name] = df_native[name].to_numpy()
        elif kind == "interval":
            left = np.asarray(store[c["key"] + "_left"])
            right = np.asarray(store[c["key"] + "_right"])
            arr = pd.arrays.IntervalArray.from_arrays(left, right, closed=c["closed"],
                                                      dtype=pd.IntervalDtype(c["subtype"], c["closed"]))
            na = np.asarray(store[c["key"] + "_na"], dtype=bool)
            if na.any():
                arr[na] = np.nan
            out[name] = arr
        elif kind == "interval_categorical":
            cats = pd.IntervalIndex.from_arrays(np.asarray(store[c["key"] + "_left"]),
                                                np.asarray(store[c["key"] + "_right"]), closed=c["closed"])
            out[name] = pd.Categorical.from_codes(np.asarray(store[c["key"] + "_codes"]), categories=cats,
                                                  ordered=c["ordered"])
        elif kind == "object":
            vals = np.asarray(store[c["key"] + "_values"], dtype=object)
            na = np.asarray(store[c["key"] + "_na"], dtype=bool)
            vals = vals.copy()
            vals[na] = np.nan
            out[name] = vals
    return pd.DataFrame(out, index=index)


def save_preprocessed(grid, path: str, compression: str | None = "lzf") -> None:
    """
    Save a grid AnnData after ``calculate_distance`` (and optionally after
    attaching ``region`` / ``region_10``), including ``grid.shortest``.
    """
    store: Dict[str, Any] = {}
    meta: Dict[str, Any] = {"version": 1}

    obs_native = _split_frame(grid.obs, "obs", store, meta)
    var_native = _split_frame(grid.var, "var", store, meta)

    uns = {}
    none_keys = []
    for k, v in grid.uns.items():
        if v is None:
            none_keys.append(k)
        else:
            uns[k] = v
    meta["uns_none"] = none_keys

    shortest = getattr(grid, "shortest", None)
    meta["has_shortest"] = shortest is not None
    if shortest is not None:
        idx = list(shortest.index)
        store["sh_index_x"] = np.asarray([t[0] for t in idx], dtype=np.int64)
        store["sh_index_y"] = np.asarray([t[1] for t in idx], dtype=np.int64)
        meta["sh_columns"] = list(shortest.columns)
        sh_native = _split_frame(shortest, "sh", store, meta)
        for col in sh_native.columns:
            store[f"sh_native_{col}"] = sh_native[col].to_numpy()

    uns[_META] = {"meta": json.dumps(meta), **store}
    out = ad.AnnData(X=grid.X, obs=obs_native, var=var_native, uns=uns,
                     obsm=dict(grid.obsm), varm=dict(grid.varm), layers=dict(grid.layers))
    out.write_h5ad(path, compression=compression)


def load_preprocessed(path: str):
    """Load a grid saved with :func:`save_preprocessed` (restores ``grid.shortest``)."""
    a = ad.read_h5ad(path)
    blob = a.uns.pop(_META)
    meta = json.loads(blob["meta"])
    store = {k: v for k, v in blob.items() if k != "meta"}

    obs = _restore_frame(a.obs, "obs", store, meta, a.obs.index)
    var = _restore_frame(a.var, "var", store, meta, a.var.index)
    uns = dict(a.uns)
    for k in meta["uns_none"]:
        uns[k] = None

    grid = ad.AnnData(X=a.X, obs=obs, var=var, uns=uns, obsm=dict(a.obsm), varm=dict(a.varm),
                      layers=dict(a.layers))

    if meta["has_shortest"]:
        xs = np.asarray(store["sh_index_x"]).tolist()
        ys = np.asarray(store["sh_index_y"]).tolist()
        index = pd.Index(list(zip(xs, ys)), tupleize_cols=False)
        sh_native = pd.DataFrame({c["name"]: np.asarray(store[f"sh_native_{c['name']}"])
                                  for c in meta["sh"] if c["kind"] == "native"}, index=index)
        sh = _restore_frame(sh_native, "sh", store, meta, index)
        setattr(grid, "shortest", sh[meta["sh_columns"]])
    return grid

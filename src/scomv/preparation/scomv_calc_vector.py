import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

def compute_min_vectors_polar(
    xy_list: List[Tuple[int, int]],
    outline_points: List[Tuple[int, int]],
    inside_points: Optional[List[Tuple[int, int]]] = None,
    invert_y: bool = True,
    make_inside_negative: bool = True,
    atol: float = 1e-8,
) -> pd.DataFrame:
    """
    For each point in xy_list, find all shortest vectors to outline_points
    (ties allowed), convert to polar (angle, radii). Optionally make radii
    negative for inside_points.

    Returns
    -------
    df : pd.DataFrame with index=xy_list, columns=["angle","radii"]
         each cell is a list (ties -> multiple angles/radii)
    """

    # --- arrays ---
    P = np.asarray(xy_list, dtype=float)          # (N,2) [std_x, std_y]
    O = np.asarray(outline_points, dtype=float)   # (M,2) [x_location, y_location]

    if P.ndim != 2 or P.shape[1] != 2:
        raise ValueError("xy_list must be list of (x,y)")
    if O.ndim != 2 or O.shape[1] != 2:
        raise ValueError("outline_points must be list of (x,y)")
    if len(O) == 0:
        raise ValueError("outline_points is empty")

    # --- compute squared distances: (N,M) ---
    # dx = std_x - x_location, dy = std_y - y_location
    dx = P[:, 0:1] - O[None, :, 0]   # (N,M)
    dy = P[:, 1:2] - O[None, :, 1]   # (N,M)
    d2 = dx*dx + dy*dy               # squared distance

    # --- per point minimal distance ---
    min_d2 = d2.min(axis=1, keepdims=True)        # (N,1)
    tie_mask = np.isclose(d2, min_d2, atol=atol)  # (N,M) ties allowed

    all_angles: List[List[float]] = []
    all_radiis: List[List[float]] = []

    for i in range(P.shape[0]):
        # vectors from tied outline points
        dx_i = dx[i, tie_mask[i]]
        dy_i = dy[i, tie_mask[i]]

        if invert_y:
            dy_i = -dy_i  # # Handle y-axis inversion for Xenium coordinates

        angles_i = np.arctan2(dy_i, dx_i).tolist()
        radii_i = np.sqrt(dx_i*dx_i + dy_i*dy_i).tolist()

        all_angles.append(angles_i)
        all_radiis.append(radii_i)

    df = pd.DataFrame({"angle": all_angles, "radii": all_radiis}, index=xy_list)

    # --- make inside points negative radii ---
    if make_inside_negative and inside_points is not None and len(inside_points) > 0:
        inside_set = set(map(tuple, inside_points))
        mask = df.index.map(lambda t: t in inside_set)
        df.loc[mask, "radii"] = df.loc[mask, "radii"].apply(lambda rs: [-r for r in rs])

    return df

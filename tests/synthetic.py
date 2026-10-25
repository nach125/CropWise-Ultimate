"""Synthetic crop-like data FOR TESTS ONLY (not real measurements, not shipped as a dataset)."""
import numpy as np, pandas as pd

def make_crop_df(per_class=60, seed=0):
    rng = np.random.default_rng(seed)
    # name: (N, P, K, temp, humidity, ph, rainfall)
    centers = {"rice": (80, 48, 40, 24, 82, 6.4, 230), "maize": (78, 48, 20, 22, 65, 6.2, 85),
               "chickpea": (40, 68, 80, 19, 17, 7.3, 80), "apple": (21, 134, 200, 22, 92, 5.9, 113),
               "coffee": (101, 28, 30, 25, 58, 6.8, 158), "lentil": (19, 68, 19, 24, 64, 6.9, 46)}
    sd = (8, 8, 8, 2, 5, .3, 20)
    rows = []
    for crop, c in centers.items():
        for _ in range(per_class):
            rows.append(dict(zip(["N","P","K","temperature","humidity","ph","rainfall"], rng.normal(c, sd)), label=crop))
    return pd.DataFrame(rows).sample(frac=1, random_state=1).reset_index(drop=True)

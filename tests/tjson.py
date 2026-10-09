"""Decoder of the typed JSON written by the R reference dumps."""

import gzip
import json

import numpy as np


def tdec(o):
    t = o["t"]
    if t == "null":
        return None
    names = o.get("n")
    if t == "list":
        items = [tdec(x) for x in o["d"]]
        if names is not None and all(names):
            return dict(zip(names, items))
        return items
    d = o["d"]
    if t == "lgl":
        a = np.array([bool(x) for x in d], dtype=bool)
    elif t == "int":
        a = np.array(d, dtype=np.int64)
    elif t == "dbl":
        a = np.array([float(x) if x not in ("Inf", "-Inf", "NaN") else float(x.lower()) for x in d],
                     dtype=float)
    elif t == "chr":
        a = list(d)
    else:
        raise ValueError(t)
    dim = o.get("dim")
    if dim is not None:
        return np.asarray(a).reshape([int(x) for x in dim], order="F")
    if t == "chr":
        return a[0] if len(a) == 1 else a
    if a.size == 1:
        return a.item()
    return a


def tload(path):
    with gzip.open(path, "rt") as f:
        return tdec(json.load(f))

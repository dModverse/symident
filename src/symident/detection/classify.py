"""Classification of directions, the report on physical coordinates and scaling peeling."""

import numpy as np

from .. import _core
from ..gfp import in_span, rref_modp
from ..rexpr import get_symbols, sym_sort
from ..sym.direction import classify_generator
from .closedform import eval_modq
from .rank import nullspace_basis


def _is_int(s):
    try:
        int(str(s))
        return True
    except ValueError:
        return False


def classify_direction(d):
    """Scaling or general on the canonical poly-primitive generator; updates type, vector, degree."""
    if d.get("vector") is None:
        return d
    if d.get("type") == "scaling":
        if not all(_is_int(v) for v in d["vector"].values()):
            return d
        lit = {k: f"{v}*{k}" for k, v in d["vector"].items()}
    else:
        lit = {k: str(v) for k, v in d["vector"].items()}
    try:
        cls = classify_generator(lit)
    except Exception:
        cls = None
    if cls is None or cls.get("type") is None:
        return d
    d = dict(d)
    d["type"] = str(cls["type"])
    d["degree"] = int(cls["degree"]) if cls.get("degree") is not None else None
    if d["type"] == "scaling":
        d["vector"] = {k: str(w) for k, w in cls["weights"].items()}
        d["support"] = sym_sort(list(cls["weights"]))
    else:
        d["vector"] = {k: str(x) for k, x in cls["components"].items()}
        d["support"] = sym_sort(list(cls["components"]))
    return d


def relabel_directions(non_id):
    """The directions `non_id`, each one with a vector reclassified as scaling or general."""
    if not non_id:
        return non_id
    return [d if d.get("vector") is None else classify_direction(d) for d in non_id]


def report_physical(result, znames, aux_names, sc, nz, P, aux_field, flag_joint=False):
    """Report on the physical coordinates when the kernel also ran over auxiliary ones."""
    aux = set(aux_names)
    phys_coords = [z for z in znames if z not in aux]
    phys_set = set(phys_coords)
    result = dict(result)
    result["non_identifiable"] = [d for d in result["non_identifiable"]
                                  if any(s in phys_set for s in d.get("support", []))]
    phys_cols = [i for i, z in enumerate(znames) if z in phys_set]
    piv = {int(x) for x in np.atleast_1d(sc["pivots"])}
    N = nullspace_basis(sc["ref"], [c for c in range(nz) if c not in piv], P)
    phys_null = rref_modp(N[phys_cols, :], P)["rank"] if phys_cols and N.shape[1] else 0
    if sc.get("rank_s") is not None and sc["rank_s"] < sc["rank"]:
        aux_cols = [i for i, z in enumerate(znames) if z not in phys_set]
        r_aux = int(_core.sym_series_rank(np.asarray(sc["ref"]["S"], dtype=np.int64), int(nz),
                                          int(sc["ref"]["N"]), P, aux_cols)["rank"]) if aux_cols else 0
        phys_null = (nz - sc["rank_s"]) - (len(aux_cols) - r_aux)
    result["dim"] = len(phys_coords)
    result["coordinates"] = phys_coords
    result["rank"] = int(len(phys_coords) - phys_null)
    result["identifiable"] = phys_null == 0
    out = []
    for d in result["non_identifiable"]:
        d = dict(d)
        d["support"] = sym_sort([s for s in d["support"] if s in phys_set])
        if d.get("vector") is not None:
            d[aux_field] = {k: v for k, v in d["vector"].items() if k in aux}
            phys = {k: v for k, v in d["vector"].items() if k not in aux}
            if flag_joint and any(set(get_symbols(str(e))) & aux for e in phys.values()):
                d["joint_form"] = True
            d["vector"] = phys
        out.append(d)
    result["non_identifiable"] = out
    return result


def peel_scalings(scal_res, znames, nz, zval, P, N):
    """Scalings common to all conditions whose tangent lies in the nullspace, independent."""
    scaling = []
    Bmat = np.zeros((nz, 0), dtype=np.int64)
    env = {z: int(v) for z, v in zip(znames, zval)}
    zi = {z: i for i, z in enumerate(znames)}

    def eval_w(w):
        try:
            wi = float(str(w))
            if wi == round(wi):
                return int(wi) % P
        except ValueError:
            pass
        return eval_modq(w, env, P)
    for d in scal_res.get("non_identifiable", []):
        names = [k for k in d["vector"] if k in zi]
        if not names:
            continue
        wsym = [str(d["vector"][k]) for k in names]
        wv = [eval_w(w) for w in wsym]
        if any(w is None for w in wv):
            continue
        v = np.zeros(nz, dtype=np.int64)
        for k, w in zip(names, wv):
            v[zi[k]] = w % P
        if not np.any(v):
            continue
        tangent = (v * (np.asarray(zval, dtype=np.int64) % P)) % P
        if not np.any(tangent) or not in_span(N, tangent, nz, P) or in_span(Bmat, tangent, nz, P):
            continue
        Bmat = np.concatenate([Bmat, tangent.reshape(nz, 1)], axis=1)
        order = sorted(range(len(names)), key=lambda i: names[i].encode())
        supp = [names[i] for i in order]
        scaling.append({"support": supp, "vector": {names[i]: wsym[i] for i in order},
                        "type": "scaling", "closed_form": True})
    return {"scaling": scaling, "Bmat": Bmat}


def joint_expand_scal(scal_res, state_base, Kc, held_param_of=None, pinned=()):
    """A scaling expanded onto the per-condition state columns and held pivots."""
    held_param_of = held_param_of or {}
    pinned = set(pinned)
    state_base = set(state_base)
    out = []
    for d in scal_res.get("non_identifiable", []):
        vec = {}
        for nm, w in d["vector"].items():
            if nm in held_param_of:
                vec[held_param_of[nm]] = w
            elif nm in state_base:
                for m in range(1, Kc + 1):
                    wn = f"{nm}|c{m}"
                    if wn not in pinned:
                        vec[wn] = w
            else:
                vec[nm] = w
        d = dict(d)
        d["vector"] = vec
        d["support"] = list(vec)
        out.append(d)
    scal_res = dict(scal_res)
    scal_res["non_identifiable"] = out
    return scal_res

"""Translation groups of leaves that move a direction only jointly."""

import numpy as np

from ..gfp import invmod
from .sparse import _ints


def translation_groups(rel_s, ref_s, res_s, kb_s, sc, P, n_leaves, first=None):
    """Leaves entering a direction only through their sum (add) or product (mul)."""
    if first is None:
        first = sc["pool_next"] + 7919
    base = res_s(ref_s, P)
    if base is None:
        return []
    piv0 = _ints(ref_s["pivots"])
    relevant = []
    for li, r in enumerate(rel_s):
        rp = r["rp"] if r is not None else None
        if rp is None or not rp.get("ok") or _ints(rp["pivots"]) != piv0:
            continue
        v = res_s(rp, P)
        if v is not None and np.any(v != base):
            relevant.append(li)
    if len(relevant) < 2:
        return []
    z1 = np.array(sc["point0"], dtype=np.int64)
    z1[:n_leaves] = sc["pool"](first + 1 + np.arange(n_leaves))
    tval = int(sc["pool"](first + n_leaves + 1)[0]) % 100003
    aval = int(sc["pool"](first + n_leaves + 2)[0]) % 99991 + 2
    ainv = invmod(aval, P)

    def add(z, r, v):
        z = z.copy()
        z[r] += tval
        z[v] -= tval
        return z

    def mul(z, r, v):
        z = z.copy()
        z[r] = z[r] % P * aval % P
        z[v] = z[v] % P * ainv % P
        return z
    moves = {"add": add, "mul": mul}
    groups = []
    for type_, mv in moves.items():
        grouped = {x for g in groups for x in [g["rep"]] + g["members"]}
        reps, members = [], []
        for v in [x for x in relevant if x not in grouped]:
            joined = False
            if reps:
                pts = [z1] + [mv(z1, r, v) for r in reps]
                out = kb_s(pts, [P] * len(pts), sc["nt_used"])
                v1 = res_s(out[0], P)
                if v1 is not None:
                    for j in range(len(reps)):
                        v2 = res_s(out[j + 1], P)
                        if v2 is not None and _ints(out[j + 1]["pivots"]) == _ints(out[0]["pivots"]) \
                                and np.array_equal(v1, v2):
                            members[j].append(v)
                            joined = True
                            break
            if not joined:
                reps.append(v)
                members.append([])
        for j in range(len(reps)):
            if members[j]:
                groups.append({"rep": reps[j], "members": members[j], "type": type_})
    return groups


def shift_groups(vec, groups, leaf_names, point0):
    """Entries fitted with group members held at the base point, mapped back to the groups."""
    import sympy as spy
    taken = set(vec) | set(leaf_names)
    U = []
    for k in range(len(groups)):
        u = f"si_g{k + 1}"
        while u in taken:
            u += "_"
        U.append(u)
    syms = list(dict.fromkeys(list(vec) + list(leaf_names) + U))
    loc = {s: spy.Symbol(s) for s in syms}
    try:
        out = {}
        for nm, s in vec.items():
            e = spy.sympify(str(s).replace("^", "**"), locals=loc)
            shift = []
            for k, g in enumerate(groups):
                b = [spy.Integer(int(point0[m])) for m in g["members"]]
                if g["type"] == "mul":
                    prod = spy.Integer(1)
                    for x in b:
                        prod *= x
                    to = spy.Symbol(U[k]) / prod
                else:
                    to = spy.Symbol(U[k]) - sum(b)
                shift.append((spy.Symbol(leaf_names[g["rep"]]), to))
            e = spy.factor(spy.cancel(e.subs(shift)))
            back = []
            for k, g in enumerate(groups):
                zs = [spy.Symbol(leaf_names[x]) for x in [g["rep"]] + g["members"]]
                if g["type"] == "mul":
                    tot = spy.Integer(1)
                    for x in zs:
                        tot *= x
                else:
                    tot = sum(zs)
                back.append((spy.Symbol(U[k]), tot))
            out[nm] = str(e.subs(back)).replace("**", "^")
        return out
    except Exception:
        return None

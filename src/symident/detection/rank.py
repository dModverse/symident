"""Saturation of the Lie and gap orders at a generic base point, with cross-prime checks."""

import time

import numpy as np

from .. import state
from ..gfp import PRIMES, Pool, null_residues, random_point
from .fields import rank_of, rank_score


def _ok(r):
    return r is not None and bool(r.get("ok"))


def saturate_certify(kcall, n_leaves, nz, max_m=0, warm=None, probe_block=1, block_call=None,
                     budget=None, block_map=None, nt_min=0, nt_min_block=None, prep=None,
                     rank_cap=None, block_profile=None, set_caps=None, checks=True):
    """Raise the Lie order (and the gap order up to max_m) until the rank saturates.

    kcall(point, p, Nt, Mtot) returns a kernel result; block_call holds one such call per
    block, where the Lie order is decided. Without `checks` the rank is not raised at
    other primes and points. Returns None without a usable point."""
    if warm is None:
        def warm(pts, primes):
            return None
    if set_caps is None:
        def set_caps(orders, point, Mtot, first):
            return None
    if rank_cap is None:
        rank_cap = nz
    block_call = block_call or []
    nt_min = int(nt_min) if nt_min is not None else 0
    P = PRIMES[0]
    pool = Pool()
    point0 = pool(np.arange(1, n_leaves + 1))
    pool_next = n_leaves + 1

    plateau_need = max(1, int(state.env_value("LIEPLATEAU", "3")))
    bo = state.env_value("LIEPLATEAU_BLOCK")
    try:
        block_override = int(bo) if bo is not None else None
    except ValueError:
        block_override = None
    lie_diag = state.env("LIEDIAG")
    timing = state.env("TIMING")
    per_block = len(block_call) > 1 and (block_override is None or block_override > 0)
    use_budget = per_block or len(block_call) == 1
    if block_override is not None and block_override > 0:
        need_used = block_override
    elif not use_budget or budget is None:
        need_used = plateau_need
    else:
        need_used = min(plateau_need, budget + 1)
    certified = use_budget and budget is not None and need_used >= budget + 1

    def scan_nt(call1, point, Mtot, frm, need, label, lo0=0, prof1=None):
        lo0 = int(lo0)
        if prof1 is not None and Mtot == 0 and not _ok(prof1(point, P, 1)):
            # a chain with a moving boundary has no profile; the orders are scanned
            prof1 = None
        if prof1 is not None and Mtot == 0:
            nt_end = nz + 2 + lo0
            nt_try = min(nt_end, max(16, frm + need + 1, lo0 + need + 2))
            while True:
                pr = prof1(point, P, nt_try)
                if not _ok(pr):
                    return None
                rk = [int(x) for x in np.atleast_1d(pr["ranks"])]
                Nt = max(1, frm)
                prev, flat, grew, stop = -1, 0, Nt, False
                while Nt <= nt_try:
                    if rk[Nt] == prev:
                        if Nt > lo0:
                            flat += 1
                    else:
                        grew = Nt
                    if rk[Nt] >= nz or (flat >= need and Nt >= 2) or Nt > nz + 1 + lo0:
                        stop = True
                        break
                    prev = rk[Nt]
                    Nt += 1
                if stop:
                    break
                nt_try = min(nt_end, 2 * nt_try)
            res = call1(point, P, Nt, Mtot)
            if not _ok(res):
                return None
            if lie_diag:
                print(f"[liediag] {label} Mtot={Mtot} ranks from Lie order {max(1, frm)}: "
                      f"{','.join(str(x) for x in rk[max(1, frm):Nt + 1])} (nz={nz}, profile to {nt_try})")
            return {"res": res, "Nt": Nt, "grew": grew, "ranks": rk}
        if frm > 1 and frm > lo0:
            lo = call1(point, P, frm, Mtot)
            if not _ok(lo):
                return None
            hi = call1(point, P, frm + need, Mtot)
            if not _ok(hi):
                return None
            if rank_score(lo) == rank_score(hi):
                if lie_diag:
                    print(f"[liediag] {label} Mtot={Mtot} rank {lo['rank']} flat over Lie order "
                          f"{frm}-{frm + need} (nz={nz})")
                return {"res": hi, "Nt": frm + need, "grew": frm}
        if Mtot > 0 and frm > 1 and not state.env("NOGALLOP"):
            memo = {}

            def at(o):
                if o not in memo:
                    memo[o] = call1(point, P, o, Mtot)
                return memo[o]

            def sc1(r):
                return rank_score(r) if _ok(r) else None
            cap = nz + 1 + lo0
            x, step = max(1, frm), need
            while True:
                a, b = at(x), at(min(cap, x + need))
                if sc1(a) is None or sc1(b) is None:
                    return None
                if sc1(a) == sc1(b) or rank_of(b) >= nz or x + need >= cap:
                    break
                x = min(cap, x + step)
                step *= 2
            top = sc1(b)
            hi, lo = min(cap, x + need), max(1, frm)
            while lo < hi:
                mid = (lo + hi) // 2
                if sc1(at(mid)) == top:
                    hi = mid
                else:
                    lo = mid + 1
            grew = hi
            Nt = min(cap, grew + need)
            if lie_diag:
                print(f"[liediag] {label} Mtot={Mtot} galloped from {frm} to order {grew} in "
                      f"{len(memo)} calls (nz={nz})")
            return {"res": at(Nt), "Nt": Nt, "grew": grew}
        Nt = max(1, frm)
        prev, flat, grew = -1, 0, Nt
        res, ranks = None, []
        while True:
            r = call1(point, P, Nt, Mtot)
            if not _ok(r):
                return None
            res = r
            ranks.append(int(rank_of(r)))
            if rank_score(r) == prev:
                if Nt > lo0:
                    flat += 1
            else:
                grew = Nt
            if rank_of(r) >= nz or (flat >= need and Nt >= 2) or Nt > nz + 1 + lo0:
                break
            prev = rank_score(r)
            Nt += 1
        if lie_diag:
            print(f"[liediag] {label} Mtot={Mtot} ranks from Lie order {max(1, frm)}: "
                  f"{','.join(map(str, ranks))} (nz={nz})")
        return {"res": res, "Nt": Nt, "grew": grew}

    def lo_of(bi):
        return nt_min if nt_min_block is None else int(nt_min_block[bi])

    def prof_of(bi):
        return block_profile[bi] if block_profile is not None and len(block_profile) == len(block_call) else None

    def saturate_nt(point, Mtot, from_orders=None):
        if prep is not None:
            point = prep(point)
            if point is None:
                return None
        if not per_block:
            sb = scan_nt(kcall, point, Mtot, 1, need_used, "stacked", nt_min)
            if sb is not None:
                sb["point"] = point
            return sb
        orders = [0] * len(block_call)
        Nt, driver, block_ranks = 1, 0, None
        t_scan = time.monotonic()
        if block_map is not None:
            def one(bi):
                known = from_orders is not None and len(from_orders) == len(block_call)
                frm = max(1, int(from_orders[bi])) if known else 1
                return scan_nt(block_call[bi], point, Mtot, frm, need_used, f"block {bi + 1}",
                               lo_of(bi), prof_of(bi))
            sbs = block_map(one, list(range(len(block_call))))
            if any(s is None for s in sbs):
                return None
            orders = [int(s["grew"]) for s in sbs]
            Nt = max([1] + orders)
            driver = int(np.argmax(orders))
            block_ranks = [s.get("ranks") for s in sbs]
        else:
            for bi in range(len(block_call)):
                sb = scan_nt(block_call[bi], point, Mtot, Nt, need_used, f"block {bi + 1}",
                            lo_of(bi), prof_of(bi))
                if sb is None:
                    return None
                orders[bi] = sb["grew"]
                if sb["grew"] > Nt:
                    Nt, driver = sb["grew"], bi
        first = Mtot == 0 and from_orders is None
        t_caps = time.monotonic()
        set_caps(orders, point, Mtot, first)
        t_call = time.monotonic()
        r = kcall(point, P, Nt, Mtot)
        if timing:
            print(f"[sat] Mtot {Mtot}: scans {t_caps - t_scan:.1f}s, caps {t_call - t_caps:.1f}s, "
                  f"call {time.monotonic() - t_call:.1f}s; orders {min(orders)}-{max(orders)}")
        if not _ok(r):
            return None
        if lie_diag:
            print(f"[liediag] block Lie orders: {','.join(map(str, orders))} -> stacked order {Nt}, rank {r['rank']}")
        return {"res": r, "Nt": Nt, "block_orders": orders, "block_driver": driver, "point": point,
                "block_ranks": block_ranks}

    hint = state.get("order_hint")
    using_hint = False
    if hint is not None and block_call and len(hint.get("block_orders") or []) == len(block_call):
        using_hint = True

        def saturate_nt(point, Mtot, from_orders=None):  # noqa: F811
            if prep is not None:
                point = prep(point)
                if point is None:
                    return None
            Nt = max(1, int(hint["nt_used"]))
            r = kcall(point, P, Nt, Mtot)
            if not _ok(r):
                return None
            bo_ = list(hint["block_orders"])
            return {"res": r, "Nt": Nt, "block_orders": bo_, "block_driver": int(np.argmax(bo_)),
                    "point": point}

    sat = None
    for attempt in range(1, 51):
        if probe_block > 1 and attempt >= 2 and (attempt - 2) % probe_block == 0:
            pts = [point0 if k == 0 else pool(pool_next + (k - 1) * n_leaves + np.arange(n_leaves))
                   for k in range(probe_block)]
            warm(pts, [P] * probe_block)
        sat = saturate_nt(point0, 0)
        if sat is not None:
            point0 = sat["point"]
            break
        point0 = pool(pool_next + np.arange(n_leaves))
        pool_next += n_leaves
    if sat is None:
        return None
    if using_hint and hint.get("sat_score") is not None and hint.get("Mtot") is not None:
        r = kcall(point0, P, sat["Nt"], hint["Mtot"]) if hint["Mtot"] > 0 else sat["res"]
        if timing:
            print(f"[sat] hint score {rank_score(r) if _ok(r) else '-'} of {hint['sat_score']}")
        if _ok(r) and rank_score(r) >= hint["sat_score"]:
            if sat.get("block_orders"):
                set_caps(sat["block_orders"], point0, hint["Mtot"], None)
            return {"ref": r, "nt_used": sat["Nt"], "mtot_used": hint["Mtot"], "saturated_m": True,
                    "point0": point0, "pool": pool, "pool_next": pool_next,
                    "block_orders": sat.get("block_orders"), "block_driver": sat.get("block_driver"),
                    "block_ranks": sat.get("block_ranks"), "budget": budget, "plateau": need_used,
                    "certified": certified, "rank": r["rank"], "rank_s": rank_of(r),
                    "pivots": r["pivots"]}
    sat_t0 = time.monotonic()

    def sat_log(what):
        if timing:
            print(f"[sat {time.monotonic() - sat_t0:7.1f}s] {what}")
    sat_log(f"scanned, Lie order {sat['Nt']}")
    nt_used, mtot_used, saturated_m = sat["Nt"], 0, True

    if max_m > 0:
        while True:
            if rank_of(sat["res"]) >= rank_cap:
                break
            if mtot_used >= max_m:
                saturated_m = False
                break
            sat_m = saturate_nt(point0, mtot_used + 1, sat.get("block_orders"))
            if sat_m is None:
                break
            sat_log(f"gap order {mtot_used + 1}: Lie order {sat_m['Nt']}, score "
                   f"{rank_score(sat_m['res'])} (was {rank_score(sat['res'])})")
            mtot_used += 1
            if rank_score(sat_m["res"]) <= rank_score(sat["res"]):
                break
            sat = sat_m
            nt_used = sat["Nt"]

    if sat.get("block_orders"):
        set_caps(sat["block_orders"], point0, mtot_used, None)
    while rank_of(sat["res"]) < min(sat["res"]["rank"], rank_cap) and mtot_used < max_m:
        r = kcall(point0, P, nt_used, mtot_used + 1)
        if not _ok(r):
            break
        mtot_used += 1
        sat["res"] = r

    sat_log("gap order")
    if checks:
        if prep is None:
            warm([point0] * (len(PRIMES) - 1), list(PRIMES[1:]))
        rank_max = rank_score(sat["res"])
        for pj in PRIMES[1:]:
            ptj = point0 if prep is None else prep(point0, pj)
            if ptj is None:
                continue
            rj = kcall(ptj, pj, nt_used, mtot_used)
            if _ok(rj):
                rank_max = max(rank_max, rank_score(rj))
        rand_pts = [random_point(n_leaves, 1)]
        if prep is not None:
            rand_pts = [(prep(rp) if prep(rp) is not None else rp) for rp in rand_pts]
        warm(rand_pts, [P] * len(rand_pts))
        for rp in rand_pts:
            try:
                rj = kcall(rp, P, nt_used, mtot_used)
            except Exception:
                rj = None
            if _ok(rj):
                rank_max = max(rank_max, rank_score(rj))
        sat_log("cross-prime and uniform-point checks")
        tries = 0
        while rank_score(sat["res"]) < rank_max:
            tries += 1
            if tries <= 20:
                point0 = pool(pool_next + np.arange(n_leaves))
                pool_next += n_leaves
            else:
                point0 = rand_pts[0]
            sat = saturate_nt(point0, mtot_used, sat.get("block_orders"))
            if sat is None:
                return None
            point0 = sat["point"]
            nt_used = sat["Nt"]
            for rp in rand_pts:
                try:
                    rj = kcall(rp, P, nt_used, mtot_used)
                except Exception:
                    rj = None
                if _ok(rj):
                    rank_max = max(rank_max, rank_score(rj))
            if tries > 22:
                break
    return {"ref": sat["res"], "nt_used": nt_used, "mtot_used": mtot_used, "saturated_m": saturated_m,
            "point0": point0, "pool": pool, "pool_next": pool_next,
            "block_orders": sat.get("block_orders"), "block_driver": sat.get("block_driver"),
            "block_ranks": sat.get("block_ranks"), "budget": budget, "plateau": need_used,
            "certified": certified, "rank": sat["res"]["rank"], "rank_s": rank_of(sat["res"]),
            "pivots": sat["res"]["pivots"]}


def nullspace_basis(ref, free_cols, P):
    """One column per free coordinate."""
    dim = int(ref["dim"])
    if not free_cols:
        return np.zeros((dim, 0), dtype=np.int64)
    return np.stack([null_residues(ref, fc, P) for fc in free_cols], axis=1)

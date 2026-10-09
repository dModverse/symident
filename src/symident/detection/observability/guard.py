"""The saturation guard: the rank checked past the reported Lie order."""

from ... import state
from ...gfp import PRIMES
from ..fields import rank_of
from .context import ok as _ok


def saturation_guard(kcall, point0, nt_used, reported_rank, margin=None):
    """The rank past the reported Lie order: stable through `margin` further orders."""
    if margin is None:
        margin = int(state.env_value("VERIFY_MARGIN", "6"))
    P = PRIMES[0]
    try:
        r0 = kcall(point0, P, int(nt_used))
    except Exception:
        r0 = None
    if not _ok(r0):
        return {"ok": None, "method": "saturation guard", "reason": "base point not re-evaluable"}
    base = int(rank_of(r0))
    max_r, grow_at = base, None
    margin = max(1, margin)

    def rank_at(k):
        try:
            rk = kcall(point0, P, int(nt_used + k))
        except Exception:
            return None
        return int(rank_of(rk)) if _ok(rk) else None
    top = rank_at(margin)
    if top is not None:
        max_r = max(max_r, top)
    if top is None or top > base:
        for k in range(1, margin + 1):
            rk = rank_at(k)
            if rk is None:
                continue
            max_r = max(max_r, rk)
            if rk > base:
                grow_at = int(nt_used + k)
                break
        if grow_at is None and top is not None and top > base:
            grow_at = int(nt_used + margin)
    stable = grow_at is None
    return {"ok": stable, "method": "saturation guard", "lie_order_used": int(nt_used),
            "orders_checked": int(nt_used + margin), "kernel_rank": base, "kernel_rank_extended": max_r,
            "grow_at": grow_at,
            "reason": (f"rank {base} stable through Lie order {nt_used + margin} ({margin} orders "
                       "beyond the reported saturation)") if stable else
            (f"rank grows {base} -> {max_r} at Lie order {grow_at} (the reported Lie order was premature)")}

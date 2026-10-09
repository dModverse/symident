"""Conditions, events and initial values as per-segment inputs of the tape compiler."""

import math
import re

from ..rexpr import as_number, get_symbols, is_number, replace_symbols, sym_sort
from ..sym.charts import log_param_backsub


def num_str(x):
    """A number as R's as.character() writes it (15 significant digits)."""
    if isinstance(x, str):
        return x
    if isinstance(x, bool):
        return "TRUE" if x else "FALSE"
    if isinstance(x, int):
        return str(x)
    x = float(x)
    if math.isnan(x):
        return "NA"
    if math.isinf(x):
        return "Inf" if x > 0 else "-Inf"
    if x == int(x) and abs(x) < 1e15:
        return str(int(x))
    s = f"{x:.15g}"
    if "e" in s:
        mant, ex = s.split("e")
        sign = ex[0]
        digits = ex[1:].lstrip("0")
        s = f"{mant}e{sign}{digits.zfill(2)}"
    return s


class Grid:
    """Condition table: row names and columns of numbers or strings."""

    def __init__(self, rows, cols):
        self.rows = [str(r) for r in rows]
        self.cols = {str(k): list(v) for k, v in cols.items()}

    @classmethod
    def from_any(cls, x):
        """A Grid from a pandas DataFrame or a dict with `rows` and `cols`; None and a Grid pass through."""
        if x is None or isinstance(x, Grid):
            return x
        if hasattr(x, "to_dict") and hasattr(x, "index"):
            return cls(list(x.index), {c: list(x[c]) for c in x.columns})
        if isinstance(x, dict) and "rows" in x:
            return cls(x["rows"], x.get("cols", {}))
        raise TypeError("conditions: a pandas DataFrame or {'rows': [...], 'cols': {...}}")

    @property
    def nrow(self):
        """The number of rows."""
        return len(self.rows)

    def colnames(self):
        """The column names."""
        return list(self.cols)

    def cell(self, i, col):
        """Entry `i` of column `col` as a string."""
        return num_str(self.cols[col][i])

    def is_numeric(self, col):
        """Whether every entry of column `col` is an int or a float."""
        return all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in self.cols[col])

    def is_character(self, col):
        """Whether column `col` holds a string."""
        return any(isinstance(v, str) for v in self.cols[col])

    def rename(self, frm, to):
        """Rename the columns `frm` to `to` in place."""
        self.cols = {(to[frm.index(k)] if k in frm else k): v for k, v in self.cols.items()}


def as_events(events):
    """Events as a list of dicts with string var, time, value, method, optional root and
    optional cond, the conditions (0-based) an event belongs to."""
    if not events:
        return []
    out = []
    for e in events:
        d = {"var": str(e["var"]), "time": num_str(e.get("time", "NA")),
             "value": num_str(e["value"]), "method": str(e.get("method", "replace")),
             "root": e.get("root")}
        if e.get("cond") is not None:
            d["cond"] = [int(c) for c in e["cond"]]
        out.append(d)
    return out


def compose_event(cur, method, value):
    """The value after an event applies `value` to `cur` by `method` (replace, add or multiply)."""
    if method == "replace":
        return value
    if method == "add":
        return f"({cur}) + ({value})"
    if method == "multiply":
        return f"({cur}) * ({value})"
    raise ValueError(f"unsupported event method '{method}'")


def resolve_conditions(conditions, events, initial, symbols, states, const_states=(), forcings=(),
                       t0=None, equilibrate=False, cond_subs=None, cond_initial=None, n_cond_obs=0):
    """Per-segment substitutions, initial values and events; one chain per condition."""
    n_grid = 0 if conditions is None else conditions.nrow
    Kcond = max(1, len(cond_subs or []), len(cond_initial or []), n_grid, n_cond_obs)
    if n_grid and cond_subs and len(cond_subs) != n_grid:
        raise ValueError(f"the per-condition `trafo` list length "
                         f"({len(cond_subs)}) must match the condition grid rows ({n_grid}).")
    if conditions is None:
        conditions = Grid([str(i + 1) for i in range(Kcond)], {})
    conds = conditions.rows or [str(i + 1) for i in range(max(1, conditions.nrow))]
    cols = conditions.colnames()
    symset = set(symbols)
    sub_cols = [c for c in cols if c in symset]
    const_states = list(const_states)
    dyn_states = [s for s in states if s not in set(const_states)]
    ic_cols = [c for c in sub_cols if c in set(dyn_states)]
    sub_cols = [c for c in sub_cols if c not in set(ic_cols)]
    ev_all = events or []
    if t0 is None:
        nums = [as_number(e["time"].strip()) for e in ev_all]
        nums = [t for t in nums if t is not None]
        t0 = min(nums) if nums else 0
    t0 = float(t0)

    def layout(ev):
        tstr = [e["time"].strip() for e in ev]
        tnum = [as_number(t) for t in tstr]
        is_t0 = [t is not None and t <= t0 for t in tnum]
        key = [tstr[i] if tnum[i] is None else num_str(tnum[i]) for i in range(len(ev))]
        if all(t is not None for t in tnum):
            post_keys = [num_str(x) for x in sorted({tnum[i] for i in range(len(ev)) if not is_t0[i]})]
        else:
            post_keys = list(dict.fromkeys(key[i] for i in range(len(ev)) if not is_t0[i]))
            kn = [as_number(k) for k in post_keys]
            kn = [k for k in kn if k is not None]
            if any(kn[i] >= kn[i + 1] for i in range(len(kn) - 1)):
                raise ValueError("with an event time given in parameters, the "
                                 "list of events sets the order of the events and must list the numeric "
                                 "times ascending.")
        seg_of = [1 if is_t0[i] else 2 + post_keys.index(key[i]) for i in range(len(ev))]
        return is_t0, seg_of, [num_str(t0)] + post_keys

    def cell(ci, col):
        return conditions.cell(ci, col)

    def resolve(val, ci):
        val = str(val)
        v = cell(ci, val) if val in cols else val
        if cond_subs and ci < len(cond_subs) and cond_subs[ci]:
            sub = cond_subs[ci]
            v = replace_symbols(list(sub), list(sub.values()), v)
        return v
    dummy_ic0 = {s: "0" for s in dyn_states}
    subs_list, ic0_list, ev0_list, seg_ev_list, time_list, rest_list = [], [], [], [], [], []
    equil_list, chain_of, pos_in_chain = [], [], []
    initial = dict(initial) if initial else None
    n_gaps = 0
    for k in range(len(conds)):
        ev = [e for e in ev_all if e.get("cond") is None or k in e["cond"]]
        is_t0, seg_of, seg_times = layout(ev)
        n_seg = len(seg_times)
        n_gaps = max(n_gaps, n_seg - 1)
        ev_var = [e["var"] for e in ev]
        ev_method = [e["method"] for e in ev]
        is_switch = [v in set(const_states) for v in ev_var]
        subs_base = {col: cell(k, col) for col in sub_cols}
        if cond_subs and k < len(cond_subs) and cond_subs[k] is not None:
            for nm, v in cond_subs[k].items():
                subs_base[nm] = v
        init_k = cond_initial[k] if cond_initial else initial
        rest = {}
        seg_ev = []
        for j in range(1, n_seg + 1):
            subs = dict(subs_base)
            for cs in const_states:
                si = [i for i in range(len(ev)) if is_switch[i] and ev_var[i] == cs and seg_of[i] <= j]
                if cs in forcings:
                    base = "0"
                elif init_k is not None and cs in init_k:
                    base = str(init_k[cs])
                else:
                    base = subs_base.get(cs, cs)
                if not si:
                    if base != cs:
                        subs[cs] = base
                    continue
                cur = base
                si.sort(key=lambda i: (seg_of[i], i))
                if j == 1 and cs not in forcings:
                    ri = [i for i in si if ev_method[i] == "replace"]
                    rest[cs] = resolve(ev[ri[-1]]["value"], k) if ri else cur
                for i in si:
                    cur = compose_event(cur, ev_method[i], resolve(ev[i]["value"], k))
                subs[cs] = cur
            if j == 1:
                le_idx = [i for i in range(len(ev)) if not is_switch[i] and is_t0[i]]
                ev0 = [{"var": ev_var[i], "method": ev_method[i], "value": resolve(ev[i]["value"], k)}
                       for i in le_idx]
                ic0 = {}
                for X in dyn_states:
                    is_forcing = X in forcings
                    has_init = init_k is not None and X in init_k
                    has_cell = X in ic_cols
                    ops = [i for i in le_idx if ev_var[i] == X]
                    if not is_forcing and not has_init and not has_cell and not ops:
                        continue
                    if is_forcing:
                        cur = "0"
                    elif has_init:
                        cur = str(init_k[X])
                    elif has_cell:
                        cur = resolve(cell(k, X), k)
                    else:
                        cur = X
                    for i in ops:
                        cur = compose_event(cur, ev_method[i], resolve(ev[i]["value"], k))
                    ic0[X] = cur
            else:
                ev0 = []
                ic0 = dummy_ic0
                seg_ev = [{"var": ev_var[i], "method": ev_method[i], "value": resolve(ev[i]["value"], k)}
                          for i in range(len(ev)) if not is_switch[i] and seg_of[i] == j]
            subs_list.append(subs)
            rest_list.append(rest)
            ic0_list.append(ic0)
            ev0_list.append(ev0)
            seg_ev_list.append([] if j == 1 else seg_ev)
            time_list.append(resolve(seg_times[j - 1], k))
            equil_list.append(bool(equilibrate))
            chain_of.append(k + 1)
            pos_in_chain.append(j)
    return {"subs": subs_list, "rest_subs": rest_list, "ic0": ic0_list, "seg_equil": equil_list,
            "events0": ev0_list, "seg_events": seg_ev_list, "chain_of": chain_of,
            "pos_in_chain": pos_in_chain, "conditions": conds, "n_conditions": len(conds),
            "n_gaps": n_gaps, "times": time_list}


def scaling_component(w, var):
    """The generator component `w*var` of a scaling with weight `w`."""
    w = str(w)
    if w == "1":
        return var
    if w == "-1":
        return f"-{var}"
    if re.fullmatch(r"-?[0-9]+(/[0-9]+)?", w):
        return f"{w}*{var}"
    return f"({w})*{var}"


def log_param_back(res, lp):
    """A log parameter analysed in X = base^theta reported in theta."""
    if res is None or not lp or not isinstance(res, dict) or res.get("rerun_blocks"):
        return res
    X = [str(e["X"]) for e in lp]
    th = [str(e["theta"]) for e in lp]
    b = [str(e["base"]) for e in lp]

    def ren(v):
        return [th[X.index(x)] if x in X else x for x in v]
    res = dict(res)
    res["coordinates"] = ren(res.get("coordinates") or [])
    out = []
    for d in res.get("non_identifiable", []):
        d = dict(d)
        d["support"] = sym_sort(ren(d.get("support", [])))
        vec = d.get("vector")
        if vec is None:
            out.append(d)
            continue
        if d.get("type") == "scaling":
            if not any(k in X for k in vec):
                out.append(d)
                continue
            vec = {k: scaling_component(v, k) for k, v in vec.items()}
            d["type"] = "general"
        d["vector"] = {ren([k])[0]: str(log_param_backsub(str(v), k, X, th, b))
                       for k, v in vec.items()}
        out.append(d)
    res["non_identifiable"] = out
    return res


def grid_pinned_states(conditions, states):
    """The states among `states` with a numeric column in `conditions`."""
    if conditions is None or not states:
        return []
    out = []
    for c in conditions.colnames():
        if c in states:
            vals = conditions.cols[c]
            if vals and all(is_number(v) for v in vals):
                out.append(c)
    return out


def event_pinned_states(events, conditions):
    """States pinned by a numeric replace or add dose in any condition."""
    if not events:
        return []
    cols = conditions.colnames() if conditions is not None else []

    def is_known(v):
        vals = [num_str(x) for x in conditions.cols[v]] if v in cols else [v]
        return bool(vals) and all(is_number(x) for x in vals)
    pinned = []
    for e in events:
        if e["method"] in ("replace", "add") and is_known(e["value"]):
            if e["var"] not in pinned:
                pinned.append(e["var"])
    return pinned


def initial_value_split(trafos, states, taken=()):
    """State names on the right of a trafo renamed to fresh initial-value coordinates."""
    trafos = [dict(t) for t in trafos]
    none = {"trafos": trafos, "from": [], "to": []}
    if not trafos or not states:
        return none
    stset = set(states)
    conflict = set()
    for tr in trafos:
        sub_e = [k for k in tr if k not in stset]
        conflict |= set(get_symbols([tr[k] for k in sub_e])) & stset
    conflict = [s for s in states if s in conflict]
    if not conflict:
        return none
    taken = list(dict.fromkeys(list(taken) + list(states) + [k for tr in trafos for k in tr] +
                               [s for tr in trafos for s in get_symbols(list(tr.values()))]))
    to = []
    for X in conflict:
        cand = f"{X}_init"
        while cand in taken:
            cand += "0"
        taken.append(cand)
        to.append(cand)
    out = []
    for tr in trafos:
        tr = {k: replace_symbols(conflict, to, v) for k, v in tr.items()}
        for X, t in zip(conflict, to):
            if X not in tr:
                tr[X] = t
        out.append(tr)
    return {"trafos": out, "from": conflict, "to": to}


def rename_result(raw, frm, to):
    """Initial-value coordinates mapped back onto the names the user gave."""
    if not frm or raw is None or not isinstance(raw, dict):
        return raw

    def ren(x):
        if isinstance(x, str):
            return replace_symbols(to, frm, x)
        return [replace_symbols(to, frm, str(v)) for v in x]

    def ren_names(v):
        return {(frm[to.index(k)] if k in to else k): val for k, val in v.items()}

    def fix_dir(d):
        d = dict(d)
        for fld in ("vector", "infinitesimals"):
            if d.get(fld) is not None:
                d[fld] = {k: (ren(v) if isinstance(v, str) else v) for k, v in ren_names(d[fld]).items()}
        if d.get("support") is not None:
            d["support"] = ren(d["support"])
        return d
    raw = dict(raw)
    if raw.get("coordinates") is not None:
        raw["coordinates"] = ren(raw["coordinates"])
    if raw.get("non_identifiable") is not None:
        raw["non_identifiable"] = [fix_dir(d) for d in raw["non_identifiable"]]
    return raw


def percond_lines(fdyn, gset, g_per_cond, conditions, cond_subs, symbols):
    """Per-condition (f, g) lines for the scaling engine."""
    from ..rexpr import eqn_lines
    n_grid = 0 if conditions is None else conditions.nrow
    K = max(len(cond_subs or []), n_grid, len(gset) if g_per_cond else 0, 1)
    symset = set(symbols)
    sub_cols = [] if conditions is None else \
        [c for c in conditions.colnames() if c in symset and c not in fdyn]
    out = []
    for k in range(K):
        subs = {col: conditions.cell(k, col) for col in sub_cols}
        if cond_subs and k < len(cond_subs) and cond_subs[k] is not None:
            subs.update(cond_subs[k])

        def sub(e):
            if not subs or not e:
                return e
            return {n: replace_symbols(list(subs), list(subs.values()), v) for n, v in e.items()}
        out.append({"f": eqn_lines(sub(fdyn)), "g": eqn_lines(sub(gset[k if g_per_cond else 0]))})
    return out

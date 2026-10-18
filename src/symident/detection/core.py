"""The detection pipeline: model preprocessing, coordinate trafos, conditions and the engines."""

import time
import warnings

from .. import session, state
from ..errors import SYMPY_FAILS, swallow, sympy_fail
from ..options import reconst_control
from ..rexpr import (
    eqn_lines,
    eval_numeric,
    free_exponent_bases,
    get_symbols,
    is_number,
    replace_symbols,
    subst,
    tokens,
)
from ..sym.charts import log_arg_chart, log_arg_event, sign_chart
from ..sym.parse import model_sympify, to_r
from ..sym.scaling import scaling_symmetries_multi, scaling_symmetries_single
from ..sym.switching import hybrid_model
from ..sym.symbolic import observability_sympy_multi
from ..sym.tape import compile_observability_tape_multi
from .classify import relabel_directions
from .conditions import (
    Grid,
    as_events,
    event_pinned_states,
    grid_pinned_states,
    initial_value_split,
    log_param_back,
    percond_lines,
    rename_result,
    resolve_conditions,
)
from .cq import equil_zero_states, pivot_decomposition, substitute_cq
from .finalize import finalize, log_arg_back
from .gauge import gauge_suggest, gauged_detection
from .observability import observability_analytic_multi


def _clean(e):
    if e is None:
        return None
    return {str(k): to_r(v) for k, v in dict(e).items()}


def _sub_or_none(fn, e):
    return None if e is None else fn(e)


def _ode_zero_states(fdyn, forcings):
    """States whose right-hand side, with inputs and earlier zero states at zero, has zero as its only root."""
    import sympy as spy

    from ..sym.parse import _make_local_parse
    _, parse = _make_local_parse([f"{k} = {v}" for k, v in fdyn.items()])
    rhs = {k: parse(str(v).replace("^", "**")) for k, v in fdyn.items()}
    zero = {spy.Symbol(u): 0 for u in forcings}
    found = []
    while True:
        new = []
        for k, e in rhs.items():
            if k in found or k in forcings:
                continue
            x = spy.Symbol(k)
            e0 = spy.expand(e.xreplace(zero))
            if e0.xreplace({x: 0}) != 0:
                continue
            num = spy.fraction(spy.together(e0))[0]
            try:
                linear = spy.Poly(num, x).degree() == 1
            except spy.PolynomialError:
                linear = False
            if linear and spy.diff(e0, x).xreplace({x: 0}) != 0:
                new.append(k)
        if not new:
            return found
        found += new
        zero.update({spy.Symbol(k): 0 for k in new})


def _drop_sinks(fdyn, g_exprs, trafo, events, reactions, totals, keep, zero=()):
    """The model without the states that no right-hand side, observable, event or trafo value holds."""
    import sympy as spy
    trafos = (trafo if isinstance(trafo, (list, tuple)) else [trafo]) if trafo is not None else []
    rhs = list(fdyn.values())
    if zero and not events:
        at0 = {spy.Symbol(z): 0 for z in zero}
        rhs = [to_r(model_sympify(e).xreplace(at0)) for e in rhs]
    used = set(get_symbols(rhs + list(g_exprs) +
                           [str(v) for t in trafos if t for v in dict(t).values()] +
                           [str(e["value"]) for e in events] + [str(e["time"]) for e in events]))
    sinks = [s for s in fdyn if s not in used and s not in set(keep) and s not in set(zero)]
    if not sinks:
        return fdyn, trafo, events, reactions, totals, []
    drop = set(sinks)
    fdyn = {k: v for k, v in fdyn.items() if k not in drop}

    def strip(t):
        return None if t is None else {k: v for k, v in dict(t).items() if k not in drop}
    trafo = [strip(t) for t in trafo] if isinstance(trafo, (list, tuple)) else strip(trafo)
    events = [e for e in events if e["var"] not in drop]
    if reactions is not None:
        keep_cols = [j for j, sp in enumerate(reactions["species"]) if sp not in drop]
        reactions = dict(reactions, species=[reactions["species"][j] for j in keep_cols],
                         smatrix=[[row[j] for j in keep_cols] for row in reactions["smatrix"]])
    if totals:
        totals = {k: v for k, v in totals.items() if not drop & set(get_symbols([str(v)]))}
    # a dropped state that still multiplies a state at zero becomes a known constant, so
    # that its rate constants stay coordinates
    held = sorted(drop & set(get_symbols(list(fdyn.values()))))
    return fdyn, trafo, events, reactions, totals, held


def _off_domain(exprs, positive, states):
    """A log or fractional power of a state term not positive under the declared signs, as a string, or None."""
    import sympy as spy
    names = None if positive is True else set(positive or [])
    for e in exprs:
        try:
            x = model_sympify(e)
        except SYMPY_FAILS as exc:
            sympy_fail(exc)
            continue
        for a in x.atoms(spy.log, spy.Pow):
            u = a.args[0] if isinstance(a, spy.log) else a.base
            if isinstance(a, spy.Pow) and (not a.exp.is_Rational or a.exp.is_Integer or u.is_number):
                continue
            if not {str(x) for x in u.free_symbols} & states:
                continue
            # coordinates are real; the declared ones positive
            pos = {s: spy.Symbol(str(s), positive=True) if names is None or str(s) in names
                   else spy.Symbol(str(s), real=True) for s in u.free_symbols}
            if not u.xreplace(pos).is_positive:
                return str(a).replace("**", "^")
    return None


# piecewise constant functions with a jump at every integer
_STEPS = ("floor", "ceiling", "round")


def _check_steps(kw):
    """ValueError for a call of floor(), ceiling() or round() in the model."""
    texts = []
    for key in ("f", "g", "trafo", "events"):
        for x in (kw.get(key) if isinstance(kw.get(key), (list, tuple)) else [kw.get(key)]):
            texts += [str(v) for v in (x.values() if isinstance(x, dict) else [x]) if v is not None]
    for t in texts:
        hit = [n for k, n in tokens(t) if k == "fun" and n in _STEPS]
        if hit:
            raise ValueError(f"{hit[0]}() in {t} jumps at every integer of its argument and is constant "
                             "in between; symident does not analyze it: write the steps it takes with "
                             "piecewise() on their thresholds")


def _check_names(kw):
    """ValueError for a state, parameter or input named pi."""
    names = list(dict(kw.get("f") or {}))
    for key in ("parameters", "fixed", "forcings", "free_initial"):
        names += [str(x) for x in kw.get(key) or []]
    trafo = kw.get("trafo")
    for t in trafo if isinstance(trafo, (list, tuple)) else [trafo]:
        names += list(dict(t or {}))
    names += [str(e.get("var")) for e in kw.get("events") or []]
    if "pi" in names:
        raise ValueError("pi is the number in the equations and cannot name a state, parameter or input")


def symmetry_detection(*args, **kw):
    """The detection on plain inputs (see _symmetry_detection); a model with switches that
    the declared signs leave open goes to the hybrid analysis."""
    import inspect
    kw = dict(inspect.signature(_symmetry_detection).bind(*args, **kw).arguments)
    _check_steps(kw)
    _check_names(kw)
    regimes = kw.pop("regimes", None)
    hy = None
    if kw.get("f") is not None:
        hy = hybrid_model({str(k): str(v) for k, v in dict(kw["f"]).items()}, kw.get("g"),
                          kw.get("positive", True), events=kw.get("events"),
                          inputs=kw.get("forcings") or (), taken=kw.get("parameters") or ())
    if isinstance(hy, dict):
        kw.update(f=hy["f"], g=hy["g"])
        hy = None
    if hy is None and regimes is not None:
        raise ValueError("`regimes` names regimes, but the model has no switch that the "
                         "declared signs leave open")
    outer = state.get("session") is None
    if outer:
        state.check_names()
    with session.opened(kw) as ses:
        ses.depth += 1
        try:
            if hy is not None:
                from .hybrid import hybrid_detection
                res = hybrid_detection(hy, kw, regimes)
            else:
                res = detection_with_primes(kw)
        finally:
            ses.depth -= 1
        if outer and hy is None and isinstance(res, dict) and isinstance(res.get("info"), dict) \
                and res["info"].get("bound") is None:
            res["info"]["bound"] = _kernel_bound(kw, res)
        if outer and isinstance(res, dict) and isinstance(res.get("info"), dict):
            _gap_settled(res["info"])
            _bound_needs_orders(res["info"])
            _checked_without_bound(res)
        if outer and isinstance(res, dict) and isinstance(res.get("info"), dict) and \
                res.get("method") == "observability":
            res["info"]["rank_status"] = _rank_status(res)
    if outer and isinstance(res, dict) and isinstance(res.get("info"), dict):
        from .. import __version__
        res["info"]["limits"] = list(ses.notes["limits"])
        res["info"]["errors"] = list(ses.notes["errors"])
        res["info"]["branches"] = list(ses.notes["branches"])
        st = res["info"].setdefault("settings", {})
        st.update(control={k: v for k, v in ses.control.items() if k != "deadline"},
                  switches={n: v for n, v in state.active().items()
                            if state.SWITCHES.get(n, ("",))[0] != "install"},
                  primes=ses.primes, version=__version__)
        res["info"]["assumptions"] = _assumptions(res)
    return res


def _checked_without_bound(res):
    """A verified direction whose error bound has no probability is checked, not verified."""
    b = res["info"].get("bound")
    if isinstance(b, dict) and b.get("probability") is not None:
        return
    for s in res.get("symmetries") or []:
        if s.get("level") == "verified":
            s["level"] = "checked"


def _assumptions(res):
    """The declared domain and switch starts the result assumes, and its error probability."""
    info = res["info"]
    b = info.get("bound")
    levels = [s.get("level") for s in res.get("symmetries") or []]
    starts = {c["name"]: c["declared"] for c in (info.get("switching") or {}).get("conditions") or []
              if c.get("declared")}
    rs = info.get("rank_status") or {}
    return {"domain": (info.get("settings") or {}).get("positive", True), "starts": starts or None,
            "error_bound": b.get("probability") if isinstance(b, dict) and "verified" in levels else
            (rs.get("probability") if rs.get("status") == "verified" else None)}


def _rank_status(res):
    """{"status", "why", "probability"} of the rank: "proven", "verified" or "lower bound"; None without a rank."""
    info = res["info"]
    if res.get("rank") is None or res.get("dim") is None:
        return None
    if res["rank"] >= res["dim"]:
        return {"status": "proven", "why": "full rank at a point", "probability": None}
    if info.get("rank_proven") is not None:
        return {"status": "proven", "why": info.get("rank_proof"), "probability": None}
    sw = (info.get("switching") or {}).get("uncertified")
    if sw:
        return {"status": "lower bound", "probability": None,
                "why": "the rule of the switches is not proven for every direction"}
    if info.get("rank_fields") is not None:
        b = info.get("rank_fields_bound") or {}
        if b.get("probability") is not None:
            return {"status": "verified", "probability": b["probability"],
                    "why": "invariant fields hold the kernel at every order, checked at random points "
                           "modulo a prime"}
        return {"status": "lower bound", "probability": None,
                "why": "invariant fields hold the kernel at every order, checked at random points modulo "
                       f"a prime without an error bound ({b.get('why') or 'not taken'})"}
    syms = res.get("symmetries") or []
    if info.get("engine") == "symbolic" and not info.get("lie_certified"):
        why = "the Lie order of the symbolic engine is not proven sufficient" + \
            (f": {info['lie_certificate']}" if info.get("lie_certificate") else "")
    elif not info.get("lie_certified"):
        why = "the Lie order is not proven sufficient"
    elif (info.get("verification") or {}).get("ok") is False:
        why = "the rank grows past the orders used"
    elif info.get("gap_certified") is False:
        why = "the gap order is not proven sufficient"
    elif info.get("series_certified") is False:
        why = "the series order of the resting states is not proven sufficient"
    elif any(not s.get("explicit") for s in syms):
        why = "a direction has no closed form"
    elif any(s.get("level") == "checked" for s in syms):
        why = "a closed form is checked without an error bound"
    elif not isinstance(info.get("bound"), dict) or info["bound"].get("probability") is None:
        why = "the closed forms have no error bound"
    else:
        return {"status": "verified", "probability": info["bound"]["probability"],
                "why": "every direction is closed and lies in the kernel; the rank holds unless a "
                       "verified closed form is wrong"}
    return {"status": "lower bound", "why": why, "probability": None}


def _gap_settled(info):
    """A proven rank proves the gap and series orders sufficient; otherwise a stopped order is a limit."""
    for key, what in (("gap", "gap order"), ("series", "series order of the resting states")):
        if info.get(f"{key}_certified") is not False:
            continue
        if info.get("rank_proven") is not None:
            info[f"{key}_certified"], info[f"{key}_certificate"] = True, None
            continue
        session.note("limits", f"{what}: {info.get(f'{key}_certificate')}; the order is not proven sufficient")


def _bound_needs_orders(info):
    """Drops the probability of info["bound"] where an unproven Lie or gap order leaves the kernel open."""
    b = info.get("bound")
    if not isinstance(b, dict) or b.get("probability") is None or info.get("engine") == "symbolic" or \
            info.get("rank_proven") is not None or info.get("rank_fields") is not None:
        return
    if not info.get("lie_certified"):
        why = "the Lie order is not proven sufficient"
    elif (info.get("verification") or {}).get("ok") is False:
        why = "the rank grows past the orders used"
    elif info.get("gap_certified") is False:
        why = "the gap order is not proven sufficient"
    else:
        return
    info["bound"] = dict(b, probability=None, applies=False,
                         why=f"{why}, and the check runs at the order reached")


def _kernel_bound(kw, res):
    """The kernel-check error bound of the verified directions, or why it does not apply; None without them."""
    from ..gfp import verify_prime
    from .bound import kernel_bound
    vecs = [s["generator"] for s in res.get("symmetries") or []
            if s.get("level") == "verified" and s.get("generator")]
    if not vecs:
        return None
    if (res.get("info") or {}).get("engine") == "symbolic":
        return {"applies": False, "why": "the symbolic engine takes the kernel exactly, up to the Lie order used"}
    if kw.get("equilibrate"):
        return {"applies": False, "why": "the resting state has no closed form"}
    every = as_events(kw.get("events")) or []
    if any(e.get("root") is not None for e in every):
        return {"applies": False, "why": "an event at a root"}
    evs, later = [], {}
    for e in every:
        try:
            t = float(str(e.get("time")))
        except ValueError:
            return {"applies": False, "why": "an event time depends on the parameters"}
        if t == 0:
            evs.append(e)
        else:
            later.setdefault(t, []).append(e)
    later = [later[t] for t in sorted(later)]
    gap_order = int(res["info"].get("gap_order_used") or 0)
    f = {str(k): str(v) for k, v in dict(kw["f"]).items()}
    gs = kw.get("g")
    gs = [gs] if not isinstance(gs, (list, tuple)) else list(gs)
    trafos = kw.get("trafo")
    trafos = [trafos] if not isinstance(trafos, (list, tuple)) else list(trafos)
    out = None
    for tr in trafos or [None]:
        tr = {str(k): str(v) for k, v in dict(tr or {}).items()}
        sub = {k: v for k, v in tr.items() if k not in f}
        x0 = {x: tr.get(x, x) for x in f}
        for e in evs:
            x, v = str(e["var"]), str(e["value"])
            x0[x] = {"add": f"({x0.get(x, x)}) + ({v})", "replace": v,
                     "multiply": f"({x0.get(x, x)})*({v})"}[e.get("method", "replace")]
        fs = {x: _subs(rhs, sub) for x, rhs in f.items()}
        x0 = {x: _subs(v, sub) for x, v in x0.items()}
        for g in gs:
            gg = {str(k): _subs(str(v), sub) for k, v in dict(g or {}).items()}
            grp = [[dict(e, value=_subs(str(e["value"]), sub)) for e in g_] for g_ in later]
            b = kernel_bound(fs, gg, x0, int(res["info"].get("lie_order_used") or 0), vecs, verify_prime(), 2,
                             events=grp, gap_order=gap_order)
            if b is None:
                return {"applies": False, "why": "a term is not rational"}
            out = b if out is None or b["probability"] > out["probability"] else out
    return out


def _subs(text, sub):
    """`text` with the parameter rules of `sub` applied, as a string."""
    if not sub:
        return str(text)
    import sympy as spy
    e = model_sympify(text)
    return to_r(e.xreplace({spy.Symbol(k): model_sympify(v) for k, v in sub.items()}))


def detection_with_primes(kw):
    """_symmetry_detection on `kw`, rerun under primes that hold the square roots the model needs."""
    import contextlib

    from ..gfp import NeedPrimesError, all_primes, use_primes
    consts, degrees = [], []
    while True:
        try:
            with use_primes(consts, degrees) if consts or degrees else contextlib.nullcontext():
                res = _symmetry_detection(**kw)
                ses = state.get("session")
                if ses is not None and ses.depth == 1:
                    ses.primes = all_primes()
                return res
        except NeedPrimesError as e:
            new_c = [c for c in e.constants if c not in consts]
            new_d = [d for d in e.degrees if d not in degrees]
            if not new_c and not new_d:
                raise ValueError(f"no engine primes hold the {e}") from None
            consts += new_c
            degrees += new_d


def _symmetry_detection(f=None, g=None, trafo=None, parameters=None, fixed=None,
                       gauge_preference=False, forcings=None, events=None, conditions=None,
                       equilibrate=False, reduce_cq=False, free_initial=None, reconstruct=False,
                       positive=True, verify=True, cores=1, control=None, scalings_only=False,
                       sym_engine="modular", verbose=False, reactions=None, totals=None,
                       regimes=None):
    """The result dictionary of the detection on plain inputs."""
    control = {**reconst_control(), **dict(control or {})}
    equilibrate = bool(equilibrate)
    method = "scaling" if scalings_only else "observability"
    if gauge_preference is not False and method != "observability":
        warnings.warn("`gauge` applies only without `scalings_only` and is ignored.", stacklevel=2)
    elif gauge_preference is not False:
        args = dict(f=f, g=g, trafo=trafo, parameters=parameters, fixed=fixed,
                    forcings=forcings, events=events, conditions=conditions,
                    equilibrate=equilibrate, reduce_cq=reduce_cq, free_initial=free_initial,
                    reconstruct=reconstruct, positive=positive, verify=verify, cores=cores,
                    control=control, scalings_only=scalings_only, sym_engine=sym_engine,
                    reactions=reactions, totals=totals)
        pref = None if gauge_preference is None else list(gauge_preference) \
            if not isinstance(gauge_preference, str) else [gauge_preference]
        return gauged_detection(args, pref, fixed, reconstruct, verbose)
    t0clock = time.monotonic()
    settings = {"positive": positive, "reduce_cq": bool(reduce_cq), "equilibrate": equilibrate,
                "reconstruct": bool(reconstruct), "verify": bool(verify), "sym_engine": sym_engine,
                "degree_cap": control["degree_cap"], "scalings_only": bool(scalings_only)}
    ic_from, ic_to = [], []
    log_arg = None
    chart_coords = []
    fixed = list(fixed or [])
    forcings = list(forcings or [])
    conditions = Grid.from_any(conditions)
    events = as_events(events)
    model_exprs_box = {}

    def deliver(raw, method, coordinates=None):
        raw = log_arg_back(raw, log_arg)
        if log_arg is not None and coordinates:
            coordinates = replace_symbols(log_arg["L"], log_arg["v"], list(coordinates))
        raw = rename_result(raw, ic_from, ic_to)
        if ic_from and coordinates:
            coordinates = replace_symbols(ic_to, ic_from, list(coordinates))
        res = finalize(raw, method, settings, elapsed=time.monotonic() - t0clock,
                       coordinates=coordinates)
        res["info"]["model_exprs"] = model_exprs_box["v"]
        res["info"]["model_rhs"] = model_exprs_box["rhs"]
        res["info"]["model_inputs"] = list(forcings)
        if raw.get("kernel") is not None and not state.get("inner") and not state.get("keep_kernel"):
            try:
                res["info"]["gauge_suggestion"] = gauge_suggest(res["symmetries"], raw["kernel"],
                                                                res["info"]["coordinates"],
                                                                fixed=fixed)
            except Exception as exc:
                swallow(exc, "gauge suggestion")
                res["info"]["gauge_suggestion"] = None
        return res

    if f is None:
        raise ValueError("Provide the model right-hand sides via `f`.")
    fdyn = _clean(f)
    states = list(fdyn)
    g_per_cond = isinstance(g, (list, tuple))
    gset = [_clean(x) for x in g] if g_per_cond else [_clean(g)]
    if g_per_cond and not gset:
        raise ValueError("`observables` is an empty list; give one dict per condition.")

    def g_char():
        return [v for x in gset if x for v in x.values()]

    def g_names():
        return list(dict.fromkeys(k for x in gset if x for k in x))

    def g_lines(k=0):
        return eqn_lines(gset[k])
    trafo_list = isinstance(trafo, (list, tuple))
    if equilibrate:
        zero = [] if reactions is None else equil_zero_states(reactions["smatrix"], reactions["species"],
                                                              reactions["rates"], forcings)
        fdyn, trafo, events, reactions, totals, held = _drop_sinks(
            fdyn, g_char(), trafo, events, reactions, totals, forcings + list(free_initial or []), zero)
        fixed = list(dict.fromkeys(fixed + held))
        states = list(fdyn)
    trafo_vals = [] if trafo is None else ([v for t in trafo if t for v in dict(t).values()]
                                           if trafo_list else list(dict(trafo).values()))
    model_exprs_box["v"] = list(dict.fromkeys(list(fdyn.values()) + g_char() +
                                            [str(v) for v in trafo_vals]))
    model_exprs_box["rhs"] = {k: str(v) for k, v in fdyn.items()}
    initial_box = {"v": None}
    cond_initial_box = {"v": None}

    def subst_model(fn):
        nonlocal gset
        gset = [_sub_or_none(fn, x) for x in gset]
        initial_box["v"] = _sub_or_none(fn, initial_box["v"])
        if cond_initial_box["v"] is not None:
            cond_initial_box["v"] = [_sub_or_none(fn, x) for x in cond_initial_box["v"]]

    if "time" not in states and "time" in get_symbols(list(fdyn.values()) + g_char()):
        evt = [float(e["time"]) for e in events if is_number(e["time"])]
        tt0 = min(evt + [0])
        fdyn = dict(fdyn, time="1")
        states = list(fdyn)
        if method == "observability":
            def add_clock(tr):
                tr = {} if tr is None else _clean(tr)
                return dict(tr, time=f"{tt0:.15g}")
            trafo = [add_clock(t) for t in trafo] if trafo_list else add_clock(trafo)
        else:
            fixed = list(dict.fromkeys(fixed + ["time"]))
    clash = [n for n in g_names() if n in states]
    if clash:
        raise ValueError(f"observable(s) {', '.join(clash)} in `observables` have the "
                         f"name of a state in `odes`. Rename the observable(s) (e.g. {clash[0]}_obs).")
    parameters = list(parameters or [])
    free_initial = list(free_initial or [])
    if free_initial:
        bad = [s for s in free_initial if s not in states]
        if bad:
            warnings.warn(f"`free_initial` names non-state(s) {', '.join(bad)}; ignored.",
                          stacklevel=2)
        free_initial = [s for s in free_initial if s in states]
        if free_initial and not (method == "observability" and equilibrate and not reduce_cq):
            warnings.warn("`free_initial` applies only with `equilibrate` and without `reduce_cq` "
                          "(the held-variable moiety parameterisation); ignored.", stacklevel=2)
            free_initial = []

    cond_subs = None
    trafo_syms = []
    trafo_subs = {}
    trafo_hit = []
    if trafo_list:
        trafos = [_clean(t) for t in trafo]
        ic_split = initial_value_split(trafos, states, taken=parameters + g_names())
        trafos, ic_from, ic_to = ic_split["trafos"], ic_split["from"], ic_split["to"]
        cond_subs = [{k: v for k, v in tr.items() if k not in set(states)} for tr in trafos]
        cond_initial_box["v"] = [({k: v for k, v in tr.items() if k in set(states)} or None)
                                 for tr in trafos]
        trafo_syms = list(dict.fromkeys(s for tr in trafos for s in get_symbols(list(tr.values()))))
    elif trafo is not None:
        ic_split = initial_value_split([_clean(trafo)], states, taken=parameters + g_names())
        tr = ic_split["trafos"][0]
        ic_from, ic_to = ic_split["from"], ic_split["to"]
        ic_entries = [k for k in tr if k in set(states)]
        subs = {k: v for k, v in tr.items() if k not in set(states)}
        sub = subst(list(subs), list(subs.values()))
        trafo_subs = {k: v for k, v in subs.items() if v.strip() != k}
        if trafo_subs:
            known = set(get_symbols(list(fdyn.values()) + g_char() + [tr[k] for k in ic_entries]))
            if events:
                known |= set(get_symbols([e["value"] for e in events] + [e["time"] for e in events]))
            if conditions is not None:
                known |= set(conditions.colnames())
                for c in conditions.colnames():
                    known |= {str(v) for v in conditions.cols[c] if isinstance(v, str)}
            trafo_hit = [k for k in trafo_subs if k in known]
        fdyn = sub(fdyn)
        subst_model(sub)
        if ic_entries:
            initial_box["v"] = sub({k: tr[k] for k in ic_entries})
        if events and trafo_subs:
            for e in events:
                e["value"] = replace_symbols(list(trafo_subs), list(trafo_subs.values()), e["value"])
                e["time"] = replace_symbols(list(trafo_subs), list(trafo_subs.values()), e["time"])
    if ic_from:
        if conditions is not None:
            conditions.rename(ic_from, ic_to)
        for e in events:
            e["value"] = replace_symbols(ic_from, ic_to, e["value"])
    if conditions is not None:
        keep = [c for c in conditions.colnames() if c not in set(states)]
        chr_ = [c for c in keep if conditions.is_character(c)]
        hit = [s for s in get_symbols([str(v) for c in chr_ for v in conditions.cols[c]])
               if s in set(states)] if chr_ else []
        if hit:
            warnings.warn(f"condition-grid value(s) name the state(s) "
                          f"{', '.join(hit)}. A grid value is substituted into `odes`, where that name "
                          "is the running state and not an initial value. Pass it through a "
                          "per-condition `trafo` instead.", stacklevel=2)
    n_cond_obs = len(gset) if g_per_cond else 0
    if g_per_cond:
        n_grid_g = 0 if conditions is None else conditions.nrow
        if n_grid_g and n_cond_obs != n_grid_g:
            raise ValueError(f"the per-condition `observables` list length ({n_cond_obs}) "
                             f"must match the condition grid rows ({n_grid_g}).")
        if cond_subs and n_cond_obs != len(cond_subs):
            raise ValueError(f"the per-condition `observables` list length ({n_cond_obs}) "
                             f"must match the per-condition `trafo` list length ({len(cond_subs)}).")
    ic_names = list(dict.fromkeys(list(initial_box["v"] or {}) +
                                 [k for ci in (cond_initial_box["v"] or []) if ci for k in ci]))
    if equilibrate and ic_names:
        warnings.warn("`equilibrate` solves the steady state from f = 0; the "
                      f"initial condition(s) for {', '.join(ic_names)} in `trafo` are ignored.", stacklevel=2)
        initial_box["v"] = None
        if cond_initial_box["v"] is not None:
            cond_initial_box["v"] = [None for _ in cond_initial_box["v"]]
    if reduce_cq and not equilibrate and reactions is not None and ic_names and totals:
        moiety = set(get_symbols(list(totals.values())))
        clash_cq = [s for s in ic_names if s in moiety]
        if clash_cq:
            warnings.warn("a `trafo` initial condition was supplied for "
                          f"{', '.join(clash_cq)}, which participate(s) in a conserved quantity; "
                          "`reduce_cq` would eliminate a moiety species and discard that "
                          "steady-state relation. Forcing reduce_cq=False so the supplied steady "
                          "state is used as given.", stacklevel=2)
            reduce_cq = False
            settings["reduce_cq"] = False
    cq_totals = []
    if reactions is not None and reduce_cq and totals:
        avoid_cq = free_exponent_bases(list(fdyn.values()) + g_char(), list(fdyn))
        cq = substitute_cq(totals, fdyn, list(fdyn), parameters, avoid=avoid_cq)
        fdyn = {k: v for k, v in cq["f"].items() if k not in set(cq["elim_states"])}
        parameters = cq["parameters"]
        states = list(fdyn)
        if cq["cq_info"]:
            cq_totals = [ci["total_name"] for ci in cq["cq_info"]]
            subst_model(subst([ci["elim_state"] for ci in cq["cq_info"]],
                             [ci["recon_expr"] for ci in cq["cq_info"]]))
    if trafo_subs and cq_totals:
        hit_cq = [k for k in trafo_subs if k in cq_totals]
        if hit_cq:
            sub_cq = subst(hit_cq, [trafo_subs[k] for k in hit_cq])
            fdyn = sub_cq(fdyn)
            subst_model(sub_cq)
            parameters = [p for p in parameters if p not in hit_cq]
            trafo_hit = list(dict.fromkeys(trafo_hit + hit_cq))
    if trafo_subs:
        miss = [k for k in trafo_subs if k not in trafo_hit]
        if miss:
            warnings.warn(f"`trafo` entr{'y' if len(miss) == 1 else 'ies'} "
                          f"{', '.join(miss)} {'does' if len(miss) == 1 else 'do'} not occur in the "
                          "model and had no effect. Note that the `total` parameters of a conserved "
                          "quantity only exist with `reduce_cq`.", stacklevel=2)
    import re as _re
    if any(_re.search(r"\b(abs|sign|max|min|pmax|pmin|ifelse|Heaviside)\s*\(", x)
           for x in list(fdyn.values()) + g_char()):
        sc = sign_chart(dict(fdyn), [dict(x or {}) for x in gset],
                          True if positive is True else ([] if positive is False else list(positive)))
        if sc.get("why") is not None:
            raise ValueError(f"{sc['why']}.")
        fdyn = {k: to_r(v) for k, v in sc["f"].items()}
        gset = [({k: to_r(v) for k, v in x.items()} or None) for x in sc["g"]]
    off = _off_domain(list(fdyn.values()) + g_char(), positive, set(states))
    if positive is not False and method != "scaling" and not (sym_engine == "symbolic"):
        ev_ic = [{e["var"]: e["value"]} for e in events if e["method"] in ("replace", "multiply")]
        lc = log_arg_chart(dict(fdyn), [dict(x or {}) for x in gset],
                            True if positive is True else list(positive),
                            list(dict.fromkeys(parameters + g_names() + ic_to)),
                            [dict(initial_box["v"] or {})] +
                            [dict(x or {}) for x in (cond_initial_box["v"] or [])] + ev_ic)
        if lc is not None and lc.get("why") is not None:
            raise ValueError(f"{lc['why']}.")
        if lc is not None:
            vs = [m["v"] for m in lc["map"]]
            Ls = [m["L"] for m in lc["map"]]
            clashes = set(forcings)
            if conditions is not None:
                clashes |= set(conditions.colnames())
            for cs in (cond_subs or []):
                clashes |= set(cs) | set(get_symbols(list(cs.values())))
            clash_l = [v for v in vs if v in clashes]
            if clash_l:
                raise ValueError(f"{', '.join(clash_l)} appear(s) inside log() or "
                                 "under a fractional power and is analyzed as exp(log_<name>); "
                                 "inputs and conditions on it are not supported.")

            def from_py(e):
                return {k: to_r(v) for k, v in e.items()} if e else None
            fdyn = from_py(lc["f"])
            states = list(fdyn)
            gset = [from_py(x) for x in lc["g"]]
            lognum = list(lc.get("consts") or [])
            if events:
                for e in events:
                    ce = log_arg_event(e["var"], e["value"], e["method"], lc["map"])
                    if ce.get("why") is not None:
                        raise ValueError(f"{ce['why']}.")
                    e["var"] = ce["var"]
                    e["value"] = to_r(ce["value"])
                    e["method"] = ce["method"]
                    lognum += list(ce.get("consts") or [])
                for e in events:
                    e["time"] = replace_symbols(vs, [f"exp({l})" for l in Ls], e["time"])
            initial_box["v"] = from_py(lc["ic"][0])
            if cond_initial_box["v"] is not None:
                cond_initial_box["v"] = [from_py(x) for x in lc["ic"][1:]]
            ab = get_symbols([x for m in lc["map"] for x in (m["a"], m["b"])])
            chart_coords = [s for s in ab if s not in set(states) | set(fixed) | set(vs)]
            parameters = list(dict.fromkeys(replace_symbols(vs, Ls, parameters) + chart_coords))
            fixed = replace_symbols(vs, Ls, fixed) if fixed else fixed
            if lognum:
                fixed = list(dict.fromkeys(fixed + [str(x) for x in lognum]))
            if vs:
                log_arg = {"v": vs, "L": Ls, "map": lc["map"]}
    if events:
        if any(e.get("root") is not None for e in events):
            raise ValueError("root events are not supported; give the event a time.")
        sym_time = [e["time"] for e in events if not is_number(e["time"])]
        if sym_time and not (method == "observability" and sym_engine == "modular"):
            raise ValueError(f"an event time given in parameters "
                             f"({', '.join(dict.fromkeys(sym_time))}) needs the modular engine.")

    if method == "observability":
        initial = initial_box["v"]
        cond_initial = cond_initial_box["v"]
        extra_syms = []
        if initial is not None:
            extra_syms += get_symbols(list(initial.values()))
        if events:
            extra_syms += get_symbols([e["value"] for e in events] + [e["time"] for e in events])
        for x in (cond_initial or []):
            if x is not None:
                extra_syms += get_symbols(list(x.values()))
        extra_syms += trafo_syms
        symbols = list(dict.fromkeys(states + g_names() + get_symbols(list(fdyn.values()) + g_char()) +
                                     extra_syms))
        const_states = [k for k, r in fdyn.items() if eval_numeric(r) == 0]
        has_net = reactions is not None
        equil_zero = []
        if equilibrate and not has_net:
            equil_zero = _ode_zero_states(fdyn, forcings)
        if equilibrate and has_net:
            equil_zero = [s for s in equil_zero_states(reactions["smatrix"], reactions["species"],
                                                       reactions["rates"], forcings) if s in states]
        held_state_params = {}
        if equilibrate and not reduce_cq and has_net and totals:
            avoid_fv = list(dict.fromkeys(equil_zero + forcings +
                                          free_exponent_bases(list(fdyn.values()) + g_char(), list(fdyn))))
            dec = pivot_decomposition(totals, states, parameters, avoid=avoid_fv, prefer=free_initial)
            piv = [p for p in dec["pivots"] if p is not None and p in states
                   and p not in set(equil_zero) | set(forcings)]
            held_state_params = {p: p for p in piv}
        res = resolve_conditions(conditions, events, initial, symbols, states, const_states,
                                 forcings, equilibrate=equilibrate, cond_subs=cond_subs,
                                 cond_initial=cond_initial, n_cond_obs=n_cond_obs)
        seg_obs = [g_lines(c - 1) for c in res["chain_of"]] if g_per_cond else None
        cond_zero = []
        if equilibrate and has_net:
            from ..par import process_map
            subs_z = []
            for i in range(len(res["subs"])):
                sub = dict(res["subs"][i])
                sub.update(res["rest_subs"][i])
                subs_z.append({k: v for k, v in sub.items() if k not in set(forcings)})
            keys = [";".join(f"{k}={v}" for k, v in s.items()) for s in subs_z]
            uk = [i for i, k in enumerate(keys) if k not in keys[:i]]

            def zero_one(i):
                sub = subs_z[i]
                rates = reactions["rates"]
                if sub:
                    rates = replace_symbols(list(sub), list(sub.values()), list(rates))
                z = equil_zero_states(reactions["smatrix"], reactions["species"], rates, forcings)
                return [s for s in z if s in states and s not in equil_zero]
            nzc = min(max(1, int(cores)), len(uk))
            zs = process_map(zero_one, uk, nzc)
            cond_zero = [zs[uk.index(keys.index(k))] for k in keys]
        model_syms = get_symbols(list(fdyn.values()) + g_char())
        pinned_ic = list(dict.fromkeys((states if equilibrate else ic_names) +
                                       event_pinned_states(events, conditions) +
                                       grid_pinned_states(conditions, states) +
                                      [s for s in forcings + fixed + const_states if s in states]))
        grid_pars = []
        if conditions is not None:
            grid_pars = [c for c in conditions.colnames() if c not in set(states)
                         and conditions.is_numeric(c) and c in set(model_syms)]
        pinned_par = list(dict.fromkeys(list(trafo_subs) + [k for cs in (cond_subs or []) for k in cs] +
                                       [s for s in fixed if s not in set(states)] + grid_pars))
        codim_spec = len(pinned_ic) + len([p for p in pinned_par if p])
        if sym_engine == "symbolic":
            if equilibrate:
                raise ValueError('engine="symbolic" does not support `equilibrate`.')
            if res["n_gaps"] > 0:
                raise ValueError('engine="symbolic" handles single-segment conditions only.')
            sr = observability_sympy_multi(
                model=eqn_lines(fdyn), observation=g_lines(), condition_subs=res["subs"],
                condition_ic0=res["ic0"], condition_obs=seg_obs, condition_rest_subs=res["rest_subs"],
                fixed=fixed or None, parameters=parameters or None, inputs=forcings or None)
            if not sr.get("ok"):
                raise ValueError('engine="symbolic": ' +
                                 str(sr.get("why") or "could not build the symbolic system."))
            sr = dict(sr, method="observability", engine="symbolic",
                      lie_order_used=int(sr["lie_order"]), conditions=int(res["n_conditions"]),
                      segments=int(res["n_conditions"]), gap_order_used=0)
            from .observability.proof import prove_symbolic
            prove_symbolic(sr, sr.pop("systems"))
            dirs = sr["non_identifiable"]
            if sr.get("rank_proven") is None and dirs and all(d.get("exact") for d in dirs) and \
                    len(dirs) == sr["dim"] - sr["rank"]:
                sr["rank_proven"], sr["rank_proof"] = len(dirs), "proven symmetries"
            sr["non_identifiable"] = relabel_directions(sr["non_identifiable"])
            return deliver(sr, method)
        if off is not None:
            multi = {"ok": False, "nonrational": [f"{off} needs a positive argument (`positive`)"]}
        else:
            multi = compile_observability_tape_multi(
                model=eqn_lines(fdyn), observation=g_lines(), condition_subs=res["subs"],
                condition_ic0=res["ic0"], condition_obs=seg_obs, condition_rest_subs=res["rest_subs"],
                fixed=fixed or None, parameters=parameters or None, equilibrate=equilibrate,
                seg_equilibrate=list(res["seg_equil"]), forcings=forcings or None,
                condition_events=res["seg_events"], condition_t0_events=res["events0"],
                condition_times=res["times"] if res["n_gaps"] > 0 else None,
                keep_coords=chart_coords or None, joint_steady_state=equilibrate,
                joint_fixed_states=equil_zero if equilibrate and equil_zero else None,
                held_state_params=held_state_params if equilibrate and held_state_params else None)
        if not multi.get("ok"):
            if multi.get("why") is not None:
                raise ValueError(f"{multi['why']}.")
            raise ValueError(
                "the observability engine requires right-hand sides, observables and initial "
                "conditions built from +, -, *, /, integer powers, exp(), b^x, trigonometric, "
                "hyperbolic and inverse trigonometric and hyperbolic functions, free power "
                "exponents x^n, and log() and fractional powers of arguments that are positive "
                "for the coordinates declared `positive`; observables may be a*log(h) + offset "
                "with a number a; anything else is not rational.\n  " +
                "\n  ".join(str(x) for x in (multi.get("nonrational") or [])) +
                "\nUse the symbolic engine for other functions.")

        def oam():
            return observability_analytic_multi(
                multi, closed_form=reconstruct, cores=cores, equil_zero_states=equil_zero,
                cond_zero_states=cond_zero, t0events=res["events0"], n_conditions=res["n_conditions"],
                chain_of=res["chain_of"], n_gaps=res["n_gaps"], implicit_steady_state=equilibrate,
                control=control, verify=verify, codim_spec=codim_spec)
        out = oam()
        if isinstance(out, dict) and out.get("rerun_blocks"):
            with state.options(stacked_first=False):
                out = oam()
        out = log_param_back(out, multi.get("log_params"))
        if out is None:
            if equilibrate:
                raise ValueError("the implicit steady-state path could not be "
                                 "evaluated (e.g. a singular resting Jacobian from an unreduced "
                                 "conserved moiety). Reduce conserved moieties (`reduce_cq`) or "
                                 "supply an explicit steady state through `trafo`.")
            raise ValueError("the observability kernel could not be evaluated at "
                             "any generic point (a denominator vanishes at every point tried).")
        out["non_identifiable"] = relabel_directions(out["non_identifiable"])
        if verify and isinstance(out.get("verification"), dict) and out["verification"].get("ok") is False:
            warnings.warn("`verify`: the rank still grows past the Lie or gap order used; "
                          "the directions may be over-reported; "
                          f"inspect info['verification'] ({out['verification'].get('reason')}).", stacklevel=2)
        return deliver(out, method)

    fixed_scal = list(dict.fromkeys(fixed + event_pinned_states(events, conditions) +
                                    grid_pinned_states(conditions, states)))
    syms = list(dict.fromkeys(states + g_names() + get_symbols(list(fdyn.values()) + g_char())))
    multi_cond = (conditions is not None and conditions.nrow > 1) or len(cond_subs or []) > 1 or n_cond_obs > 1
    if multi_cond:
        pc = percond_lines(fdyn, gset, g_per_cond, conditions, cond_subs, syms)
        r = scaling_symmetries_multi(per_cond_model=[x["f"] for x in pc],
                                      per_cond_obs=[x["g"] for x in pc], inputs=forcings or None,
                                      fixed=fixed_scal or None, logs=True)
    else:
        r = scaling_symmetries_single(model=eqn_lines(fdyn), observation=g_lines(),
                                       inputs=forcings or None, fixed=fixed_scal,
                                       parameters=parameters or None, verbose=verbose)
    for d in r.get("non_identifiable") or []:
        d["exact"] = not r.get("skipped")
    if r.get("skipped"):
        session.note("limits", f"scalings: {r['skipped']} term(s) without polynomial form left out of the "
                               "scaling conditions; a scaling they forbid may be reported")
    return deliver(r, "scaling")

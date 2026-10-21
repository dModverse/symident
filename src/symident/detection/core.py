"""The detection pipeline: model preprocessing, coordinate trafos, conditions and the engines."""

import re
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
    sym_sort,
    tokens,
)
from ..sym.charts import log_arg_chart, log_arg_event, sign_chart
from ..sym.parse import model_sympify, to_r
from ..sym.scaling import scaling_symmetries_multi
from ..sym.switching import hybrid_model
from ..sym.symbolic import observability_sympy_multi
from ..sym.tape import compile_observability_tape_multi
from .classify import relabel_directions
from .conditions import (
    Grid,
    as_events,
    at_time,
    event_pinned_states,
    grid_pinned_states,
    initial_value_split,
    log_param_back,
    num_str,
    rename_result,
    resolve_conditions,
    scaling_component,
    start_time,
)
from .cq import check_conserved, equil_zero_states, pivot_decomposition, substitute_cq
from .finalize import finalize, log_arg_back
from .gauge import gauge_suggest, gauged_detection
from .observability import observability_analytic_multi


def _clean(e):
    if e is None:
        return None
    return {str(k): to_r(v) for k, v in dict(e).items()}


def _sub_or_none(fn, e):
    return None if e is None else fn(e)


def _at_start(ic, start):
    """Initial values `ic` with the clock read at the start."""
    return None if ic is None else {k: at_time(v, start) for k, v in ic.items()}


def _clock_texts(fdyn, gset, cond_subs, conditions):
    """The right-hand sides, observables, per-condition trafo values and grid strings: where the model reads time."""
    texts = list(fdyn.values()) + [v for g in gset if g for v in g.values()]
    texts += [v for cs in cond_subs or [] if cs for v in cs.values()]
    if conditions is not None:
        texts += [v for vals in conditions.cols.values() for v in vals if isinstance(v, str)]
    return texts


def _clock_offset(fdyn, gset, cond_subs, conditions, tt0):
    """The model read at time + tt0, for a clock that counts from zero at the start tt0."""
    sub = subst(["time"], [f"time + {num_str(tt0)}" if tt0 > 0 else f"time - {num_str(-tt0)}"])
    fdyn = {k: v if k == "time" else sub(v) for k, v in fdyn.items()}
    gset = [_sub_or_none(sub, g) for g in gset]
    cond_subs = None if cond_subs is None else [{k: sub(v) for k, v in (cs or {}).items()} for cs in cond_subs]
    if conditions is not None:
        conditions = Grid(conditions.rows, {c: [sub(v) if isinstance(v, str) else v for v in vals]
                                            for c, vals in conditions.cols.items()})
    return fdyn, gset, cond_subs, conditions


def _zeroed(before, after):
    """The right-hand sides `after`, "0" where a substitution of `before` makes one vanish identically."""
    import sympy as spy
    out = {}
    for k, v in after.items():
        if v != before.get(k) and str(v) != "0":
            try:
                v = "0" if spy.expand(model_sympify(v)) == 0 else v
            except SYMPY_FAILS as exc:
                sympy_fail(exc)
        out[k] = v
    return out


def _rest_idle_states(fdyn, forcings, zero=()):
    """Non-input states whose right-hand side, nonzero as written, vanishes with the inputs and `zero` at zero."""
    import sympy as spy

    from ..sym.parse import _make_local_parse
    _, parse = _make_local_parse([f"{k} = {v}" for k, v in fdyn.items()])
    at0 = {spy.Symbol(u): 0 for u in list(forcings) + list(zero)}
    out = []
    for k, v in fdyn.items():
        if k in set(forcings) or k in set(zero):
            continue
        e = parse(str(v).replace("^", "**"))
        if e != 0 and spy.simplify(e.xreplace(at0)) == 0:
            out.append(k)
    return out


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

    from ..sym.domain import is_positive
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
            # coordinates are real; the declared ones and expressions positive
            if not is_positive(u, positive):
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
    for key in ("parameters", "fixed", "forcings", "free_initial", "unknown_inputs", "known_inputs"):
        names += [str(x) for x in kw.get(key) or []]
    trafo = kw.get("trafo")
    for t in trafo if isinstance(trafo, (list, tuple)) else [trafo]:
        names += list(dict(t or {}))
    names += [str(e.get("var")) for e in kw.get("events") or []]
    if "pi" in names:
        raise ValueError("pi is the number in the equations and cannot name a state, parameter or input")


_NAME = re.compile(r"[A-Za-z.][A-Za-z0-9._]*")


def _check_input_starts(kw):
    """ValueError for an initial value of an input in `trafo` or a condition column: inputs start at zero."""
    trafo = kw.get("trafo")
    grid = Grid.from_any(kw.get("conditions"))
    for u in [str(u) for u in kw.get("forcings") or []]:
        vals = [num_str(dict(t or {}).get(u, "0")).strip()
                for t in (trafo if isinstance(trafo, (list, tuple)) else [trafo])]
        if grid is not None and u in grid.colnames():
            vals += [grid.cell(i, u) for i in range(grid.nrow)]
        bad = [v for v in vals if v not in ("0", u)]
        if bad:
            raise ValueError(f"the input {u} starts at zero and cannot take the initial value {bad[0]}; an event "
                             f"sets it, such as {{'var': '{u}', 'time': 0, 'value': '{bad[0]}', "
                             "'method': 'replace'}")


def _check_declared(kw, known):
    """ValueError for a declared name that the model does not hold; a warning for such a name in `positive`."""
    f = {str(k): str(v) for k, v in dict(kw.get("f") or {}).items()}
    states = set(f)
    eq = set(get_symbols(_model_texts(kw)))
    trafo = kw.get("trafo")
    trafos = [dict(t or {}) for t in (trafo if isinstance(trafo, (list, tuple)) else [trafo])]
    events = as_events(kw.get("events"))
    grid = Grid.from_any(kw.get("conditions"))
    totals = dict(kw.get("totals") or {})
    texts = [str(v) for t in trafos for v in t.values()] + [e["value"] for e in events] + \
        [e["time"] for e in events] + [str(v) for v in totals.values()]
    if grid is not None:
        texts += [v for vals in grid.cols.values() for v in vals if isinstance(v, str)]
    names = states | eq | {str(k) for t in trafos for k in t} | set(get_symbols(texts))
    if kw.get("reduce_cq"):
        names |= set(totals)
    unknown = {str(w) for w in kw.get("unknown_inputs") or []}
    checks = [
        ([u for u in known if u in states], "known input {} has an equation in `odes`; a known input is a "
                                            "function of time without one"),
        ([u for u in known if u not in eq], "known input {} enters no right-hand side or observable"),
        ([str(u) for u in kw.get("forcings") or [] if str(u) not in states],
         "input {} is not a state of `odes`; an input is a state that events set"),
        ([e["var"] for e in events if e["var"] not in states and e["var"] not in unknown],
         "an event changes {}, which is not a state of `odes`"),
        ([c for c in (grid.colnames() if grid is not None else []) if c not in names],
         "the condition column {} names no symbol of the model"),
        ([str(x) for x in kw.get("fixed") or [] if str(x) not in names], "`fixed` names {}, which is no symbol of "
                                                                        "the model"),
        ([str(x) for x in kw.get("parameters") or [] if str(x) not in names],
         "`parameters` names {}, which is no symbol of the model"),
    ]
    for bad, why in checks:
        if bad:
            raise ValueError(why.format(bad[0]))
    pos = kw.get("positive")
    if isinstance(pos, (list, tuple)):
        bad = [str(p) for p in pos if _NAME.fullmatch(str(p)) and str(p) not in names]
        if bad:
            warnings.warn(f"`positive` names {', '.join(bad)}, which {'is' if len(bad) == 1 else 'are'} no "
                          "symbol of the model; ignored", stacklevel=3)


def _model_texts(kw):
    """The right-hand sides and observables of the detection arguments `kw`."""
    gs = kw.get("g")
    return [str(v) for v in dict(kw.get("f") or {}).values()] + \
        [str(v) for g in (gs if isinstance(gs, (list, tuple)) else [gs]) for v in dict(g or {}).values()]


def _check_unknown_inputs(kw, hybrid):
    """ValueError for unknown inputs on a model or setting they do not support yet."""
    unknown = [str(w) for w in kw.get("unknown_inputs") or []]
    if not unknown:
        return
    f = dict(kw.get("f") or {})
    texts = _model_texts(kw)
    grid = Grid.from_any(kw.get("conditions"))
    if grid is not None:
        texts += [str(v) for c in grid.colnames() if grid.is_character(c) for v in grid.cols[c]]
    used = set(get_symbols(texts))
    for w in unknown:
        if w in f:
            raise ValueError(f"unknown input {w} has an equation in `odes`; an unknown input is a "
                             "function of time without one")
        if w in set(kw.get("forcings") or []) | set(kw.get("fixed") or []) | set(kw.get("parameters") or []):
            raise ValueError(f"unknown input {w} is also listed in `inputs`, `fixed` or `parameters`")
        if w not in used:
            raise ValueError(f"unknown input {w} enters no right-hand side or observable")
    for ev in as_events(kw.get("events")) or []:
        if str(ev.get("var")) in unknown:
            raise ValueError(f"an event changes the unknown input {ev['var']}; an unknown input has no "
                             "events, since it is any function of time")
    why = None
    if hybrid:
        why = "switches, since a direction would need one input change on both sides of every surface"
    elif kw.get("sym_engine") == "symbolic":
        why = 'engine="symbolic"'
    elif kw.get("trafo") is not None and any(w in dict(t or {}) for t in (
            kw["trafo"] if isinstance(kw["trafo"], (list, tuple)) else [kw["trafo"]]) for w in unknown):
        why = "a `trafo` entry for an unknown input"
    if why is not None:
        raise ValueError(f"unknown inputs are not supported together with {why} yet")


def _input_jets(f, unknown, order, taken):
    """(right-hand sides, jet names) with each input a chain of `order` jet states w, w_d1, ..., the last constant."""
    jets = {}
    for w in unknown:
        names = [w] + [f"{w}_d{k}" for k in range(1, order)]
        clash = [n for n in names[1:] if n in taken]
        if clash:
            raise ValueError(f"{clash[0]} names a symbol of the model and a time derivative of the "
                             f"unknown input {w}; rename it")
        jets[w] = names
    chain = {}
    for names in jets.values():
        for k, x in enumerate(names):
            chain[x] = names[k + 1] if k + 1 < len(names) else "0"
    return {**chain, **{k: v for k, v in f.items() if k not in chain}}, jets


def _jet_order(kw):
    """The jets per unknown input: twice the model's symbols plus the guard margin."""
    n = len(set(get_symbols(_model_texts(kw))) | set(dict(kw.get("f") or {}))) - len(set(kw.get("fixed") or []))
    return 2 * max(1, n) + int(state.env_value("VERIFY_MARGIN", "6")) + 4


def _with_input_jets(kw):
    """The detection with unknown inputs as jet chains, lengthened until exact at every Lie order taken."""
    unknown = [str(w) for w in kw.get("unknown_inputs") or []]
    if not unknown:
        return detection_with_primes(kw)
    order = int(dict(kw.get("control") or {}).get("input_jet_order") or 0) or _jet_order(kw)
    while True:
        res = detection_with_primes(dict(kw, unknown_inputs=unknown, jet_order=order))
        ui = (res.get("info") or {}).get("unknown_inputs") if isinstance(res, dict) else None
        if not ui or not ui.get("short") or dict(kw.get("control") or {}).get("input_jet_order"):
            break
        order *= 2
    if ui and ui.get("short"):
        session.note("limits", f"unknown inputs: the {ui['jet_order']} jets of each input are exact up to "
                               f"Lie order {ui['exact_order']}, below the orders the analysis took; the rank "
                               f"holds at Lie order {ui['exact_order']}")
    return res


def _gauged(kw):
    """The gauged detection on the arguments `kw`, unknown inputs included."""
    from ..gfp import all_primes
    gp = kw["gauge_preference"]
    pref = None if gp is None else [gp] if isinstance(gp, str) else list(gp)
    res = gauged_detection({k: v for k, v in kw.items() if k != "gauge_preference"}, pref, kw.get("fixed"),
                           kw.get("reconstruct"), kw.get("verbose"))
    ses = state.get("session")
    if ses is not None and ses.depth == 1:
        ses.primes = all_primes()
    return res


def _with_known_inputs(kw, known):
    """The arguments with each known input a jet chain whose initial values are fixed symbols."""
    taken = set(get_symbols(_model_texts(kw))) | set(dict(kw.get("f") or {}))
    f, jets = _input_jets(dict(kw["f"]), known, _jet_order(kw), taken)
    start = {x: f"{x}_0" for names in jets.values() for x in names}
    clash = [v for v in start.values() if v in taken]
    if clash:
        raise ValueError(f"{clash[0]} names a symbol of the model and the start value of a known input; rename it")
    fixed = list(kw.get("fixed") or []) + list(start.values())
    if kw.get("equilibrate"):
        # zero at rest like every input, the jets known from the start on
        events = as_events(kw.get("events")) + [{"var": x, "time": "0", "value": v, "method": "replace",
                                                 "root": None} for x, v in start.items()]
        return dict(kw, f=f, fixed=fixed, events=events, forcings=list(kw.get("forcings") or []) + list(start))
    trafo = kw.get("trafo")
    if isinstance(trafo, (list, tuple)):
        trafo = [dict(t or {}, **start) for t in trafo]
    else:
        trafo = dict(trafo or {}, **start)
    return dict(kw, f=f, trafo=trafo, fixed=fixed)


def symmetry_detection(*args, **kw):
    """The detection on plain inputs (see _symmetry_detection); a model with switches that
    the declared signs leave open goes to the hybrid analysis."""
    import inspect
    known = [str(u) for u in kw.pop("known_inputs", None) or []]
    kw = dict(inspect.signature(_symmetry_detection).bind(*args, **kw).arguments)
    if state.get("session") is None:
        _check_declared(kw, known)
    if known:
        kw = _with_known_inputs(kw, known)
    _check_steps(kw)
    _check_names(kw)
    _check_input_starts(kw)
    regimes = kw.pop("regimes", None)
    hy = None
    if kw.get("f") is not None:
        hy = hybrid_model({str(k): str(v) for k, v in dict(kw["f"]).items()}, kw.get("g"),
                          kw.get("positive", True), events=kw.get("events"),
                          inputs=kw.get("forcings") or (), taken=kw.get("parameters") or ())
    _check_unknown_inputs(kw, hy is not None and not isinstance(hy, dict))
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
            elif kw.get("gauge_preference", False) is not False and not kw.get("scalings_only"):
                res = _gauged(kw)
            else:
                res = _with_input_jets(kw)
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


def _past_premature_order(oam, out, tries=3):
    """The analysis repeated from past the Lie order at which the saturation guard saw the rank grow."""
    plateau = max(1, int(state.env_value("LIEPLATEAU", "3")))
    for _ in range(tries):
        ver = out.get("verification") if isinstance(out, dict) else None
        grow = (ver or {}).get("grow_at")
        if not isinstance(ver, dict) or ver.get("ok") is not False or grow is None:
            return out
        with state.options(stacked_first=False, nt_floor=int(grow) + plateau):
            nxt = oam()
        if not isinstance(nxt, dict):
            return out
        out = nxt
    return out


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
    elif (info.get("unknown_inputs") or {}).get("short"):
        why = "the jet chains of the unknown inputs end below the Lie order reached"
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
        return {"applies": False, "why": "the error bound does not cover a start at rest (equilibrate)"}
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
    inputs = {str(u) for u in kw.get("forcings") or []}
    for tr in trafos or [None]:
        tr = {str(k): str(v) for k, v in dict(tr or {}).items()}
        sub = {k: v for k, v in tr.items() if k not in f}
        x0 = {x: "0" if x in inputs else tr.get(x, x) for x in f}
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


def _log_back(expr, la):
    """`expr` with the log coordinates of the chart `la` written in the coordinates they replace."""
    if la is None:
        return str(expr)
    import sympy as spy
    at = {spy.Symbol(m["L"]): spy.log(spy.Symbol(m["v"])) for m in la["map"]}
    return to_r(model_sympify(str(expr)).xreplace(at))


def _scalings_without_jets(dirs, jets):
    """Scalings on the model's coordinates, the weights of the unknown inputs as their input change."""
    every = {x for names in jets.values() for x in names}
    out = []
    for d in dirs:
        vec = {k: v for k, v in (d.get("vector") or {}).items() if k not in every}
        if not vec:
            continue
        out.append(dict(d, vector=vec, support=sym_sort(list(vec)),
                        input_vector={w: scaling_component(d["vector"].get(w, "0"), w) for w in jets},
                        input_form={k: scaling_component(v, k) for k, v in vec.items()}))
    return out


def _resolved_scalings(fdyn, gset, g_per_cond, res, inputs, fixed, jets, multi, held):
    """The scalings of every segment's equations with its initial values, event values and times held, on the
    coordinates of the compiled tapes `multi` (None: the free or `held` states and the parameters), with the jet
    weights."""
    import sympy as spy
    model, obs, xm, xo = [], [], [], []
    for i, subs in enumerate(res["subs"]):
        sub = subst(list(subs), list(subs.values()))
        first = res["pos_in_chain"][i] == 1
        evs = res["events0"][i] if first else res["seg_events"][i]
        model.append(eqn_lines(sub(dict(fdyn))))
        g = gset[res["chain_of"][i] - 1] if g_per_cond else gset[0]
        obs.append(eqn_lines(sub(dict(g or {}))) or [])
        xm.append([f"{k} = {sub(v)}" for k, v in (res["ic0"][i] if first else {}).items()] +
                  [f"{e['var']} = {sub(e['value'])}" for e in evs if e["method"] != "multiply"])
        xo.append([f"_ = {sub(e['value'])}" for e in evs if e["method"] == "multiply"] +
                  ([f"_ = {res['times'][i]}"] if get_symbols([res["times"][i]]) else []))
    r = scaling_symmetries_multi(per_cond_model=model, per_cond_obs=obs, inputs=inputs or None,
                                 fixed=fixed or None, logs=True, extra_model=xm, extra_obs=xo)
    if r.get("skipped"):
        session.note("limits", f"scalings: {r['skipped']} term(s) without polynomial form left out of the "
                               "scaling conditions; a scaling they forbid may be reported")
    every = {x for names in jets.values() for x in names}
    if multi is not None:
        drop = set(multi.get("z_state_names") or []) if multi.get("joint_steady_state") else set()
        drop |= set(multi.get("exp_atoms") or []) | set(multi.get("recast_atom_names") or [])
        back = {str(e["X"]): str(e["theta"]) for e in multi.get("log_params") or []}
        coords = [back.get(z, z) for z in multi["znames"] if z not in drop | set(multi.get("nuisance") or [])]
    else:
        # at rest only the held states keep a value of their own
        free = set(held) if held is not None else {
            x for i, ic in enumerate(res["ic0"]) if res["pos_in_chain"][i] == 1
            for x in fdyn if x not in ic or x in get_symbols([ic[x]])}
        coords = [z for z in r["coordinates"] if (z in free or z not in fdyn) and z not in set(fixed) | every]
    rows, dirs = [], []
    for d in r.get("non_identifiable") or []:
        vec = {k: v for k, v in d["vector"].items() if (k in coords or k in every) and str(v) != "0"}
        row = [spy.Rational(str(vec.get(c, 0))) for c in coords]
        if spy.Matrix(rows + [row]).rank() == len(rows):
            continue
        rows.append(row)
        dirs.append(dict(d, vector=vec, support=sym_sort(list(vec)), exact=not r.get("skipped")))
    return {"method": "scaling", "non_identifiable": dirs, "coordinates": coords}


def _mark_nuisance(multi, spec):
    """The nuisance coordinates of `spec` marked in the compiled tapes: states, kept, tops, exact order, prolonged."""
    every = set(spec["states"])
    multi["nuisance"] = [z for z in multi["znames"] if z in every]
    multi["nuisance_tops"] = list(spec.get("tops") or [])
    multi["nuisance_kept"] = list(spec.get("kept") or [])
    multi["nuisance_order"] = int(spec["exact_order"])
    multi["nuisance_prolonged"] = bool(spec.get("prolonged"))
    multi["nuisance_jet_states"] = list(spec.get("jet_states") or [])
    cols = [i for i, z in enumerate(multi["znames"]) if z in every]
    for t in multi["tapes"]:
        r = list(t.get("lie_reach") or [])
        for i in cols:
            if i < len(r):
                r[i] = -1
        t["lie_reach"] = r


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
                       regimes=None, unknown_inputs=None, jet_order=None, nuisance=None):
    """The result dictionary of the detection on plain inputs."""
    control = {**reconst_control(), **dict(control or {})}
    equilibrate = bool(equilibrate)
    method = "scaling" if scalings_only else "observability"
    if gauge_preference is not False and scalings_only:
        warnings.warn("`gauge` applies only without `scalings_only` and is ignored.", stacklevel=2)
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
        if jets:
            every = {x for names in jets.values() for x in names}
            for s in res["symmetries"]:
                # the input change of a proven field; a kernel vector holds it only at the start
                lift = s.get("state_components") or {}
                ch = lift if any(w in lift for w in jets) else (s.get("input_change") or {})
                s["input_change"] = {w: _log_back(ch.get(w, "0"), log_arg) for w in jets} \
                    if s.get("level") == "exact" else None
                if lift:
                    s["state_components"] = {k: v for k, v in lift.items() if k not in every} or None
            nu = raw.get("nuisance") or {}
            res["info"]["unknown_inputs"] = {
                "names": list(jets), "jets": jets, "jet_order": int(jet_order),
                "exact_order": nu.get("exact_order"), "short": bool(nu.get("short")),
                "nuisance_rank": None if not nu else len(nu["names"]) - int(nu["pinned"])}
        elif nuisance and raw.get("nuisance"):
            nu = raw["nuisance"]
            res["info"]["nuisance"] = {"exact_order": nu["exact_order"], "short": bool(nu["short"]),
                                       "rank": len(nu["names"]) - int(nu["pinned"])}
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
    g_per_cond = isinstance(g, (list, tuple))
    gset = [_clean(x) for x in g] if g_per_cond else [_clean(g)]
    if g_per_cond and not gset:
        raise ValueError("`observables` is an empty list; give one dict per condition.")
    jets = {}
    if unknown_inputs:
        taken = set(get_symbols(list(fdyn.values()) + [v for x in gset if x for v in x.values()])) | \
            set(fdyn) | set(fixed) | set(parameters or [])
        fdyn, jets = _input_jets(fdyn, [str(w) for w in unknown_inputs], int(jet_order), taken)
    jet_start = {}
    if jets and equilibrate:
        # zero at rest like every input, the jets free from the start on
        jet_start = {x: f"{x}_0" for names in jets.values() for x in names}
        clash = [v for v in jet_start.values() if v in taken]
        if clash:
            raise ValueError(f"{clash[0]} names a symbol of the model and the start value of an unknown "
                             "input; rename it")
        forcings = forcings + list(jet_start)
        events = events + [{"var": x, "time": "0", "value": v, "method": "replace", "root": None}
                           for x, v in jet_start.items()]
    states = list(fdyn)

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
        zero = list(dict.fromkeys(zero + _ode_zero_states(fdyn, forcings)))
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
        if free_initial and not (equilibrate and not reduce_cq):
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
                known |= set(get_symbols([v for vals in conditions.cols.values() for v in vals
                                          if isinstance(v, str)]))
            trafo_hit = [k for k in trafo_subs if k in known]
        fdyn = _zeroed(fdyn, sub(fdyn))
        subst_model(sub)
        if trafo_subs and conditions is not None:
            conditions = Grid(conditions.rows, {
                c: [replace_symbols(list(trafo_subs), list(trafo_subs.values()), v) if isinstance(v, str) else v
                    for v in vals] for c, vals in conditions.cols.items()})
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
        over = [k for k in list(trafo_subs) + [k for cs in cond_subs or [] for k in cs]
                if k in conditions.colnames() and k not in states]
        if over:
            warnings.warn(f"`trafo` replaces {', '.join(dict.fromkeys(over))}, which a condition column of that name "
                          "sets per condition; the column has no effect", stacklevel=2)
    if "time" not in states:
        tt0 = start_time(events, conditions, cond_subs, len(gset) if g_per_cond else 1)
        start = num_str(tt0)
        # event values read the clock at their event time, initial values at the start
        for e in events:
            if e.get("root") is None:
                e["value"] = at_time(e["value"], e["time"])
        initial_box["v"] = _at_start(initial_box["v"], start)
        if cond_initial_box["v"] is not None:
            cond_initial_box["v"] = [_at_start(x, start) for x in cond_initial_box["v"]]
        if "time" in get_symbols(_clock_texts(fdyn, gset, cond_subs, conditions)):
            fdyn = dict(fdyn, time="1")
            states = list(fdyn)
            if equilibrate:
                # an input: zero at rest, counting from the start
                forcings = forcings + ["time"]
                if tt0 != 0:
                    fdyn, gset, cond_subs, conditions = _clock_offset(fdyn, gset, cond_subs, conditions, tt0)
            elif trafo_list:
                cond_initial_box["v"] = [dict(x or {}, time=start) for x in cond_initial_box["v"]]
            else:
                initial_box["v"] = dict(initial_box["v"] or {}, time=start)
    if conditions is not None:
        keep = [c for c in conditions.colnames() if c not in set(states)]
        chr_ = [c for c in keep if conditions.is_character(c)]
        hit = [s for s in get_symbols([str(v) for c in chr_ for v in conditions.cols[c]])
               if s in set(states) and s not in jets] if chr_ else []
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
        # inputs are zero at rest whatever their initial value
        told = [x for x in ic_names if x not in set(forcings)]
        if told:
            warnings.warn("`equilibrate` solves the steady state from f = 0; the "
                          f"initial condition(s) for {', '.join(told)} in `trafo` are ignored.", stacklevel=2)
        initial_box["v"] = None
        if cond_initial_box["v"] is not None:
            cond_initial_box["v"] = [None for _ in cond_initial_box["v"]]
    if totals and (reduce_cq or equilibrate):
        check_conserved(totals, fdyn)
    cq_totals = []
    if reduce_cq and totals:
        # a species with an initial value stays a state where another species of its moiety can be eliminated
        told = [] if equilibrate else ic_names
        avoid_cq = free_exponent_bases(list(fdyn.values()) + g_char(), list(fdyn)) + told
        cq = substitute_cq(totals, fdyn, list(fdyn), parameters, avoid=avoid_cq)
        clash_cq = [s for s in cq["elim_states"] if s in set(told)]
        if clash_cq:
            warnings.warn("a `trafo` initial condition was supplied for "
                          f"{', '.join(clash_cq)}, which `reduce_cq` would eliminate through a conserved "
                          "quantity, discarding that relation. Forcing reduce_cq=False so the supplied "
                          "initial values are used as given.", stacklevel=2)
            reduce_cq = False
            settings["reduce_cq"] = False
    if reduce_cq and totals:
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
            fdyn = _zeroed(fdyn, sub_cq(fdyn))
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
    if any(re.search(r"\b(abs|sign|max|min|pmax|pmin|ifelse|Heaviside)\s*\(", x)
           for x in list(fdyn.values()) + g_char()):
        sc = sign_chart(dict(fdyn), [dict(x or {}) for x in gset],
                          True if positive is True else ([] if positive is False else list(positive)))
        if sc.get("why") is not None:
            raise ValueError(f"{sc['why']}.")
        fdyn = {k: to_r(v) for k, v in sc["f"].items()}
        gset = [({k: to_r(v) for k, v in x.items()} or None) for x in sc["g"]]
    off = _off_domain(list(fdyn.values()) + g_char(), positive, set(states))
    if positive is not False and method != "scaling" and sym_engine != "symbolic":
        ev_ic = [{e["var"]: e["value"]} for e in events if e["method"] in ("replace", "multiply")]
        # condition columns and constant states without events become numbers per condition
        per_cond = set(conditions.colnames()) if conditions is not None else set()
        for cs in (cond_subs or []):
            per_cond |= set(cs) | set(get_symbols(list(cs.values())))
        moved = {e["var"] for e in events}
        per_cond |= {k for k, r in fdyn.items() if eval_numeric(r) == 0 and k not in moved}
        lc = log_arg_chart(dict(fdyn), [dict(x or {}) for x in gset],
                            True if positive is True else list(positive),
                            list(dict.fromkeys(parameters + g_names() + ic_to)),
                            [dict(initial_box["v"] or {})] +
                            [dict(x or {}) for x in (cond_initial_box["v"] or [])] + ev_ic, skip=per_cond)
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
        if sym_time and method == "observability" and sym_engine != "modular":
            raise ValueError(f"an event time given in parameters "
                             f"({', '.join(dict.fromkeys(sym_time))}) needs the modular engine.")

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
    # a constant state set by an event to a value of another state holds that value, not the running state
    while True:
        dep = {e["var"] for e in events or [] if e["var"] in const_states and
               set(get_symbols([str(e["value"])])) & (set(states) - set(const_states))}
        if not dep:
            break
        const_states = [k for k in const_states if k not in dep]
    has_net = reactions is not None
    equil_zero = []
    if equilibrate and not has_net:
        equil_zero = _ode_zero_states(fdyn, forcings)
    if equilibrate and has_net:
        # a signed net rate such as -k*(x - u) rests at zero without a dead producer
        equil_zero = [s for s in dict.fromkeys(
            equil_zero_states(reactions["smatrix"], reactions["species"], reactions["rates"], forcings) +
            _ode_zero_states(fdyn, forcings)) if s in states]
    held_state_params = {}
    if equilibrate and not reduce_cq and totals:
        avoid_fv = list(dict.fromkeys(equil_zero + forcings +
                                      free_exponent_bases(list(fdyn.values()) + g_char(), list(fdyn))))
        dec = pivot_decomposition(totals, states, parameters, avoid=avoid_fv, prefer=free_initial)
        piv = [p for p in dec["pivots"] if p is not None and p in states
               and p not in set(equil_zero) | set(forcings)]
        held_state_params = {p: p for p in piv}
    if equilibrate:
        # a state whose right-hand side vanishes at rest keeps a resting value of its own
        for x in _rest_idle_states(fdyn, forcings, equil_zero):
            held_state_params.setdefault(x, x)
    res = resolve_conditions(conditions, events, initial, symbols, states, const_states,
                             forcings, equilibrate=equilibrate, cond_subs=cond_subs,
                             cond_initial=cond_initial, n_cond_obs=n_cond_obs, renamed=list(jets))
    seg_obs = [g_lines(c - 1) for c in res["chain_of"]] if g_per_cond else None
    cond_zero = []
    if equilibrate and method == "observability":
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
            rhs = {k: replace_symbols(list(sub), list(sub.values()), v) if sub else v for k, v in fdyn.items()}
            z = _ode_zero_states(rhs, forcings)
            if has_net:
                rates = reactions["rates"]
                if sub:
                    rates = replace_symbols(list(sub), list(sub.values()), list(rates))
                z = list(dict.fromkeys(equil_zero_states(reactions["smatrix"], reactions["species"], rates,
                                                         forcings) + z))
            return [s for s in z if s in states and s not in equil_zero]
        nzc = min(max(1, int(cores)), len(uk))
        zs = process_map(zero_one, uk, nzc)
        cond_zero = [zs[uk.index(keys.index(k))] for k in keys]
    model_syms = get_symbols(list(fdyn.values()) + g_char())
    pinned_ic = list(dict.fromkeys((states if equilibrate else ic_names) +
                                   event_pinned_states(events, conditions) +
                                   grid_pinned_states(conditions, [s for s in states if s not in jets]) +
                                  [s for s in forcings + fixed + const_states if s in states]))
    grid_pars = []
    if conditions is not None:
        grid_pars = [c for c in conditions.colnames() if c not in set(states)
                     and conditions.is_numeric(c) and c in set(model_syms)]
    pinned_par = list(dict.fromkeys(list(trafo_subs) + [k for cs in (cond_subs or []) for k in cs] +
                                   [s for s in fixed if s not in set(states)] + grid_pars))
    codim_spec = len(pinned_ic) + len([p for p in pinned_par if p])
    if sym_engine == "symbolic" and method == "observability":
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
            held_state_params=held_state_params if equilibrate and held_state_params else None,
            const_states=const_states, positive=positive)
    if multi.get("ok") and jets:
        # the k-th Lie derivative of an observable holds the input up to its k-th derivative
        nm = jet_start.get
        _mark_nuisance(multi, {"states": [nm(x, x) for names in jets.values() for x in names],
                               "kept": [nm(w, w) for w in jets],
                               "tops": [nm(names[-1], names[-1]) for names in jets.values()],
                               "exact_order": int(jet_order) - 1, "prolonged": not jet_start,
                               "jet_states": list(jet_start)})
    elif multi.get("ok") and nuisance:
        _mark_nuisance(multi, nuisance)
    if method == "scaling":
        r = _resolved_scalings(fdyn, gset, g_per_cond, res, [u for u in forcings if u not in jet_start], fixed, jets,
                               multi if multi.get("ok") else None, held_state_params if equilibrate else None)
        if jets:
            r = dict(r, non_identifiable=_scalings_without_jets(r.get("non_identifiable") or [], jets))
        return deliver(r, "scaling")
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
    out = _past_premature_order(oam, out)
    out = log_param_back(out, multi.get("log_params"))
    if out is None:
        if equilibrate:
            hint = "declare each conserved quantity in `totals`" if reduce_cq else \
                "reduce conserved moieties (`reduce_cq`)"
            raise ValueError("the implicit steady-state path could not be "
                             "evaluated (e.g. a singular resting Jacobian from an unreduced "
                             f"conserved moiety). {hint[0].upper()}{hint[1:]} or "
                             "supply an explicit steady state through `trafo`.")
        raise ValueError("the observability kernel could not be evaluated at "
                         "any generic point (a denominator vanishes at every point tried).")
    out["non_identifiable"] = relabel_directions(out["non_identifiable"])
    if verify and isinstance(out.get("verification"), dict) and out["verification"].get("ok") is False:
        warnings.warn("`verify`: the rank still grows past the Lie or gap order used; "
                      "the directions may be over-reported; "
                      f"inspect info['verification'] ({out['verification'].get('reason')}).", stacklevel=2)
    return deliver(out, method)

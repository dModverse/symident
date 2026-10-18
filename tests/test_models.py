"""The spec models through detection and reduction, checked on properties that need no reference."""

import glob
import os
import warnings

import pytest
import sympy as spy
from spec_util import env_of, kw_of, kwargs_of, load, obj_of, options_of

from symident import state
from symident.detection import symmetry_detection
from symident.install import find_msolve
from symident.reduction import symmetry_reduction
from symident.reduction.core import LEVELS
from symident.sym.parse import model_sympify

HERE = os.path.dirname(__file__)
DET = sorted(glob.glob(os.path.join(HERE, "data", "specs", "detect", "*.json")))
RED = sorted(glob.glob(os.path.join(HERE, "data", "specs", "reduce", "*.json")))
MSOLVE = find_msolve()[0] is not None
NEEDS_MSOLVE = {"test-symmetryDetection_151_symmetryDetection.json"}
# an error the user can act on; anything else is a fault of symident
USER_ERRORS = (ValueError, NotImplementedError)


def cases(paths):
    return [pytest.param(p, id=os.path.basename(p)[:-5]) for p in paths]


def parse(x, positive=False):
    e = model_sympify(str(x))
    return e.xreplace({s: spy.Symbol(str(s), positive=True) for s in e.free_symbols}) if positive else e


def detect_spec(g):
    env = env_of(g)
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with state.options(**options_of(g)):
                return symmetry_detection(**kwargs_of(g["spec"]))
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def plain(spec):
    """Whether a spec is one condition without events, inputs, switches or a per-condition trafo,
    so that a lifted generator can be checked on the equations directly."""
    tr = spec.get("trafo")
    # a coordinate named like a state is its initial value, which the lifted field moves apart
    return (spec.get("positive") in (None, True) and isinstance(tr, dict) and set(spec["f"]) <= set(tr)
            and not spec.get("conditions") and not spec.get("events") and not spec.get("forcings")
            and not spec.get("equilibrate") and not spec.get("reduceCQ") and isinstance(spec.get("g"), dict)
            and "piecewise" not in str(spec.get("f")))


def lifted_identities(spec, d):
    """The residuals of a lifted generator X: X(f_i) - sum_j f_j d(xi_i)/d(x_j), X(g_k), and
    xi_x - X(x0) at the initial values."""
    def sym(k):
        return spy.Symbol(k, positive=True)
    f = {sym(k): parse(v, True) for k, v in spec["f"].items()}
    g = [parse(v, True) for v in spec["g"].values()]
    tr = {sym(k): parse(v, True) for k, v in (spec.get("trafo") or {}).items()}
    xi = {sym(k): parse(v, True) for k, v in dict(d["generator"], **d["state_components"]).items()}
    pars = {k: v for k, v in tr.items() if k not in f}
    f = {x: e.xreplace(pars) for x, e in f.items()}
    g = [e.xreplace(pars) for e in g]

    def X(e):
        return sum(c * spy.diff(e, k) for k, c in xi.items())
    out = [X(f[x]) - sum(f[y] * spy.diff(xi.get(x, 0), y) for y in f) for x in f]
    out += [X(e) for e in g]
    x0 = {x: tr[x] for x in f}
    out += [(xi.get(x, 0) - X(x0[x])).xreplace(x0) for x in f]
    return out


@pytest.mark.parametrize("path", cases(DET))
def test_detection(path):
    g = load(path)
    if os.path.basename(path) in NEEDS_MSOLVE and not MSOLVE:
        pytest.skip("needs msolve")
    try:
        res = detect_spec(g)
    except USER_ERRORS:
        return
    syms = res.get("symmetries") or []
    if res.get("rank") is not None and res.get("dim") is not None and res.get("method") == "observability":
        assert res["rank"] + len(syms) == res["dim"]
    assert res["identifiable"] is (len(syms) == 0) or res["identifiable"] is None
    for d in syms:
        assert d.get("level") in LEVELS
        assert (d.get("generator") is None) == (d["level"] == "open")
    if plain(g["spec"]) and res.get("method") == "observability":
        for d in syms:
            if d.get("level") == "exact" and d.get("state_components") is not None:
                assert all(spy.simplify(r) == 0 for r in lifted_identities(g["spec"], d))


def vanishes(e):
    """Whether `e` simplifies to 0: a root of a positive symbol shifted by a constant becomes a new
    positive symbol, and the radicands are factored so that a square leaves its root."""
    for _ in range(3):
        if spy.simplify(e) == 0:
            return True
        for r in [w for w in e.atoms(spy.Pow) if w.exp == spy.Rational(1, 2) and len(w.base.free_symbols) == 1]:
            x = next(iter(r.base.free_symbols))
            if not (x.is_positive and r.base.is_polynomial(x) and spy.degree(r.base, x) == 1):
                continue
            a, c = spy.Poly(r.base, x).all_coeffs()
            if x in e.free_symbols:
                u = spy.Symbol(f"{x}_root", positive=True)
                e = e.xreplace({x: (u**2 - c) / a})
        e = e.replace(lambda x: x.is_Pow and x.exp == spy.Rational(1, 2),
                      lambda x: spy.sqrt(spy.factor(spy.together(x.base))))
    return False


@pytest.mark.parametrize("path", cases(RED))
def test_reduction(path):
    g = load(path)
    obj = obj_of(g["object"])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = symmetry_reduction(obj, **kw_of(g.get("args")))
    except USER_ERRORS:
        return
    gens = [{spy.Symbol(k): parse(v) for k, v in d["generator"].items()}
            for d in obj["symmetries"] if d.get("generator") and d["type"] == "general"]
    # the curved blocks are reduced after the scalings are pinned
    scal = {spy.Symbol(k): parse(v) for b in res["blocks"] if b["type"] == "scaling" and b["status"] == "reduced"
            for k, v in (b.get("pins") or {}).items()}
    gens = [{k: c.xreplace(scal) for k, c in xi.items() if k not in scal} for xi in gens]
    for b in res["blocks"]:
        assert b["status"] in ("reduced", "fixed", "invariant_only", "unresolved", "skipped")
        assert b["level"] in LEVELS
        assert (b["level"] == "open") >= bool(b.get("unproven"))
        if b["status"] != "reduced" or b["type"] == "scaling":
            continue
        own = [xi for xi in gens if {str(k) for k, c in xi.items() if c != 0} <= set(b["support"])]
        for inv in b["invariants"]:
            e = parse(inv)
            assert all(spy.cancel(sum(c * spy.diff(e, k) for k, c in xi.items())) == 0 for xi in own), inv
        # every symbol positive but the carriers that range over R
        real = {q for q, dom in (b.get("solved_domain") or {}).items() if dom == "real"}

        def own(e):
            return e.xreplace({x: spy.Symbol(str(x), real=True) if str(x) in real else
                               spy.Symbol(str(x), positive=True) for x in e.free_symbols})
        pins = {own(spy.Symbol(k)): own(parse(v)) for k, v in (b.get("pins") or {}).items()}
        for q, inv in (b.get("new_parameters") or {}).items():
            back = own(parse(inv)).xreplace(pins)
            assert vanishes(back - own(spy.Symbol(q))), f"{q} = {inv} under the pins"
    assert (res.get("checks") or {}).get("pins", True) is True

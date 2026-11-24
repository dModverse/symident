"""Sections of directions that translate linear and logarithmic coordinates, directions that move unknown
inputs, and the interval certificate of independent invariants."""

import warnings

import sympy as spy

import symident as si
from symident.reduction.blocks import independent_set


def red(obj, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return si.reduce(obj, **kw)


def mkobj(gens, coords, positive, rhs=None, unknown=None):
    dirs = [
        {"type": "general", "generator": g, "weights": None, "support": list(g), "explicit": True, "level": "exact"}
        for g in gens
    ]
    info = {"engine": "modular", "settings": {"positive": positive}, "coordinates": list(coords)}
    if rhs is not None:
        info["model_rhs"] = rhs
    if unknown is not None:
        info["unknown_inputs"] = {"names": list(unknown), "jets": {w: [w, f"{w}_d1"] for w in unknown}}
    return {
        "identifiable": False,
        "rank": len(coords) - len(gens),
        "dim": len(coords),
        "symmetries": dirs,
        "info": info,
    }


def offset_model(off="c", rev="E"):
    # a membrane-like state with an exponential rate, measured with an unknown offset
    return si.Model({"x": f"-a*exp(x/s)*(x - {rev})"}, {"y": f"x + {off}"})


def test_offset_pinned_with_invariants_and_loop():
    r = si.detect(offset_model(), reconstruct=True, cores=1, positive=["a", "s"])
    x = red(r)
    assert x.removed and not x.remaining
    b = x.to_dict()["blocks"][0]
    assert b["stage"] == "linear flow" and b["pins"] == {"c": "0"}
    assert b["injective"] and b["reach"] and not b["unproven"]
    assert {k: v for k, v in x.trafo.items() if v != k} == {"c": "0"}
    assert x.checks["pins"] and x.checks["invariants"] and x.checks["loop"]["ok"]
    # the invariants are constant along the flow
    g = r.to_dict()["symmetries"][0]["generator"]
    loc = {n: spy.Symbol(n) for n in ("E", "a", "c", "s", "x")}
    inv = {k: spy.sympify(v.replace("^", "**"), locals=loc) for k, v in b["new_parameters"].items()}
    X = {loc[k]: spy.sympify(str(v).replace("^", "**"), locals=loc) for k, v in g.items()}
    for e in inv.values():
        assert spy.simplify(sum(X[v] * spy.diff(e, v) for v in X)) == 0


def test_offset_choice_by_structure_not_name():
    # the coordinate that enters only the observable is pinned whatever the name
    for off, rev in (("c", "E"), ("Aoff", "zE"), ("zz", "a0")):
        r = si.detect(offset_model(off, rev), reconstruct=True, cores=1, positive=["a", "s"])
        b = red(r, check=False).to_dict()["blocks"][0]
        assert b["pins"] == {off: "0"}, (off, b["pins"])


def test_two_directions_one_block():
    # two directions at rates free of the moved coordinates; the pins need a minor of one sign
    obj = mkobj([{"p": "1", "q": "1"}, {"q": "u", "r": "1"}], ["p", "q", "r", "u"], ["u"])
    x = red(obj, positive=["u"])
    bs = [b for b in x.to_dict()["blocks"] if b["status"] == "reduced"]
    assert len(bs) == 1 and bs[0]["stage"] == "linear flow"
    b = bs[0]
    assert len(b["pins"]) == 2 and b["injective"] and b["reach"]
    ((q, inv),) = b["new_parameters"].items()
    p_, q_, r_, u_ = spy.symbols("p q r u")
    e = spy.sympify(inv)
    for g in ({p_: 1, q_: 1}, {q_: u_, r_: 1}):
        assert spy.simplify(sum(c * spy.diff(e, v) for v, c in g.items())) == 0
    assert x.checks["pins"] and x.checks["invariants"]


def test_positive_coordinate_in_logarithm():
    # a positive coordinate scaled at a rate free of the moved coordinates is translated in the logarithm
    obj = mkobj([{"k": "-k/s", "E": "1"}], ["k", "E", "s"], ["k", "s"])
    x = red(obj, positive=["k", "s"])
    b = x.to_dict()["blocks"][0]
    assert b["stage"] == "linear flow" and b["injective"] and b["reach"]
    assert b["pins"] in ({"E": "0"}, {"k": "1"})


def test_not_a_linear_flow():
    # a translation of positive coordinates leaves the orthant: no linear-flow section
    from symident.reduction.expr import gen_prep
    from symident.reduction.linear import linear_flow
    from symident.reduction.sign import set_positive

    old = set_positive(None)
    try:
        pr = gen_prep({"generator": {"p": "1", "q": "-1"}})
        assert linear_flow([pr]) is None
        # a component that depends on a moved coordinate
        set_positive([])
        pr = gen_prep({"generator": {"p": "q", "q": "1"}})
        assert linear_flow([pr]) is None
        # a minor that changes sign on the domain
        pr = gen_prep({"generator": {"p": "u", "q": "u"}})
        assert linear_flow([pr]) is None
    finally:
        set_positive(old)


def test_unknown_input_directions_named():
    # a direction whose components depend on states and unknown inputs has no chart in the parameters
    obj = mkobj(
        [{"k1": "k1*n*x2^n", "w1": "-(w1 + 1)*(x2^n + (w1 + 1)^n)"}],
        ["w1", "x1", "x2", "k1", "n"],
        True,
        rhs={"x1": "k1/(1 + (x2/(1 + w1))^n) - x1", "x2": "1 - x2", "w1": "w1_d1"},
        unknown=["w1"],
    )
    x = red(obj)
    b = x.to_dict()["blocks"][0]
    assert b["status"] == "unresolved" and b.get("absorbed_by_inputs")
    assert "w1 absorbs" in b["reason"] and "k1 is non-identifiable" in b["reason"]
    assert "rational" not in b["reason"]


def test_interval_minor_exact_coefficients():
    # dependent invariants with rational coefficients are not certified independent by rounding
    assert independent_set(["exp(a/3)*b", "exp(a)*b^3"], ["a", "b"], 2) == ["exp(a/3)*b"]
    assert independent_set(["exp(a/7)*b^(1/3)", "exp(3*a/7)*b"], ["a", "b"], 2) == ["exp(a/7)*b^(1/3)"]
    assert independent_set(["exp(a/7)*b^(1/3)", "exp(3*a/7)*b^2"], ["a", "b"], 2) == [
        "exp(a/7)*b^(1/3)",
        "exp(3*a/7)*b^2",
    ]
    # a function without interval version is evaluated exactly
    assert independent_set(["atan(a)*exp(b/3)", "exp(b)*atan(a)^3"], ["a", "b"], 2) == ["atan(a)*exp(b/3)"]

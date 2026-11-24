"""equilibrate with states at rest inside exponentials, logarithms and trigonometric functions, compared with
the same model whose resting state is given through the trafo; models without a resting state."""

import warnings

import pytest
import sympy as spy

import symident as si
from symident.errors import InputError

STEP = [{"var": "u", "time": "1", "value": "1", "method": "replace"}]


def det(f, g, trafo=None, events=STEP, inputs=("u",), **kw):
    kw.setdefault("cores", 2)
    model = si.Model(f, g, trafo=trafo, events=events, inputs=list(inputs))
    return si.detect(model, reconstruct=True, **kw)


def same_direction(a, b):
    """Whether two generators are proportional, with the factor a function of the coordinates."""
    if set(a) != set(b):
        return False
    k = next(iter(a))
    ra = {n: spy.sympify(f"({v})/({a[k]})".replace("^", "**")) for n, v in a.items()}
    rb = {n: spy.sympify(f"({v})/({b[k]})".replace("^", "**")) for n, v in b.items()}
    return all(spy.simplify(ra[n] - rb[n]) == 0 for n in a)


def parameter_form(d, drop=()):
    """The components of a generator off the coordinates in `drop`."""
    return {k: v for k, v in d["generator"].items() if k not in drop}


# ---- Hodgkin-Huxley neuron in current clamp: rest, then a current step

RATES = {"m": ("am*exp(V/kam)", "bm*exp(-V/kbm)"), "h": ("ah*exp(-V/kah)", "bh*exp(V/kbh)"),
         "n": ("an*exp(V/kan)", "bn*exp(-V/kbn)")}
HH = {"V": "(Iapp - gNa*m^3*h*(V - ENa) - gK*n^4*(V - EK) - gL*(V - EL))/C", "Iapp": "0",
      **{g: f"{a}*(1 - {g}) - {b}*{g}" for g, (a, b) in RATES.items()}}
HH_POS = ["C", "gNa", "gK", "gL"] + [p + g for g in "mhn" for p in ("a", "b", "ka", "kb")]
HH_EVENTS = [{"var": "Iapp", "time": "1", "value": "10", "method": "replace"},
             {"var": "Iapp", "time": "6", "value": "0", "method": "replace"}]


def hh(obs, equilibrate):
    trafo = None
    if not equilibrate:
        # the resting state given through the trafo
        trafo = {"V": "V0", "Iapp": "0"}
        for g, (a, b) in RATES.items():
            a0, b0 = a.replace("V", "V0"), b.replace("V", "V0")
            trafo[g] = f"({a0})/(({a0}) + ({b0}))"
        trafo["EL"] = (f"V0 + (gNa*({trafo['m']})^3*({trafo['h']})*(V0 - ENa) + "
                       f"gK*({trafo['n']})^4*(V0 - EK))/gL")
    return det(HH, obs, trafo=trafo, events=HH_EVENTS, inputs=["Iapp"], equilibrate=equilibrate,
               positive=HH_POS, cores=4)


def test_hodgkin_huxley_at_rest():
    eq = hh({"y": "V"}, True)
    hand = hh({"y": "V"}, False)
    assert eq.identifiable and hand.identifiable
    assert (eq.rank, eq.dim) == (hand.rank, hand.dim) == (19, 19)
    assert eq.info["rank_status"]["status"] == "proven"


def test_hodgkin_huxley_offset_at_rest():
    eq = hh({"y": "V + Vlj"}, True)
    assert (eq.rank, eq.dim) == (19, 20)
    assert eq.info["rank_status"]["status"] == "proven"
    (d,) = eq.symmetries
    assert d["level"] == "exact"
    assert set(d["support"]) == {"EK", "EL", "ENa", "Vlj", "am", "bm", "ah", "bh", "an", "bn"}
    # the direction of the offset in closed form
    shift = {"EK": "1", "EL": "1", "ENa": "1", "Vlj": "-1", "am": "-am/kam", "bm": "bm/kbm", "ah": "ah/kah",
             "bh": "-bh/kbh", "an": "-an/kan", "bn": "bn/kbn"}
    assert same_direction(d["generator"], shift)
    # the same direction with the resting state given through the trafo
    hand = hh({"y": "V + Vlj"}, False)
    assert (hand.rank, hand.dim) == (19, 20)
    # a parameter that the trafo sets is no coordinate there
    assert same_direction(parameter_form(hand.symmetries[0], ("V0",)), {k: v for k, v in shift.items() if k != "EL"})


def test_hodgkin_huxley_offset_banks_per_entry(monkeypatch):
    # each entry fitted on points that move only the leaves of the entry from the base point
    import symident.detection.forward as fw
    monkeypatch.setattr(fw, "_UNION_CAP", 0)
    eq = det(HH, {"y": "V + Vlj"}, events=HH_EVENTS, inputs=["Iapp"], equilibrate=True, positive=HH_POS,
             cores=4, control=si.reconst_control(relevance_cap_dir=1))
    (d,) = eq.symmetries
    assert d["level"] == "exact" and eq.info["rank_status"]["status"] == "proven"
    assert same_direction(d["generator"], {"EK": "1", "EL": "1", "ENa": "1", "Vlj": "-1", "am": "-am/kam",
                                           "bm": "bm/kbm", "ah": "ah/kah", "bh": "-bh/kbh", "an": "-an/kan",
                                           "bn": "bn/kbn"})


# ---- small models: a trigonometric and a logarithmic term of a state at rest

def test_trigonometric_state_at_rest():
    f = {"x": "p - k*sin(x - phi) + u", "u": "0"}
    eq = det(f, {"y": "x + c"}, equilibrate=True, positive=["k"])
    hand = det(f, {"y": "x + c"}, trafo={"x": "x0", "p": "k*sin(x0 - phi)", "u": "0"}, positive=["k"])
    assert (eq.rank, eq.dim) == (hand.rank, hand.dim) == (3, 4)
    (d,) = eq.symmetries
    assert d["level"] == "exact" and eq.info["rank_status"]["status"] == "proven"
    assert same_direction(d["generator"], parameter_form(hand.symmetries[0], ("x0",)))


def test_logarithmic_state_at_rest():
    f = {"x": "a*(1 + u) - b*x*log(x/K)", "u": "0"}
    pos = ["a", "b", "K", "s", "x"]
    eq = det(f, {"y": "s*x"}, equilibrate=True, positive=pos)
    hand = det(f, {"y": "s*x"}, trafo={"x": "x0", "a": "b*x0*log(x0/K)", "u": "0"},
               positive=["b", "K", "s", "x", "x0"])
    assert (eq.rank, eq.dim) == (hand.rank, hand.dim) == (3, 4)
    (d,) = eq.symmetries
    assert d["type"] == "scaling" and set(d["generator"]) == {"a", "K", "s"}
    assert set(hand.symmetries[0]["generator"]) == {"x0", "K", "s"}


def test_state_only_inside_exponential_at_rest():
    # a state that enters the balances only through an exponential: the resting value is drawn, the leaf solved
    f = {"A": "kA*(1 + u) - kE*exp(A)", "B": "A - kd*B", "u": "0"}
    pos = ["kA", "kE", "kd"]
    eq = det(f, {"y": "B + c"}, equilibrate=True, positive=pos)
    hand = det(f, {"y": "B + c"}, trafo={"A": "A0", "kA": "kE*exp(A0)", "B": "A0/kd", "u": "0"}, positive=pos)
    assert (eq.rank, eq.dim) == (hand.rank, hand.dim) == (3, 4)
    (d,) = eq.symmetries
    # the direction in the resting states is that of the trafo at the resting state
    g = {k: spy.sympify(v) for k, v in d["generator"].items()}
    A, B, kA, kd, kE = spy.symbols("A B kA kd kE")
    rest = {B: A / kd, kA: kE * spy.exp(A)}
    assert spy.simplify((g["kE"] / g["c"]).subs(rest) - kE * kd) == 0


def test_resting_state_not_varied_by_free_parameters():
    # a state that enters the balance also outside the exponential is solved, and no parameter outside the
    # exponent moves the resting value
    with pytest.raises(InputError, match="non-rational terms"):
        det({"x": "k - x*exp(x/k) + u", "z": "x - d*z", "u": "0"}, {"y": "z"}, equilibrate=True)


def test_state_only_inside_a_term_is_drawn_as_by_hand():
    # a state that enters the balance only through the exponential: the resting value is drawn, the leaf solved
    f = {"x": "k - exp(x/k) + u", "z": "x - d*z", "u": "0"}
    eq = det(f, {"y": "z"}, equilibrate=True)
    hand = det(f, {"y": "z"}, trafo={"x": "k*log(k)", "z": "k*log(k)/d", "u": "0"})
    assert (eq.rank, eq.dim) == (hand.rank, hand.dim) == (2, 2)
    assert eq.info["settings"]["rest_leaves"]["drawn"] == {"x": "exp(x/k)"}


# ---- models without a resting state

def test_no_resting_state():
    with pytest.raises(InputError, match="balance of x at rest"):
        det({"x": "k + u", "z": "x - d*z", "u": "0"}, {"y": "z"}, equilibrate=True)


def test_resting_state_outside_the_domain():
    # the only resting state is negative for positive parameters
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = det({"x": "k + a*x + u", "u": "0"}, {"y": "x"}, equilibrate=True)
    rest = res.info["resting"]
    if "not installed" in str(rest.get("why")):
        pytest.skip("needs msolve")
    assert any("declared domain" in str(x.message) for x in w)
    assert rest["checked"] and rest["points"] > 0 and rest["in_domain"] == 0
    ok = det({"x": "k - a*x + u", "u": "0"}, {"y": "x"}, equilibrate=True)
    assert ok.info["resting"]["in_domain"] == ok.info["resting"]["points"]

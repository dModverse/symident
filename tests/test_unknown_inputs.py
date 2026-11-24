"""Unknown inputs: jet chains whose initial values are nuisance coordinates of the kernel."""

import pytest
import sympy as spy

import symident as si

GLACIER = {"L": "-(c*T + L)/tau"}


def _same(a, b):
    return spy.cancel(spy.sympify(str(a).replace("^", "**")) - spy.sympify(str(b).replace("^", "**"))) == 0


def _change(r, par):
    """The input change of the direction that moves `par`, scaled to the component xi(par) = par."""
    (s,) = [s for s in r.symmetries if par in s.generator]
    f = spy.sympify(par) / spy.sympify(s.generator[par].replace("^", "**"))
    return {w: spy.cancel(f * spy.sympify(v.replace("^", "**"))) for w, v in s.input_change.items()}


def test_an_unknown_input_absorbs_what_the_parameters_would_explain():
    r = si.detect(si.Model(GLACIER, {"y": "L"}, unknown_inputs=["T"]), reconstruct=True)
    assert (r.rank, r.dim) == (1, 3) and r.info["coordinates"] == ["L", "c", "tau"]
    assert [s.level for s in r.symmetries] == ["exact", "exact"]
    assert r.info["rank_status"]["status"] == "proven"
    assert _same(_change(r, "c")["T"], "-T")
    assert _same(_change(r, "tau")["T"], "(L + c*T)/c")
    ui = r.info["unknown_inputs"]
    assert ui["names"] == ["T"] and ui["jets"]["T"][:2] == ["T", "T_d1"] and not ui["short"]


def test_a_time_varying_rate_as_an_unknown_input():
    # Martinelli 2022, arXiv 2211.13507, Section 6
    m = si.Model(
        {"TU": "lam - rho*TU - eta*TU*V", "TI": "eta*TU*V - delta*TI", "V": "N*delta*TI - c*V"},
        {"y1": "V", "y2": "TU + TI"},
        unknown_inputs=["eta"],
    )
    r = si.detect(m, reconstruct=True)
    # eta cancels in the derivative of y2, so it enters from order 2 on: the jet columns that can enter the
    # rows have full rank, which proves the rank modulo the input
    assert (r.rank, r.dim) == (7, 8) and r.info["rank_status"]["status"] == "proven"
    (s,) = r.symmetries
    assert s.level == "exact" and {"N", "delta"} <= set(s.support) and not {"lam", "rho", "c"} & set(s.support)


def test_a_polynomial_stand_in_is_not_an_unknown_input():
    cubic = dict(GLACIER, T="T1", T1="T2", T2="T3", T3="0")
    stand_in = si.detect(si.Model(cubic, {"y": "L"}), reconstruct=True)
    unknown = si.detect(si.Model(GLACIER, {"y": "L"}, unknown_inputs=["T"]), reconstruct=True)
    assert not any("tau" in s.support for s in stand_in.symmetries)
    assert any("tau" in s.support for s in unknown.symmetries)


def test_the_input_change_of_a_linear_input():
    r = si.detect(si.Model({"x": "-k*x + w"}, {"y": "x"}, unknown_inputs=["w"]), reconstruct=True)
    assert (r.rank, r.dim) == (1, 2) and r.symmetries[0].level == "exact"
    assert _same(_change(r, "k")["w"], "k*x")
    assert r.symmetries[0].state_components is None


@pytest.mark.parametrize(
    "rhs, change",
    [
        ("-k*x + exp(w)", "k*x*exp(-w)"),
        ("-k*x + a*w^2", "k*x/(2*a*w)"),
        ("-k*x + w*log(x)", "k*x/log(x)"),
    ],
)
def test_the_input_change_through_a_recast_or_a_log_chart(rhs, change):
    r = si.detect(si.Model({"x": rhs}, {"y": "x"}, unknown_inputs=["w"]), reconstruct=True)
    assert {s.level for s in r.symmetries} == {"exact"} and r.info["rank_status"]["status"] == "proven"
    assert _same(_change(r, "k")["w"], change)


def test_a_measured_unknown_input_leaves_the_model_identifiable():
    r = si.detect(si.Model({"x": "-k*x + w"}, {"y": "x", "yw": "w"}, unknown_inputs=["w"]))
    assert r.identifiable and (r.rank, r.dim) == (2, 2)


def test_conditions_share_an_unknown_input_or_rename_it():
    shared = si.detect(
        si.Model(
            {"x": "-k*x + s*w"},
            {"y": "x"},
            unknown_inputs=["w"],
            conditions={"rows": ["a", "b"], "cols": {"s": [1, 2]}},
        )
    )
    assert shared.identifiable
    own = si.detect(
        si.Model(
            {"x": "-k*x + s*w"},
            {"y": "x"},
            unknown_inputs=["w", "w2"],
            conditions={"rows": ["a", "b"], "cols": {"s": [1, 2], "w": ["w", "w2"]}},
        ),
        reconstruct=True,
    )
    assert (own.rank, own.dim) == (1, 2) and own.symmetries[0].level == "exact"
    ch = _change(own, "k")
    assert _same(ch["w"], "k*x") and _same(ch["w2"], "k*x/2")


def test_a_chain_below_the_lie_order_bounds_the_rank_from_below():
    odes = {"x1": "w - k1*x1", "x2": "k1*x1 - k2*x2", "x3": "k2*x2 - k3*x3"}
    r = si.detect(si.Model(odes, {"y": "x3"}, unknown_inputs=["w"]), control=si.reconst_control(input_jet_order=2))
    assert r.info["unknown_inputs"]["short"]
    assert r.info["rank_status"]["status"] == "lower bound"
    assert "jet chains" in r.info["rank_status"]["why"]
    assert any("unknown inputs" in x for x in r.info["limits"])


def test_scalings_name_the_input_weight_apart_from_the_coordinates():
    r = si.detect(si.Model(GLACIER, {"y": "L"}, unknown_inputs=["T"]), scalings_only=True)
    (s,) = r.symmetries
    assert set(s.generator) == {"c"} and _same(_change(r, "c")["T"], "-T")


def test_an_event_tells_the_rate_from_a_continuous_input():
    dose = [{"var": "x", "time": 1, "value": 1, "method": "add"}]
    r = si.detect(si.Model({"x": "-k*x + b*w"}, {"y": "x"}, unknown_inputs=["w"], events=dose), reconstruct=True)
    (s,) = r.symmetries
    assert (r.rank, r.dim) == (2, 3) and s.level == "exact" and set(s.generator) == {"b"}
    assert _same(_change(r, "b")["w"], "-w") and r.info["rank_status"]["status"] == "proven"
    idle = [{"var": "z", "time": 1, "value": 1, "method": "add"}]
    r = si.detect(si.Model({"x": "-k*x + b*w", "z": "0"}, {"y": "x"}, unknown_inputs=["w"], events=idle))
    assert r.rank == 1


def test_an_unknown_input_is_zero_at_rest():
    r = si.detect(
        si.Model({"x": "p - k*x + b*w"}, {"y": "x"}, unknown_inputs=["w"]), equilibrate=True, reconstruct=True
    )
    assert set(r.info["coordinates"]) == {"b", "k", "p"} and r.rank == 1
    (s,) = [s for s in r.symmetries if "b" in s.generator]
    assert s.level == "exact" and _same(_change(r, "b")["w"], "-w")
    r = si.detect(si.Model({"x": "p - k*x + b*u"}, {"y": "x"}, known_inputs=["u"]), equilibrate=True)
    assert r.identifiable and set(r.info["coordinates"]) == {"b", "k", "p"}


@pytest.mark.parametrize(
    "kw, what",
    [
        (dict(events=[{"var": "w", "time": 1, "value": 1, "method": "add"}]), "event changes the unknown input"),
        (dict(trafo={"w": "w0"}), "trafo"),
    ],
)
def test_settings_without_unknown_inputs_are_refused(kw, what):
    with pytest.raises(ValueError, match=what):
        si.detect(si.Model({"x": "-k*x + w"}, {"y": "x"}, unknown_inputs=["w"], **kw))


def test_unknown_inputs_are_checked_against_the_model():
    with pytest.raises(ValueError, match="symbolic"):
        si.detect(si.Model({"x": "-k*x + w"}, {"y": "x"}, unknown_inputs=["w"]), engine="symbolic")
    with pytest.raises(ValueError, match="enters no"):
        si.detect(si.Model({"x": "-k*x"}, {"y": "x"}, unknown_inputs=["w"]))
    with pytest.raises(ValueError, match="has an equation"):
        si.detect(si.Model({"x": "-k*x + w", "w": "-w"}, {"y": "x"}, unknown_inputs=["w"]))
    with pytest.raises(ValueError, match="time derivative"):
        si.detect(si.Model({"x": "-k*x + w + w_d1"}, {"y": "x"}, unknown_inputs=["w"]))


@pytest.mark.parametrize("gauge", [None, ["tau"]])
def test_a_gauge_keeps_the_unknown_inputs(gauge):
    m = si.Model(GLACIER, {"y": "L"}, unknown_inputs=["T"])
    free = si.detect(m, reconstruct=True)
    r = si.detect(m, reconstruct=True, gauge=gauge)
    assert (r.rank, r.dim, r.info["coordinates"]) == (free.rank, free.dim, free.info["coordinates"])
    assert "T" not in r.info["coordinates"] and r.info["unknown_inputs"]["names"] == ["T"]


def test_full_rank_with_unknown_inputs_is_verified():
    # the jet columns have full column rank: no point can raise the rank modulo them
    r = si.detect(si.Model({"x": "-k*x + w"}, {"y": "x", "yw": "w"}, unknown_inputs=["w"]))
    assert r.identifiable and r.info["projection"]["status"] == "proven"
    assert r.info["rank_status"]["status"] == "proven"
    # inputs that enter only together leave the jet columns short of full rank
    r = si.detect(si.Model({"x": "-k*x + w1 + w2"}, {"y": "x", "yw": "w1 + w2"}, unknown_inputs=["w1", "w2"]))
    st = r.info["rank_status"]
    assert r.identifiable and r.info["projection"]["status"] == "verified"
    assert st["status"] == "verified" and 0 < st["probability"] < 1e-6


def test_an_unknown_input_under_a_log_chart_is_no_coordinate():
    r = si.detect(si.Model({"x": "k0 + k/(1 + (x/(1 + w))^n) - x"}, {"y": "x"}, unknown_inputs=["w"]), reconstruct=True)
    assert "w" not in r.info["coordinates"] and r.info["rank_status"]["status"] == "proven"
    assert {s.level for s in r.symmetries} == {"exact"}
    # the lift in the log coordinate of the input has no closed form in the input: no input change of zero
    for s in r.symmetries:
        assert s.input_change is None or any(str(v) != "0" for v in s.input_change.values())
        assert not set(s.state_components or {}) - {"x"}


# a flux w between two states: one direction moves N, d and the initial values, and w absorbs it
FLUX = ({"a": "-w*a", "b": "w*a - d*b"}, {"y1": "a + b", "y2": "N*b"})


def test_a_pointwise_lift_reduces_a_direction_the_input_absorbs():
    r = si.detect(si.Model(*FLUX, unknown_inputs=["w"]), reconstruct=True)
    (s,) = r.symmetries
    assert s.level == "exact" and {"N", "d"} <= set(s.generator)
    red = si.reduce(r)
    assert red.removed and not red.remaining and red.checks["loop"]["ok"]
    (b,) = red.to_dict()["blocks"]
    assert b["inputs"]["proof"] == "lift" and b["inputs"]["absorb"] == ["w"]
    assert {k for k, v in red.trafo.items() if v != k} <= {"N", "d"}
    assert "Unknown inputs:" in repr(red)
    # the lift satisfies D(xi_x) = V(f) and V(h) = 0 as identities
    (lf,) = b["inputs"]["lift"].values()
    a, bb, d, N, w = spy.symbols("a b d N w")
    loc = {"a": a, "b": bb, "d": d, "N": N, "w": w}
    xi = {
        k: spy.sympify(v.replace("^", "**"), locals=loc)
        for part in ("params", "states", "inputs")
        for k, v in lf[part].items()
    }
    V = {spy.Symbol(k): v for k, v in xi.items()}
    f = {a: -w * a, bb: w * a - d * bb}

    def act(e):
        return sum(spy.diff(e, k) * v for k, v in V.items())

    for x, fx in f.items():
        Dxi = sum(spy.diff(V.get(x, 0), y) * f[y] for y in f)
        assert spy.cancel(Dxi - act(fx)) == 0
    assert spy.cancel(act(a + bb)) == 0 and spy.cancel(act(N * bb)) == 0


TWO_COMPARTMENTS = {"x1": "-(k12 + k1e)*x1 + k21*x2 + b*u", "x2": "k12*x1 - k21*x2"}


def test_the_inversion_of_the_input_proves_every_section():
    r = si.detect(si.Model(TWO_COMPARTMENTS, {"y1": "x1"}, unknown_inputs=["u"]), reconstruct=True)
    red = si.reduce(r)
    assert len(red.removed) == 5 and not red.remaining
    for b in red.to_dict()["blocks"]:
        assert b["inputs"]["proof"] == "inversion" and b["inputs"]["determinant"] == "b"
        assert "u positive" in b["inputs"]["condition"]


def test_a_direction_without_a_proof_that_the_input_absorbs_it_stays():
    # a second, separate observable leaves no inversion; the rates of the hidden compartment need a lift that
    # depends on the past of the trajectory
    odes = dict(TWO_COMPARTMENTS, z="-m*z")
    r = si.detect(si.Model(odes, {"y1": "x1", "y2": "z"}, unknown_inputs=["u"]), reconstruct=True)
    red = si.reduce(r)
    gens = dict(zip([f"X{c}" for c in "₁₂₃₄₅"], [s.generator for s in r.symmetries]))
    left = {k for b in red.to_dict()["blocks"] if b["status"] == "unresolved" for k in gens[b["labels"][0]]}
    assert "k21" in left and "x2" in left
    assert all("pointwise lift" in b["reason"] for b in red.to_dict()["blocks"] if b["status"] == "unresolved")
    done = {k for b in red.to_dict()["blocks"] if b["status"] == "reduced" for k in gens[b["labels"][0]]}
    assert {"b", "k1e"} <= done


def test_the_count_takes_the_jets_of_an_unknown_input_as_unknowns():
    # x -> -x with w -> -w keeps the output: an exact map of the states and the input
    r = si.detect(si.Model({"x": "-k*x + w"}, {"y": "x^2"}, unknown_inputs=["w"]), reconstruct=True)
    out = r.discrete()
    info = out["info"]
    assert info["count"] == 2 and info["certificate"]["complete"] == "verified" and 0 < info["count_bound"] < 1e-6
    (s,) = out["symmetries"]
    assert s["map"] == {"x": "-x"} and s["level"] == "exact" and s["state_maps"] == [{"x": "-x", "w": "-w"}]


def test_the_count_leaves_out_the_jets_the_output_does_not_read():
    # the flux cancels in y1: the jets above the order the outputs read would leave a free direction
    r = si.detect(si.Model(*FLUX, unknown_inputs=["w"]), reconstruct=True)
    info = r.discrete()["info"]
    assert info["count"] == 1 and info["certificate"]["complete"] == "verified"
    assert info["reduced"]["coordinates"] and "w" not in info["reduced"]["coordinates"]


def test_a_direction_the_inputs_absorb_in_several_conditions_keeps_without_a_proof():
    m = si.Model(
        {"x": "-k*x + s*w"},
        {"y": "x"},
        unknown_inputs=["w", "w2"],
        conditions={"rows": ["a", "b"], "cols": {"s": [1, 2], "w": ["w", "w2"]}},
    )
    red = si.reduce(si.detect(m, reconstruct=True))
    (b,) = red.to_dict()["blocks"]
    assert red.remaining and b["status"] == "unresolved" and "one condition" in b["reason"]

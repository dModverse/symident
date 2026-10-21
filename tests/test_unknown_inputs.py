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
    m = si.Model({"TU": "lam - rho*TU - eta*TU*V", "TI": "eta*TU*V - delta*TI", "V": "N*delta*TI - c*V"},
                 {"y1": "V", "y2": "TU + TI"}, unknown_inputs=["eta"])
    r = si.detect(m, reconstruct=True)
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


@pytest.mark.parametrize("rhs, change", [
    ("-k*x + exp(w)", "k*x*exp(-w)"),
    ("-k*x + a*w^2", "k*x/(2*a*w)"),
    ("-k*x + w*log(x)", "k*x/log(x)"),
])
def test_the_input_change_through_a_recast_or_a_log_chart(rhs, change):
    r = si.detect(si.Model({"x": rhs}, {"y": "x"}, unknown_inputs=["w"]), reconstruct=True)
    assert {s.level for s in r.symmetries} == {"exact"} and r.info["rank_status"]["status"] == "proven"
    assert _same(_change(r, "k")["w"], change)


def test_a_measured_unknown_input_leaves_the_model_identifiable():
    r = si.detect(si.Model({"x": "-k*x + w"}, {"y": "x", "yw": "w"}, unknown_inputs=["w"]))
    assert r.identifiable and (r.rank, r.dim) == (2, 2)


def test_conditions_share_an_unknown_input_or_rename_it():
    shared = si.detect(si.Model({"x": "-k*x + s*w"}, {"y": "x"}, unknown_inputs=["w"],
                                conditions={"rows": ["a", "b"], "cols": {"s": [1, 2]}}))
    assert shared.identifiable
    own = si.detect(si.Model({"x": "-k*x + s*w"}, {"y": "x"}, unknown_inputs=["w", "w2"],
                             conditions={"rows": ["a", "b"], "cols": {"s": [1, 2], "w": ["w", "w2"]}}),
                    reconstruct=True)
    assert (own.rank, own.dim) == (1, 2) and own.symmetries[0].level == "exact"
    ch = _change(own, "k")
    assert _same(ch["w"], "k*x") and _same(ch["w2"], "k*x/2")


def test_a_chain_below_the_lie_order_bounds_the_rank_from_below():
    odes = {"x1": "w - k1*x1", "x2": "k1*x1 - k2*x2", "x3": "k2*x2 - k3*x3"}
    r = si.detect(si.Model(odes, {"y": "x3"}, unknown_inputs=["w"]),
                  control=si.reconst_control(input_jet_order=2))
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
    r = si.detect(si.Model({"x": "p - k*x + b*w"}, {"y": "x"}, unknown_inputs=["w"]), equilibrate=True,
                  reconstruct=True)
    assert set(r.info["coordinates"]) == {"b", "k", "p"} and r.rank == 1
    (s,) = [s for s in r.symmetries if "b" in s.generator]
    assert s.level == "exact" and _same(_change(r, "b")["w"], "-w")
    r = si.detect(si.Model({"x": "p - k*x + b*u"}, {"y": "x"}, known_inputs=["u"]), equilibrate=True)
    assert r.identifiable and set(r.info["coordinates"]) == {"b", "k", "p"}


@pytest.mark.parametrize("kw, what", [
    (dict(events=[{"var": "w", "time": 1, "value": 1, "method": "add"}]), "event changes the unknown input"),
    (dict(trafo={"w": "w0"}), "trafo"),
])
def test_settings_without_unknown_inputs_are_refused(kw, what):
    with pytest.raises(ValueError, match=what):
        si.detect(si.Model({"x": "-k*x + w"}, {"y": "x"}, unknown_inputs=["w"], **kw))
    with pytest.raises(ValueError, match="switches"):
        si.detect(si.Model({"x": "-k*x + w + ifelse(x > c, a, 0)"}, {"y": "x"}, unknown_inputs=["w"]))


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

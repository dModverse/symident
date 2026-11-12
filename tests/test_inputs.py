"""A malformed model or argument raises an InputError that names it, before any analysis."""

import pytest

import symident as si
from symident.errors import InputError

F, G = {"x": "-k*x"}, {"y": "s*x"}
EV = {"var": "x", "time": 1, "value": "1", "method": "add"}
RX = {"species": ["A", "B"], "rates": ["k1*A", "k2*B"]}


def _model(odes=F, obs=G, **kw):
    return si.detect(si.Model(odes, obs, **kw))


CASES = {
    "stoichiometry rows for the species": lambda: si.Model(
        reactions=dict(RX, stoichiometry=[[-1, 1], [1, -1], [0, 0]]), observables=G),
    "stoichiometry transposed": lambda: si.Model(reactions=dict(RX, stoichiometry=[[-1], [1]]), observables=G),
    "rates short": lambda: si.Model(reactions={"species": ["A", "B"], "stoichiometry": [[-1, 1]], "rates": []},
                                    observables=G),
    "empty right-hand side": lambda: _model({"x": ""}),
    "right-hand side None": lambda: _model({"x": None}),
    "empty observable": lambda: _model(obs={"y": " "}),
    "no states": lambda: _model({}),
    "division by zero": lambda: _model({"x": "k/0"}),
    "syntax error": lambda: _model({"x": "-k*x +"}),
    "unbalanced parenthesis": lambda: _model({"x": "-k*(x"}),
    "keyword as a state": lambda: _model({"lambda": "-k*lambda"}, {"y": "s*lambda"}),
    "parameter named like a state": lambda: _model(parameters=["x"]),
    "event without time": lambda: _model(events=[{"var": "x", "value": "1", "method": "add"}]),
    "event without var": lambda: _model(events=[{"time": 1, "value": "1", "method": "add"}]),
    "event method": lambda: _model(events=[dict(EV, method="set")]),
    "event condition out of range": lambda: _model(events=[dict(EV, cond=[5])]),
    "event variable": lambda: _model(events=[dict(EV, var="q")]),
    "condition column length": lambda: _model(conditions={"rows": ["a", "b"], "cols": {"k": [1]}}),
    "total not conserved": lambda: _model({"A": "-k*A", "B": "k*A"}, {"y": "B"}, totals={"T": "A - B"}),
    "model None": lambda: si.detect(None),
    "model dict": lambda: si.detect({"x": "-k*x"}),
    "engine": lambda: si.detect(si.Model(F, G), engine="foo"),
    "cores zero": lambda: si.detect(si.Model(F, G), cores=0),
    "cores negative": lambda: si.detect(si.Model(F, G), cores=-3),
    "trace": lambda: si.detect(si.Model(F, G), trace="foo"),
    "gauge string": lambda: si.detect(si.Model(F, G), gauge="k"),
    "control setting": lambda: si.detect(si.Model(F, G), control={"nope": 1}),
    "reduce of None": lambda: si.reduce(None),
    "reduce degree": lambda: si.reduce(si.detect(si.Model(F, G)), d_poly=-1),
}


@pytest.mark.parametrize("case", CASES)
def test_a_wrong_form_is_an_input_error(case):
    with pytest.raises(InputError):
        CASES[case]()


def test_the_error_names_the_argument():
    with pytest.raises(InputError, match="stoichiometry"):
        CASES["stoichiometry transposed"]()
    with pytest.raises(InputError, match="x is an empty"):
        CASES["empty right-hand side"]()
    with pytest.raises(InputError, match="event 1 has neither `time` nor `root`"):
        CASES["event without time"]()
    with pytest.raises(InputError, match="total T"):
        CASES["total not conserved"]()


def test_an_input_error_is_a_value_error():
    assert issubclass(InputError, ValueError)


def test_a_reaction_network_with_one_row_per_reaction_builds():
    m = si.Model(reactions=dict(RX, stoichiometry=[[-1, 1], [1, -1]]), observables={"y": "s*B"})
    assert sorted(m.odes) == ["A", "B"]


def test_a_product_base_of_a_symbolic_power_is_split():
    plain = si.detect(si.Model({"x": "k1*x^n/(K^n + x^n) - k2*x"}, G), reconstruct=True)
    scaled = si.detect(si.Model({"x": "k1*x^n/((2*K)^n + x^n) - k2*x"}, G), reconstruct=True)
    assert [s.level for s in scaled.symmetries] == [s.level for s in plain.symmetries]


def test_an_ignored_trafo_entry_is_a_limit():
    m = si.Model({"x": "k1 - k2*x"}, G, trafo={"x": "x0"})
    with pytest.warns(UserWarning, match="are ignored"):
        r = si.detect(m, equilibrate=True)
    assert any("are ignored" in w for w in r.info["limits"])


def test_a_total_that_is_not_conserved_raises_without_the_debug_switch():
    with si.switches(DEBUG=False), pytest.raises(InputError, match="not conserved"):
        _model({"A": "-k*A", "B": "k*A"}, {"y": "B"}, totals={"T": "A - B"})

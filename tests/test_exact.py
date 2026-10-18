"""Exact proofs: the identities of a lifted field cancel over Q, one equation at a time."""

import symident as si
from symident.gfp import PRIMES
from symident.sym.exact import (
    certify,
    event_identities,
    initial_identities,
    output_identities,
    regime_identities,
)

REGIME = {"f": ["x = -k*x + b", "z = c*x - d*z"], "g": ["y = s*z"]}
# a scaling of the regime, lifted onto the states
SCALE = {"x": "x", "z": "z", "b": "b", "c": "0", "s": "-s"}


def test_a_symmetry_and_its_lift_pass():
    assert certify(regime_identities(REGIME, SCALE))["ok"] is True
    assert certify(initial_identities(["x = x0", "z = 0"], dict(SCALE, x0="x0")))["ok"] is True
    ev = [{"var": "x", "method": "add", "value": "u"}]
    assert certify(event_identities(ev, dict(SCALE, u="u"), dict(SCALE, u="u"), ["x", "z"]))["ok"] is True


def test_a_wrong_field_fails_with_the_identity_named():
    wrong = dict(SCALE, s="s")
    out = certify(regime_identities(REGIME, wrong))
    assert out["ok"] is False and "Y(" in out["why"]
    ev = [{"var": "x", "method": "add", "value": "u"}]
    assert certify(event_identities(ev, SCALE, SCALE, ["x", "z"]))["ok"] is False


def test_closed_forms_reach_the_exact_level():
    # the general direction needs a different lift in each condition
    m = si.Model({"x1": "kb + kin*u - (k1 + kl)*x1", "x2": "k1*x1 - k2*x2"}, {"y": "s*x2"},
                 trafo={"x1": "D", "x2": "0"}, conditions={"rows": ["a", "b"], "cols": {"u": [1, 2]}})
    r = si.detect(m, reconstruct=True)
    assert [s.level for s in r.symmetries] == ["exact", "exact"]
    assert r.info["rank_proof"] == "proven symmetries"


def test_a_multiple_of_the_closed_form_is_lifted():
    # the closed form is normalized on one coordinate; only a multiple of it is a field
    r = si.detect(si.Model({"x1": "-x2", "x2": "x1"}, {"y": "x1^2 + x2^2"}), reconstruct=True)
    (s,) = r.symmetries
    assert s.level == "exact" and set(s.state_components) == {"x1", "x2"}
    assert r.info["rank_proof"] == "proven symmetries" and r.info["proof"] == []


def test_a_start_at_rest_proves_directions_through_the_constant_output():
    r = si.detect(si.Model({"x": "k1 - k2*x"}, {"y": "x"}, trafo={"x": "k1/k2"}), reconstruct=True)
    assert [s.level for s in r.symmetries] == ["exact"]
    assert r.info["rank_proof"] == "proven symmetries"


def test_a_conserved_output_proves_the_rates_it_hides():
    r = si.detect(si.Model({"A": "-k1*A + k2*B", "B": "k1*A - k2*B"}, {"y": "A + B"}), reconstruct=True)
    assert {s.level for s in r.symmetries} == {"exact"} and len(r.symmetries) == 3


def test_output_identities_refute_a_wrong_direction():
    rg = {"f": ["x = k1 - k2*x"], "g": ["y = x"]}
    assert certify(output_identities(rg, ["x = k1/k2"], {"k1": "k1", "k2": "k2"}))["ok"] is True
    assert certify(output_identities(rg, ["x = k1/k2"], {"k1": "k1"}))["ok"] is False
    # away from rest the output is not a polynomial in t
    assert output_identities(rg, ["x = 0"], {"k1": "k1"}) is None


def test_directions_of_a_resting_start_are_exact_at_a_rational_rest():
    # a direction of the resting start only is decided exactly by the resting output
    m = si.Model({"x1": "kb + kin*u - (k1 + kl)*x1", "x2": "k1*x1 - k2*x2"}, {"y": "s*x2"},
                 conditions={"rows": ["a", "b"], "cols": {"u": [1, 2]}})
    r = si.detect(m, reconstruct=True, equilibrate=True)
    assert {s.level for s in r.symmetries} == {"exact"} and r.info["proof"] == []
    r = si.detect(si.Model({"x": "b - a*x"}, {"y": "s*x"}), reconstruct=True, equilibrate=True)
    assert {s.level for s in r.symmetries} == {"exact"}


def test_pi_in_a_resting_model_is_proven():
    r = si.detect(si.Model({"x": "pi*k - d*x"}, {"y": "s*x"}), reconstruct=True, equilibrate=True)
    assert {s.level for s in r.symmetries} == {"exact"} and r.info["proof"] == []


def test_a_held_moiety_pivot_rests_at_its_parameter():
    # the resting output and the resting subsystem take the pivot as a parameter
    pair = {"species": ["A", "B"], "smatrix": [[-1, 1], [1, -1]], "rates": ["k1*A", "k2*B"]}
    r = si.detect(si.Model(reactions=pair, observables={"Aobs": "alpha*A"}), reconstruct=True, equilibrate=True)
    assert {s.level for s in r.symmetries} == {"exact"} and r.info["rank_proof"] == "proven symmetries"
    net = {"species": ["A", "B", "C"], "smatrix": [[-1, 1, 0], [1, -1, 0], [0, 0, -1]],
           "rates": ["k1*A", "k2*B", "kc*C"]}
    m = si.Model(reactions=net, observables={"Aobs": "alpha*A", "Cobs": "s*C"},
                 events=[{"var": "C", "time": 5, "value": "1", "method": "add"}])
    r = si.detect(m, reconstruct=True, equilibrate=True)
    assert {s.level for s in r.symmetries} == {"exact"} and r.info["proof"] == []


def test_parameters_of_unobserved_or_vanishing_states_are_exact():
    # a parameter of a state that no observable of its condition sees
    m = si.Model({"A": "-k*A", "B": "-q*B"}, [{"yA": "A"}, {"yB": "B"}],
                 conditions={"rows": ["elisa", "wb"], "cols": {"q": ["q1", "q2"]}})
    assert [s.level for s in si.detect(m, reconstruct=True).symmetries] == ["exact"]
    # parameters that act only through a state that starts and stays at zero
    m = si.Model({"x1": "-k1*x1", "x2": "k1*x1 - k2*x2 + a*x3", "x3": "-k3*x3"}, {"y": "x2"},
                 trafo={"x1": "1", "x2": "0", "x3": "0"})
    r = si.detect(m, reconstruct=True)
    assert {s.level for s in r.symmetries} == {"exact"} and r.info["rank_proof"] == "proven symmetries"


def test_a_start_event_is_part_of_the_initial_value():
    # a start event is composed into the initial value the lift has to match
    m = si.Model({"A": "-k*A"}, {"y": "s*A"}, conditions={"rows": ["c1"], "cols": {"d0": [1]}},
                 events=[{"var": "A", "time": 0, "value": "d0", "method": "add"}])
    (s,) = si.detect(m, reconstruct=True).symmetries
    assert s.level == "exact" and s.state_components == {"A": "A"}


def test_symbolic_exponents_are_proven_in_recast_coordinates():
    # powers with a symbolic exponent enter as their own states; the lift follows by the chain rule
    r = si.detect(si.Model({"p": "kpr/(1 + kinh*p^nhill) - dp*p"}, {"y": "s*p"}), reconstruct=True)
    assert [s.level for s in r.symmetries] == ["exact"]
    m = si.Model({"FB": "k_pr_FB - d_FB*FB", "x": "k_pr_x*K^n/(K^n + FB^n) - d_x*x"}, {"xobs": "scale*x"})
    r = si.detect(m, reconstruct=True, equilibrate=True)
    assert {s.level for s in r.symmetries} == {"exact"} and r.info["proof"] == []


def test_closed_forms_of_the_symbolic_engine_are_proven():
    r = si.detect(si.Model({"m": "ktx - dm*m", "p": "ktl*m - dp*p"}, {"y": "p"}), engine="symbolic")
    assert [s.level for s in r.symmetries] == ["exact"] and r.info["lie_certified"]
    assert r.info["rank_status"] == {"status": "proven", "why": "Lie order proven sufficient, exact kernel",
                                     "probability": None}
    # exponentials of single coordinates and a logarithm of a number enter as symbols
    m = si.Model({"x": "-exp(lk)*x + exp(lb)"}, {"y": "log10(s*x)"}, trafo={"x": "10^lx", "s": "10^ls"})
    r = si.detect(m, engine="symbolic")
    assert r.rank == 3 and [s.level for s in r.symmetries] == ["exact"] and r.info["rank_proven"] == 1
    # a state at rest from its initial value on is replaced by that value
    r = si.detect(si.Model({"x": "-k*(x - 1)"}, {"y": "s*x"}, trafo={"x": "1"}), engine="symbolic")
    assert r.info["rank_proof"] == "Lie order proven sufficient, exact kernel"


def test_without_a_certified_lie_order_the_symbolic_engine_proves_by_lifts():
    # a start on an invariant line hides a direction from the rows at the initial values
    m = si.Model({"x1": "-k*x1 + a*(x2 - x1)", "x2": "-k*x2 + a*(x1 - x2)"}, {"y": "s*x1/q"},
                 trafo={"x1": "1", "x2": "1"})
    r = si.detect(m, engine="symbolic")
    assert {tuple(s.support): s.level for s in r.symmetries} == {("a",): "checked", ("q", "s"): "exact"}
    assert not r.info["lie_certified"]
    st = r.info["rank_status"]
    assert st["status"] == "lower bound" and st["why"].endswith("stay below the generic rank")


def test_a_direction_of_the_reachable_set_says_why_it_stays_below_exact():
    m = si.Model({"A": "-k1*A + k2*B", "B": "k1*A - k2*B"}, {"Aobs": "alpha*A"}, trafo={"k1": "k2*B/A"},
                 events=[{"var": "A", "time": 0, "value": "1", "method": "add"}])
    r = si.detect(m, reconstruct=True)
    (p,) = r.info["proof"]
    assert p["reason"].startswith("lift of the closed form: [Y, F] on")
    assert "output is not a polynomial in t" in p["reason"]


def test_rest_parametrization_solves_states_or_linear_rates():
    import sympy as spy

    from symident.sym.exact import rest_parametrization
    x, y, a, b, c = spy.symbols("x y a b c")
    par = rest_parametrization([b - a*x, a*x - c*y], [x, y], [a, b, c])
    assert par["params"] == [] and spy.simplify(par["sub"][y] - b/c) == 0
    # nonlinear in a state: the linear one is eliminated, a rate solves the rest
    eqs = [b - a*x**2, a*x - c*y]
    par = rest_parametrization(eqs, [x, y], [a, b, c])
    assert len(par["params"]) == 1 and set(par["sub"]) == {y} | set(par["params"])
    assert all(spy.cancel(e.xreplace(par["sub"])) == 0 for e in eqs)


def test_kernel_check_bound_from_the_degrees():
    from symident.detection.bound import identity_degree, kernel_bound
    f = {"x1": "kb - (k1 + kl)*x1", "x2": "k1*x1 - k2*x2"}
    # polynomial rates: each Lie derivative raises the degree by one
    assert identity_degree(f, {"y": "s*x2"}, {"x1": "D", "x2": "0"}, 3, {"k1": "k1", "s": "-s"}) == 6
    assert identity_degree({"x": "-k*exp(x)"}, {"y": "x"}, {"x": "x0"}, 3, {"k": "k"}) is None
    b = kernel_bound(f, {"y": "s*x2"}, {"x1": "D", "x2": "0"}, 3, [{"k1": "k1"}] * 2, 10 ** 9 + 7, 2)
    assert b["degree"] == 6 and b["probability"] == 2 * (6 / (10 ** 9 + 7)) ** 2


def test_directions_below_exact_get_the_bound():
    m = si.Model({"x1": "kb + kin*u - (k1 + kl)*x1", "x2": "k1*x1 - k2*x2"}, {"y": "s*x2"},
                 trafo={"x1": "D", "x2": "0"}, conditions={"rows": ["a", "b"], "cols": {"u": [1, 2]}})
    with si.switches(NOFIELDS=True):
        r = si.detect(m, reconstruct=True)
    assert {s.level for s in r.symmetries} == {"verified"}
    assert {p["reason"] for p in r.info["proof"]} == {"switched off (NOFIELDS)"}
    assert 0 < r.info["bound"]["probability"] < 1e-12 and r.info["bound"]["points"] == 2


def test_the_regime_detections_of_the_combined_lift_keep_to_their_budget():
    m = si.Model({"A": "-(ka + kb)*A", "B1": "ka*A - k1*B1", "B2": "kb*A - k2*B2", "C": "k3*B1 + k4*B2 - kc*C"},
                 {"y": "C"}, trafo={"A": "A0", "B1": "0", "B2": "0", "C": "0"}, fixed=["A0"])
    r = si.detect(m, reconstruct=True, control=si.reconst_control(exact_work_cap=1))
    assert any(lim.startswith("exact proof: work") for lim in r.info["limits"])
    assert all("above exact_work_cap=1" in p["reason"] for p in r.info["proof"])


def test_unproven_directions_are_listed_and_printed_with_their_level():
    m = si.Model({"A": "-k1*A + k2*B", "B": "k1*A - k2*B"}, {"Aobs": "alpha*A"}, trafo={"k1": "k2*B/A"},
                 events=[{"var": "A", "time": 0, "value": "1", "method": "add"}])
    r = si.detect(m, reconstruct=True)
    assert [s.level for s in r.unproven()] == ["verified"] and r.unproven()[0].explicit
    text = repr(r)
    assert "X₁  (verified)" in text and "not proven" in text and "probability at most" in text
    exact = si.detect(si.Model({"x": "k1 - k2*x"}, {"y": "s*x"}), reconstruct=True)
    assert exact.unproven() == [] and "exact = proven over Q." in repr(exact)


def test_the_bound_covers_events_after_the_start():
    ev = [{"var": "A", "time": 0, "value": "d", "method": "add"},
          {"var": "A", "time": 5, "value": "d", "method": "add"}]
    m = si.Model({"A": "-k*A"}, {"y": "s*A"}, trafo={"A": "0"}, events=ev)
    with si.switches(NOFIELDS=True):
        r = si.detect(m, reconstruct=True)
    assert [s.level for s in r.unproven()] == ["verified"]
    assert 0 < r.info["bound"]["probability"] < 1e-10
    # an event time that depends on the parameters enters as a rational function of them
    m = si.Model({"A": "-k*A"}, {"y": "s*A"}, trafo={"A": "0"},
                 events=[dict(ev[0]), dict(ev[1], time="tau")])
    with si.switches(NOFIELDS=True):
        r = si.detect(m, reconstruct=True)
    assert r.unproven() and 0 < r.info["bound"]["probability"] < 1e-10


def test_the_bound_holds_in_the_recast_coordinates_of_a_symbolic_exponent():
    m = si.Model({"x": "v*u^n/(K^n + u^n) - d*x", "u": "-e*u"}, {"y": "x"})
    r = si.detect(m, reconstruct=True)
    assert r.unproven() and 0 < r.info["bound"]["probability"] < 1e-10


def test_the_rank_status_says_how_far_the_number_of_directions_holds():
    # every direction closed at a Lie order proven sufficient: the rank rests on the verified closed forms
    f = {"A": "k1 - d1*A + kx*X", "B": "k2*A - d2*B", "X": "kp*plasmid - dx*X"}
    rest = [{"X": f"kp*{u}/dx", "A": f"(k1 + kx*kp*{u}/dx)/d1", "B": f"k2*(k1 + kx*kp*{u}/dx)/(d1*d2)",
             "plasmid": str(u)} for u in (0, 1)]
    m = si.Model(f, {"yA": "s*(A + X)", "yB": "B"}, trafo=rest,
                 conditions={"rows": ["ctrl", "oe"], "cols": {"dose": [1, 1]}},
                 events=[{"var": "A", "time": 0, "value": "dose", "method": "add"}])
    r = si.detect(m, reconstruct=True)
    st = r.info["rank_status"]
    assert st["status"] == "verified" and 0 < st["probability"] < 1e-10
    assert "The rank is verified, not proven" in repr(r)
    r = si.detect(si.Model({"x1": "-k*x1 + a*(x2 - x1)", "x2": "-k*x2 + a*(x1 - x2)"}, {"y": "x1"},
                           trafo={"x1": "1", "x2": "1"}), engine="symbolic")
    assert r.info["rank_status"]["status"] == "lower bound" and "lower bound" in repr(r)
    exact = si.detect(si.Model({"x": "k1 - k2*x"}, {"y": "s*x"}), reconstruct=True)
    assert exact.info["rank_status"]["status"] == "proven"


def _late_switch(x0=None):
    tr = {"w": "0", "v": "0", "u": "0"}
    if x0 is not None:
        tr["x"] = x0
    return si.Model({"x": "u*w - k*x", "w": "c*(1 - u)*v", "v": "1 - u", "u": "0"}, {"y": "x"}, trafo=tr,
                    events=[{"var": "u", "time": 0, "value": "0", "method": "replace"},
                            {"var": "u", "time": 1, "value": "1", "method": "replace"}])


CAP1 = si.reconst_control(gap_order_cap=1)


def test_a_gap_order_is_raised_to_its_rank_bound():
    for x0 in (None, "0"):
        r = si.detect(_late_switch(x0), reconstruct=True)
        assert r.identifiable and r.info["gap_certified"] is True and r.info["gap_order_used"] >= 2
        assert r.info["limits"] == []
    ev = [{"var": "u", "time": t, "value": v, "method": "replace"} for t, v in ((0, "0"), (1, "1"), (2, "0"))]
    m = si.Model({"x": "u*w - k*x", "w": "c*(1 - u)*v", "v": "1 - u", "u": "0"}, {"y": "x"},
                 trafo={"w": "0", "v": "0", "u": "0"}, events=ev)
    assert si.detect(m, reconstruct=True).identifiable


def test_a_gap_order_below_its_bound_leaves_the_rank_a_lower_bound():
    r = si.detect(_late_switch(), reconstruct=True, verify=False, control=CAP1)
    assert r.info["gap_certified"] is False
    assert r.info["gap_certificate"] == "the rank did not grow from gap order 0 to 1 (the rank stays below " \
                                        "its bound 3 up to the cap)"
    assert r.info["rank_status"] == {"status": "lower bound", "probability": None,
                                     "why": "the gap order is not proven sufficient"}
    assert any(lim.startswith("gap order: ") for lim in r.info["limits"])
    assert "the gap order is not proven sufficient" in " ".join(repr(r).split())
    r = si.detect(_late_switch(), reconstruct=True, verify=False, control=si.reconst_control(gap_order_cap=0))
    assert r.info["gap_certified"] is False and "gap_order_cap=0" in r.info["gap_certificate"]
    # a proven rank proves the gap order sufficient
    ev = [{"var": "A", "time": 0, "value": "d", "method": "add"},
          {"var": "A", "time": 5, "value": "d", "method": "add"}]
    r = si.detect(si.Model({"A": "-k*A"}, {"y": "s*A"}, trafo={"A": "0"}, events=ev), reconstruct=True)
    assert r.info["rank_status"]["status"] == "proven" and r.info["gap_certified"] is True
    assert r.info["limits"] == []


def test_the_bound_needs_proven_orders():
    for x0 in (None, "0"):
        r = si.detect(_late_switch(x0), reconstruct=True, control=CAP1)
        assert r.unproven() and r.info["bound"]["probability"] is None
        assert r.info["bound"]["why"].endswith(", and the check runs at the order reached")
        text = " ".join(repr(r).split())
        assert "checked = in the kernel at random points" in text and "no bound on the chance" in text
        assert {s.level for s in r.unproven()} == {"checked"}


def _fields_model():
    return si.Model({"A": "-k1*A + k2*B", "B": "k1*A - k2*B"}, {"Aobs": "alpha*A"}, trafo={"k1": "k2*B/A"},
                    events=[{"var": "A", "time": 0, "value": "1", "method": "add"}])


def test_a_rank_through_invariant_fields_is_verified():
    r = si.detect(_fields_model(), reconstruct=True)
    assert r.info["rank_fields"] == 1 and r.info["rank_proven"] is None
    b = r.info["rank_fields_bound"]
    assert b["prime"] not in PRIMES and b["points"] == 2 and b["degree"] > 0
    assert r.info["rank_status"]["status"] == "verified" and r.info["rank_status"]["probability"] == b["probability"]
    assert 0 < b["probability"] < 1e-10
    assert "invariant fields hold the kernel at every order" in " ".join(repr(r).split())
    assert "rank: verified (1 directions, invariant fields at random points, probability" in str(r.summary())


def test_a_field_wrong_over_q_but_right_modulo_the_engine_prime_does_not_verify_the_rank(monkeypatch):
    from symident.detection import fields
    from symident.detection.observability import analysis
    setup = fields.field_setup

    def shifted(*a, **k):
        out = setup(*a, **k)
        out["gens"] = [[dict(x, alpha=f"({x.get('alpha', '0')}) + {PRIMES[0]}*alpha") for x in gl]
                       for gl in out["gens"]]
        return out
    monkeypatch.setattr(fields, "field_setup", shifted)
    monkeypatch.setattr(analysis, "field_setup", shifted)
    r = si.detect(_fields_model(), reconstruct=True)
    assert r.info["rank_fields"] is None and r.info["rank_fields_bound"] is None
    assert "invariant fields" not in r.info["rank_status"]["why"]


def test_the_gap_guard_finds_a_rank_past_the_gap_order_used():
    r = si.detect(_late_switch(), reconstruct=True, control=CAP1)
    v = r.info["verification"]
    assert v["ok"] is False and v["method"] == "gap guard" and v["kernel_rank_extended"] > r.rank
    assert r.info["rank_status"]["why"] == "the rank grows past the orders used"


def test_identifiable_false_follows_the_rank_status():
    r = si.detect(_late_switch(), reconstruct=True, verify=False, control=CAP1)
    assert r.identifiable is False and r.info["rank_status"]["status"] == "lower bound"
    assert "identifiable" in si.Symmetries.__doc__ and "rank_status" in si.Symmetries.__doc__


def test_the_field_check_bounds_its_random_points():
    from symident.sym.fields import verify_distribution
    p, q = PRIMES[0], 1000000007

    def check(g, field, vec):
        lift = {"regime": 0, "point": {"x": 3, "k": 5, "s": 7}, "vectors": [vec], "times": []}
        return verify_distribution([{"f": ["x = -k*x"], "g": [g]}], [[field]], [], [lift], p, q=q)
    out = check("y = s*x", {"x": "x", "s": "-s"}, {"x": 3, "s": p - 7})
    assert out["ok"] is True and out["bound"]["prime"] == q and 0 < out["bound"]["probability"] < 1e-12
    assert check("y = s*x", {"x": "x", "s": f"-s + {p}*s"}, {"x": 3, "s": p - 7})["ok"] is False


def test_a_kernel_check_on_part_of_the_entries_is_checked_not_verified(monkeypatch):
    from symident.detection import finalize, verify
    entry = {"vector": {"a": "1"}}
    monkeypatch.setattr(verify, "_in_nullspace_at", lambda *a: "partial")
    assert verify.verify_in_nullspace(entry, 0, ["a"], [], [1], 1, None, None, 0, 1)
    assert entry["partial_check"]
    assert finalize.level_of(entry, entry["vector"]) == "checked"
    full = {"vector": {"a": "1"}}
    monkeypatch.setattr(verify, "_in_nullspace_at", lambda *a: True)
    assert verify.verify_in_nullspace(full, 0, ["a"], [], [1], 1, None, None, 0, 1)
    assert finalize.level_of(full, full["vector"]) == "verified"
    assert finalize.level_of(dict(entry, exact=True), entry["vector"]) == "exact"

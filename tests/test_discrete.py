"""Discrete symmetries: the other points of the domain with the output of a generic one."""

import pytest
import sympy as spy

import symident as si
from symident.sym.msolve import MsolveMissingError, _msolve_binary


def _has_msolve():
    try:
        _msolve_binary()
    except MsolveMissingError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _has_msolve(), reason="needs msolve")


def same(a, b):
    return spy.simplify(spy.sympify(a) - spy.sympify(b)) == 0


def two_compartment():
    return si.Model({"x1": "-(k21 + k01)*x1 + k12*x2 + u", "x2": "k21*x1 - k12*x2"}, {"y": "x2"},
                    known_inputs=["u"])


def test_two_compartment_exchanges_the_elimination_rates():
    r = si.detect(two_compartment(), cores=1)
    out = r.discrete(points=[{"x1": 1, "x2": 2, "k01": 3, "k12": 5, "k21": 7},
                             {"x1": 1, "x2": 2, "k01": 4, "k12": 4, "k21": 7}])
    (s,) = out["symmetries"]
    # the coordinates SIAN.jl reports locally but not globally identifiable
    assert s["level"] == "exact" and set(s["changes"]) == {"x1", "k01", "k12"}
    assert same(s["map"]["k01"], "k12") and same(s["map"]["k12"], "k01")
    assert same(s["map"]["x1"], "x1 + (k01 - k12)*x2/k21")
    # sigma(theta) is in the domain while its initial value of x1 stays positive
    assert s["preserves_domain"] is None and len(s["domain_conditions"]) == 1
    assert [spy.sympify(e) for e in s["fixed_set"]] == [spy.sympify("k01 - k12")]
    info = out["info"]
    assert info["count"] == 2 and info["count_final"] and info["order_stable"]
    assert info["fundamental_domain"] == ["k01 < k12"]
    # on the fixed set the two solutions meet
    assert [c["in_domain"] for c in info["counts"]] == [2, 1]
    assert r.discrete_symmetries == out["symmetries"]


def test_a_start_rational_in_the_coordinates():
    def run(trafo):
        m = two_compartment()
        m.trafo = trafo
        return si.detect(m, cores=1).discrete()["symmetries"]
    (s,) = run({"x1": "1/k21", "x2": "0"})
    assert set(s["changes"]) == {"k01", "k12"}
    assert run({"x1": "1/(k01 + 1)", "x2": "0"}) == []


def test_a_given_start_takes_the_state_map_solved_from_the_outputs():
    # the swap needs x1 -> x1 + (k01 - k12)*x2/k21, which maps the start 0 to itself; the identity fails
    m = two_compartment()
    m.trafo = {"x1": "0", "x2": "0"}
    out = si.detect(m, cores=1).discrete()
    (s,) = out["symmetries"]
    assert s["level"] == "exact" and set(s["changes"]) == {"k01", "k12"} and out["info"]["count_final"]


def test_a_symmetry_outside_the_domain_is_not_relevant():
    r = si.detect(si.Model({"x": "-theta^2*x"}, {"y": "x"}), cores=1, discrete={"points": 2})
    (s,) = r.discrete_symmetries
    assert same(s["map"]["theta"], "-theta") and s["preserves_domain"] is False and s["level"] == "exact"
    info = r.info["discrete"]
    assert info["count"] == 2 and info["fundamental_domain"] == []
    assert all(c["real"] == 2 and c["in_domain"] == 1 for c in info["counts"])


def test_the_count_in_the_domain_differs_between_regions():
    m = si.Model({"x": "-a*x", "z": "-(b - a)^2*z"}, {"y1": "x", "y2": "z"}, trafo={"x": "1", "z": "1"})
    out = si.detect(m, cores=1).discrete(points=[{"a": 2, "b": 1}, {"a": 1, "b": 3}])
    (s,) = out["symmetries"]
    assert same(s["map"]["b"], "2*a - b") and s["level"] == "exact"
    assert s["preserves_domain"] is None and [spy.sympify(c.split(">")[0]) for c in s["domain_conditions"]] == \
        [spy.sympify("2*a - b")]
    assert [c["in_domain"] for c in out["info"]["counts"]] == [2, 1]


def test_rates_identified_only_as_a_set():
    # HIV model of Rey Barreiro and Villaverde (2023): c, k1 + w1 and w2 enter only through symmetric functions
    m = si.Model({"x1": "-b*x1*x4 - d*x1 + s", "x2": "b*q1*x1*x4 - w1*x2 - k1*x2",
                  "x3": "b*x1*q2*x4 - x3*w2 + k1*x2", "x4": "-c*x4 + k2*x3"}, {"y1": "x1", "y2": "x4"})
    r = si.detect(m, cores=1, reconstruct=True)
    assert not r.identifiable
    out = r.discrete(points=1)
    info = out["info"]
    assert info["count"] == 6 and info["count_final"] and "reduced" in info
    assert len(out["symmetries"]) == 5 and all(s["level"] == "exact" for s in out["symmetries"])
    # the three rates in the coordinates of the reduced model, ordered by the fundamental domain
    rates = {spy.sympify(s["map"]["c"]) for s in out["symmetries"] if "c" in s["map"]} | {spy.Symbol("c")}
    # SIAN.jl reports c and w2 locally but not globally identifiable; k1 and w1 alone are not identifiable
    assert {"c", "w2"} <= {x for s in out["symmetries"] for x in s["changes"]}
    assert len(rates) == 3
    assert len(info["fundamental_domain"]) == 3
    red = r.reduce()
    assert red.discrete is not None and red.discrete["fundamental_domain"] == info["fundamental_domain"]


def test_a_square_outside_the_domain_leaves_the_model_globally_identifiable():
    # beta-cell model of Finegood et al. (1995), as in SIAN.jl: SIAN reports alpha locally but not globally
    # identifiable over the complex numbers, from alpha -> -alpha
    m = si.Model({"G": "u1 - (c + si*II)*G",
                  "beta": "beta*((1/(48*24*60))/(1 + (8/G)^2) - (1/(40*24*60))/(1 + (G/5)^8))",
                  "II": "p1*beta*(G^2/(alpha^2 + G^2)) - gm*II"}, {"y1": "G"}, known_inputs=["u1"])
    r = si.detect(m, cores=1, reconstruct=True)
    out = r.discrete(points=2)
    (s,) = out["symmetries"]
    assert s["changes"] == ["alpha"] and same(s["map"]["alpha"], "-alpha")
    assert s["level"] == "exact" and s["preserves_domain"] is False
    info = out["info"]
    assert info["count"] == 2 and info["count_final"] and info["fundamental_domain"] == []
    assert all(c["real"] == 2 and c["in_domain"] == 1 for c in info["counts"])


def test_a_vanishing_denominator_is_no_point_of_the_fiber():
    # the x4 equation of the p53 model of Rey Barreiro and Villaverde (2023), the other states as known inputs:
    # where p22^4 = -1 a denominator and a numerator vanish together, which the saturation must exclude
    f = {"x4": "p20 - p21*(1-p24)*(1-p25)/((p22^4)+1) - (p20*x4) + (p21*(x3^4))"
               " + (1+p23*u1)*(1-p24*x1)*(1-p25*x2)/(p22^4+x3^4)"}
    m = si.Model(f, {"y4": "x4"}, known_inputs=["u1", "x1", "x2", "x3"])
    out = si.detect(m, cores=1).discrete(points=1)
    (s,) = out["symmetries"]
    assert same(s["map"]["p22"], "-p22") and s["preserves_domain"] is False
    info = out["info"]
    # p22 -> -p22 and the two complex roots of p22^4
    assert info["count"] == 4 and [(c["complex"], c["real"], c["in_domain"]) for c in info["counts"]] == [(4, 2, 1)]


def maps_of(m):
    out = si.detect(m, cores=1).discrete()
    return [(s["map"], s["level"]) for s in out["symmetries"]], out["info"]


def test_conditions_share_the_coordinates_of_the_fiber():
    m = si.Model({"x": "-k*x"}, {"y": "x"}, conditions={"rows": ["a", "b"], "cols": {"k": ["k1", "k2"]}})
    maps, info = maps_of(m)
    assert maps == [] and info["count"] == 1 and info["count_final"]
    assert set(info["orders"]) == {"a:y", "b:y"}
    # observing the first compartment in a second condition removes the swap of the elimination rates
    tc = two_compartment()
    tc.observables = [{"y": "x2"}, {"y": "x1"}]
    maps, info = maps_of(tc)
    assert maps == [] and info["count"] == 1


FREE = {"x1": "-(k21 + k01)*x1 + k12*x2", "x2": "k21*x1 - k12*x2"}


def dosed(var, time="1", value="D", method="add", start=None):
    return si.Model(FREE, {"y": "x2"}, trafo=start or {"x1": "0", "x2": "0"}, fixed=["D"] if "D" in value else (),
                    events=[{"var": var, "time": time, "value": value, "method": method}])


def test_a_dose_after_rest_keeps_or_breaks_the_swap():
    # the impulse response from the first compartment is symmetric in k01 and k12, the one from the second is not
    maps, info = maps_of(dosed("x1"))
    (sig, level), = maps
    assert level == "exact" and same(sig["k01"], "k12") and info["count"] == 2 and info["count_final"]
    maps, info = maps_of(dosed("x2"))
    assert maps == [] and info["count"] == 1


def test_an_estimated_event_time_is_a_coordinate_of_the_fiber():
    # the segment after the dose is expanded at the start of the condition, so its series holds tau
    maps, info = maps_of(dosed("x1", time="tau"))
    (sig, level), = maps
    assert level == "exact" and "tau" not in sig and info["count"] == 2


def test_a_map_that_the_truncated_fiber_keeps_is_dropped_further_up():
    # the swap keeps the equations up to eps^1 after the reset but not x1/(1 + x2) at eps^2: the orders rise
    maps, info = maps_of(dosed("x1", value="x1/(1 + x2)", method="replace", start={"x1": "1", "x2": "0"}))
    assert maps == [] and info["count"] == 1 and info["gap_order"] >= 2


def test_detect_in_global_mode_counts_after_the_local_analysis():
    r = si.detect(two_compartment(), mode="global", global_options={"points": 1})
    (s,) = r.discrete_symmetries
    assert s["level"] == "exact" and r.info["discrete"]["count"] == 2
    assert "Global:" in repr(r) and "sigma1 (exact)" in repr(r)
    # a non-identifiable model is reduced first
    r = si.detect(si.Model({"x": "-k*s*x", "z": "x"}, {"y": "z"}), mode="global")
    assert r.discrete_symmetries == [] and r.info["discrete"]["reduced"]
    # a setting the global analysis does not handle warns and keeps the local result
    with pytest.warns(UserWarning, match="global analysis not done"):
        r = si.detect(si.Model({"x": "-k*x + w"}, {"y": "x"}, unknown_inputs=["w"]), mode="global")
    assert r.rank is not None and "unknown inputs" in r.info["discrete"]["unsupported"]
    with pytest.raises(ValueError, match="mode"):
        si.detect(two_compartment(), mode="globally")
    with pytest.warns(DeprecationWarning):
        si.detect(two_compartment(), discrete={"points": 1})


def test_the_json_interface_runs_the_global_mode():
    import json

    from symident import rjson
    spec = {"f": {"x1": "-(k21 + k01)*x1 + k12*x2 + u", "x2": "k21*x1 - k12*x2"}, "g": {"y": "x2"},
            "known_inputs": ["u"], "mode": "global", "global_options": {"points": 1}}
    out = json.loads(rjson.detect_json(json.dumps(spec)))
    assert out["info"]["discrete"]["count"] == 2 and len(out["discrete_symmetries"]) == 1


def test_a_start_at_rest_takes_the_resting_state_in_closed_form():
    # at rest with a production into the first compartment, the output k21*p/(k01*k12) keeps the swap
    tc = {"x1": "p - (k21 + k01)*x1 + k12*x2", "x2": "k21*x1 - k12*x2"}
    m = si.Model(tc, {"y": "x2"}, events=[{"var": "x1", "time": 1, "value": "D", "method": "add"}], fixed=["D"])
    r = si.detect(m, equilibrate=True, mode="global")
    (s,) = r.discrete_symmetries
    assert s["level"] == "exact" and same(s["map"]["k01"], "k12") and r.info["discrete"]["count_final"]
    # a state the observables do not read, which never rests, is left out
    m = si.Model({"x": "p - d*x", "z": "k*x"}, {"y": "x"}, fixed=["D"],
                 events=[{"var": "x", "time": 0, "value": "D", "method": "add"}])
    r = si.detect(m, equilibrate=True, mode="global")
    assert r.discrete_symmetries == [] and r.info["discrete"]["count"] == 1


def test_a_switch_enters_with_copies_that_start_on_the_surface():
    # the threshold a is fixed where a copy starts on the surface x = a
    m = si.Model({"x": "piecewise(-k1*x, x > a, -k2*x)"}, {"y": "x"}, trafo={"x": "x0"})
    r = si.detect(m, mode="global")
    assert r.discrete_symmetries == [] and r.info["discrete"]["count"] == 1
    # k -> -k keeps every regime and the surface; it leaves the domain
    m = si.Model({"x": "piecewise(-k^2*x, x > a, -c*x)"}, {"y": "x"}, trafo={"x": "x0"})
    r = si.detect(m, mode="global")
    (s,) = r.discrete_symmetries
    assert s["level"] == "exact" and same(s["map"]["k"], "-k") and s["preserves_domain"] is False
    with pytest.warns(UserWarning, match="parameters alone"):
        si.detect(si.Model({"x": "piecewise(-k1*x, k1 > k2, -k2*x)"}, {"y": "x"}, regimes=["k1 - k2 > 0"]),
                  mode="global")


def _swap_model(k01="k01", k12="k12", k21="k21"):
    # the swap of the elimination rates, and an independent scaling c, d that the section pins
    f = {"x1": f"-({k21} + {k01})*x1 + {k12}*x2 + u", "x2": f"{k21}*x1 - {k12}*x2", "z": "-c*d*z"}
    return si.Model(f, {"y": "x2", "y2": "z"}, known_inputs=["u"], trafo={"z": "1"})


def test_each_side_of_a_discrete_symmetry_is_a_branch_chart():
    import json

    from symident import rjson
    r = si.detect(_swap_model(), mode="global", global_options={"points": 1})
    red = r.reduce()
    bs = {b["branch"]: b for b in red.discrete["branches"]}
    assert set(bs) == {"k01 > k12", "k12 > k01"} and red.trafo == {**red.trafo, "c": "1"}
    # free values in the positive orthant onto the side, chained with the section c = 1
    b = bs["k12 > k01"]
    assert same(b["solved"]["k12"], "k01 + delta") and b["injective"] and b["reach"] and not b["unproven"]
    assert b["boundary"] == ["k01 - k12 = 0"] and b["trafo"]["c"] == "1"
    # without a choice the branches are alternatives; with one, its chart is the trafo
    assert {a["branch"] for a in red.alternatives if a.get("branch")} == set(bs)
    chosen = r.reduce(branch="k01 < k12")
    assert same(chosen.trafo["k12"], "k01 + delta") and chosen.checks["loop"]["ok"]
    with pytest.raises(si.errors.InputError, match="branches are"):
        r.reduce(branch="k21 > k01")
    out = json.loads(rjson.reduce_json(json.dumps(r.to_dict()), json.dumps({"branch": "k12 > k01"})))
    assert same(out["trafo"]["k12"], "k01 + delta")


def test_the_branches_do_not_depend_on_the_names():
    r = si.detect(_swap_model("z9", "c4", "p3"), mode="global", global_options={"points": 1})
    bs = r.reduce().discrete["branches"]
    assert {b["branch"] for b in bs} == {"z9 > c4", "c4 > z9"}
    assert all(b["injective"] and b["reach"] for b in bs)


CHAIN = {"R1": "b*u - sh*R1", "R2": "sh*R1 - sh*R2", "R": "sh*R2 - kR*R", "A": "c*R - kA*A"}
REST = {"R1": "0", "R2": "0", "R": "0", "A": "0"}


def test_a_chain_swaps_the_decay_rates_of_its_last_two_members():
    # the response of A to u is symmetric in kR and kA
    r = si.detect(si.Model(CHAIN, {"y": "A"}, known_inputs=["u"], trafo=REST), mode="global")
    (s,) = r.discrete_symmetries
    assert s["level"] == "exact" and s["map"] == {"kA": "kR", "kR": "kA"} and "candidates" in s["found_by"]
    (phi,) = s["state_maps"]
    assert set(phi) == {"R"} and same(phi["R"], "R + (kR - kA)*A/c")
    info = r.info["discrete"]
    assert info["count"] == 2 and info["count_final"] and info["candidates"]["proven"] == 1
    cert = info["certificate"]
    assert cert["orbit"] == 2 and cert["complete"] == "axiom" and cert["invariance"][0]["proven"]


def test_log_coordinates_and_a_bleaching_log_output_are_rewritten():
    tr = {**REST, **{k: f"exp({k})" for k in ("b", "c", "sh", "kR", "kA", "s", "off", "tau")}}
    m = si.Model(CHAIN, {"y": "log(s*A + off) - tau*time"}, known_inputs=["u"], trafo=tr)
    r = si.detect(m, mode="global")
    (s,) = r.discrete_symmetries
    assert s["level"] == "exact" and s["map"] == {"kA": "kR", "kR": "kA"} and s["fixed_set"] == ["kA - kR"]
    info = r.info["discrete"]
    assert info["count_final"] and info["fundamental_domain"] == ["kA < kR"]
    assert {w["rewrite"] for w in info["rewrites"]} == {"coordinate", "observable", "time"}
    obs = next(w for w in info["rewrites"] if w["rewrite"] == "observable")
    assert same(obs["to"].replace("^", "**"), "(s*A + off)*exp(-tau*time)")
    # the section of the scaling b, c is the translation of the log coordinates to zero
    assert info["reduced"]["section"]["b"] == "0" or info["reduced"]["section"]["c"] == "0"


def test_a_term_that_stays_transcendental_is_named():
    with pytest.raises(ValueError, match=r"sin\(a\)"):
        si.detect(si.Model({"x": "-k*x + sin(a)*x^2"}, {"y": "x"}), cores=1).discrete()


def test_maps_of_log_coordinates_go_back_as_translations():
    from symident.detection.rationalize import map_to_given, relation_to_given
    a, b, c = spy.symbols("a b c")
    logs = {"a": (spy.E, 1), "b": (spy.E, 1)}
    out = map_to_given({a: 2 * b, b: a / 2, c: c * a}, logs)
    assert same(out[a], "b + log(2)") and same(out[b], "a - log(2)") and same(out[c], "c*exp(a)")
    assert relation_to_given("a - 2*b", logs) in ("a - b - log(2)", "-b + a - log(2)") or \
        same(relation_to_given("a - 2*b", logs), "a - b - log(2)")


def test_the_reduction_of_the_fiber_is_the_same_in_cpp_and_python():
    import random

    from symident.detection import discrete
    p = 1000003
    rng = random.Random(7)
    for _ in range(30):
        nv, m0 = 12, 4

        def poly():
            P = {}
            for _ in range(rng.randint(1, 3)):
                mm = tuple(sorted({v: rng.randint(1, 2) for v in rng.sample(range(nv), rng.randint(0, 2))}.items()))
                P[mm] = rng.randint(1, p - 1)
            return P
        # equations a*v + b that fix variables, and jets that one equation holds alone
        polys = [poly() for _ in range(8)] + [{((v, 1),): rng.randint(1, p - 1), (): rng.randint(0, p - 1)}
                                              for v in rng.sample(range(nv), 3)]
        polys += [{((v, 1),): 1, **poly()} for v in rng.sample(range(m0, nv), 2)]
        sat = [{((v, 1),): 1} for v in rng.sample(range(nv), 2)]
        a = discrete._reduce_mod(polys, sat, nv, m0, p)
        b = discrete._reduce_q(polys, sat, nv, m0, p)
        assert (a is None) == (b is None)
        if a is not None:
            assert sorted(map(sorted, map(dict.items, a[0]))) == sorted(map(sorted, map(dict.items, b[0])))
            assert a[1:] == b[1:]

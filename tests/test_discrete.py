"""Discrete symmetries: the other points of the domain with the output of a generic one."""

import math
from fractions import Fraction

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
    r = si.detect(si.Model({"x": "-theta^2*x"}, {"y": "x"}), cores=1, mode="global",
                  global_options={"points": 2})
    (s,) = r.discrete_symmetries
    assert same(s["map"]["theta"], "-theta") and s["preserves_domain"] is False and s["level"] == "exact"
    info = r.info["discrete"]
    assert info["count"] == 2 and info["fundamental_domain"] == [] and info["count_domain"] == 1
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
    # the exact count finds no relation of this rational model: a small search keeps the test fast
    out = r.discrete(points=2, search_entries=10 ** 7)
    (s,) = out["symmetries"]
    assert s["changes"] == ["alpha"] and same(s["map"]["alpha"], "-alpha")
    assert s["level"] == "exact" and s["preserves_domain"] is False
    info = out["info"]
    assert info["count"] == 2 and info["count_final"] and info["fundamental_domain"] == []
    assert all(c["real"] == 2 and c["in_domain"] == 1 for c in info["counts"])


def _stage_toy(monkeypatch, d):
    """A block in x0, x1, x2 with the stage x0 + x1 = 0, 4 x0 x1 + d = 0 and x2 = x0, and the calls of
    _stage_points as (p, found)."""
    import types

    from symident.detection import discrete
    fib = types.SimpleNamespace(names=["x0", "x1", "x2"], allnames=["x0", "x1", "x2"], sep_mode={})
    fib._ms_block = lambda idx, polys, sat, p: discrete.Fiber._ms_block(fib, idx, polys, sat, p)
    polys = [{((0, 1),): 1, ((1, 1),): 1}, {((0, 1), (1, 1)): 4, (): d}, {((2, 1),): 1, ((0, 1),): -1}]
    calls, orig = [], discrete._stage_points

    def record(*a):
        got = orig(*a)
        calls.append((a[4], got is not None))
        return got
    monkeypatch.setattr(discrete, "_stage_points", record)
    return discrete, fib, polys, calls


def test_a_block_that_msolve_counts_at_once_is_not_split_into_stages(monkeypatch):
    discrete, fib, polys, calls = _stage_toy(monkeypatch, 4)
    assert discrete._staged_count(fib, [0, 1, 2], polys, [], discrete.PRIMES[0]) == 2
    assert calls == []


def test_a_block_past_an_eighth_of_the_cap_of_msolve_goes_by_stages(monkeypatch):
    from symident.detection import work
    discrete, fib, polys, calls = _stage_toy(monkeypatch, 4)
    caps = []
    orig = discrete._block_count

    def record(*a):
        caps.append(work.cap("msolve_entries"))
        return None if len(caps) == 1 else orig(*a)
    monkeypatch.setattr(discrete, "_block_count", record)
    with work.caps(msolve_entries=800):
        assert discrete._staged_count(fib, [0, 1, 2], polys, [], discrete.PRIMES[0]) == 2
    # the block at once within an eighth of the cap, then the stage and the rest within the whole cap
    assert caps[0] == 100 and calls == [(discrete.PRIMES[0], True)]


def test_a_stage_with_points_outside_gf_p_is_not_solved_over_q(monkeypatch):
    discrete, fib, polys, calls = _stage_toy(monkeypatch, 0)
    p = discrete.PRIMES[0]
    # d a non-residue modulo p: the points of the stage, x0 = +-sqrt(d)/2, lie outside GF(p) and Q
    polys[1][()] = next(d for d in range(2, 100) if pow(d, (p - 1) // 2, p) == p - 1)
    seen, orig = set(), discrete._block_count

    def first_fails(fib_, idx, polys_, sat, p_):
        # the first count at once of a block over Q fails, as on a block too large for it
        if p_ == 0 and tuple(idx) not in seen:
            seen.add(tuple(idx))
            return None
        return orig(fib_, idx, polys_, sat, p_)
    monkeypatch.setattr(discrete, "_block_count", first_fails)
    assert discrete._staged_count(fib, [0, 1, 2], polys, [], 0) == 2
    assert calls == [(p, False)]


def test_a_vanishing_denominator_is_no_point_of_the_fiber():
    # the x4 equation of the p53 model of Rey Barreiro and Villaverde (2023), the other states as known inputs:
    # where p22^4 = -1 a denominator and a numerator vanish together, which the saturation must exclude
    f = {"x4": "p20 - p21*(1-p24)*(1-p25)/((p22^4)+1) - (p20*x4) + (p21*(x3^4))"
               " + (1+p23*u1)*(1-p24*x1)*(1-p25*x2)/(p22^4+x3^4)"}
    m = si.Model(f, {"y4": "x4"}, known_inputs=["u1", "x1", "x2", "x3"])
    from symident.detection import work
    # a short search and few steps of the monodromy keep the test fast
    with work.caps(monodromy_products=10 ** 7):
        out = si.detect(m, cores=1).discrete(points=1, search_entries=10 ** 7)
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
    assert "the bound covers models without events" in info["certificate"]["count_bound_why"]


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
    dose = [{"var": "x", "time": 1, "value": 1, "method": "add"}]
    with pytest.warns(UserWarning, match="global analysis not done"):
        r = si.detect(si.Model({"x": "-k*x + w"}, {"y": "x", "yw": "w"}, unknown_inputs=["w"], events=dose),
                      mode="global")
    assert r.rank is not None and "unknown inputs" in r.info["discrete"]["unsupported"]
    with pytest.raises(ValueError, match="mode"):
        si.detect(two_compartment(), mode="globally")


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
    assert cert["orbit"] == 2 and cert["invariance"][0]["proven"]
    # the coefficients of the input-output relation of A and u lie in the field of the outputs; c, sh and the
    # symmetric functions of kR and kA follow from them
    assert cert["complete"] == "exact" and cert["count_bound"] is None
    assert any(r.endswith("proven") for r in cert["exact"]["relations"])


def test_log_coordinates_and_a_bleaching_log_output_are_rewritten():
    tr = {**REST, **{k: f"exp({k})" for k in ("b", "c", "sh", "kR", "kA", "s", "off", "tau")}}
    m = si.Model(CHAIN, {"y": "log(s*A + off) - tau*time"}, known_inputs=["u"], trafo=tr)
    r = si.detect(m, mode="global", global_options={"search_entries": 10 ** 7})
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



def test_a_fiber_whose_invariants_are_output_functions_is_exact():
    r = si.detect(si.Model({"x": "-k^2*x"}, {"y": "x"}, trafo={"x": "x0"}), mode="global", cores=1)
    info = r.info["discrete"]
    cert = info["certificate"]
    assert info["count"] == 2 and cert["orbit"] == 2 and cert["complete"] == "exact" and info["count_final"]
    # x0 and k^2 generate the field fixed by k -> -k; both are rational in the Taylor coefficients
    assert {g["function"] for g in cert["exact"]["generators"]} == {"x0", "-k**2"}
    assert cert["count_bound"] is None and "axiom" not in cert and "proven" in repr(r)


def test_a_known_constant_is_a_symbol_of_the_base_field_of_the_exact_count():
    m = si.Model({"x": "-k^2*D*x"}, {"y": "x"}, trafo={"x": "x0"}, fixed=["D"])
    cert = si.detect(m, mode="global", cores=1).info["discrete"]["certificate"]
    # k^2 D and D known give k^2
    assert cert["complete"] == "exact" and cert["orbit"] == 2


def test_a_point_where_the_fiber_leaves_the_domain_counts_none_and_is_not_verified(monkeypatch):
    from symident.detection import discrete, fieldproof
    monkeypatch.setattr(fieldproof, "exact_fiber",
                        lambda *a, **k: {"exact": False, "why": "off", "generators": [], "seconds": 0.0})
    m = si.Model({"x": "-k^2*x"}, {"y": "x"}, trafo={"x": "x0"})
    cert = si.detect(m, mode="global", cores=1).info["discrete"]["certificate"]
    assert cert["complete"] == "verified" and cert["count_bound"] * 2 ** cert["count_bits"] >= 1
    orig = discrete.q_count

    def at_zero(fib, point):
        # at k = 0 both points k and -k meet the zero of the coordinate, which the saturation removes
        return orig(fib, {**point, spy.Symbol("k"): spy.Integer(0)})
    monkeypatch.setattr(discrete, "q_count", at_zero)
    cert = si.detect(m, mode="global", cores=1).info["discrete"]["certificate"]
    assert cert["complete"] == "axiom" and cert["count_bound"] is None
    assert "no finite count over Q at the random integer point" in cert["count_bound_why"]


def test_the_series_on_degree_bounds_shares_denominators():
    from symident.sym.degrees import Atoms, Deg, common
    A = Atoms()
    a, b = Deg(A, 1), Deg(A, 1)
    d = a - b
    # (a + b)/(a - b) + 1/(a - b) has the one denominator a - b
    v = (a + b) / d + 1 / d
    assert v.dn == 1 and list(v.den.values()) == [1] and A.of(v.den) == 1
    w = (a + b) / d
    assert common([w, w * w]) == (2, [2, 2])
    # numbers stay exact, and only an exact zero compares equal to zero
    assert (Deg.const(A, Fraction(1, 3)) * 3).c == 1 and (a - a) != 0 and (a * 0) == 0


def test_the_fixed_field_of_a_swap_is_generated_by_the_symmetric_functions():
    from symident.detection.fieldproof import generators
    a, b, c = spy.symbols("a b c")
    grp = [{}, {a: b, b: a}]
    alpha, F, traces, others = generators(grp, [a, b, c])
    # a has two distinct images; the coefficients of (Z - a)(Z - b) and the fixed c generate F
    assert alpha == a and others == [b]
    assert {spy.expand(f) for f in F} == {c, -a - b, a * b} and [spy.expand(t) for t in traces] == [a + b]


def test_a_bezout_number_past_the_bits_of_the_point_leaves_the_axiom(monkeypatch):
    from symident.detection import discrete, fieldproof
    monkeypatch.setattr(fieldproof, "exact_fiber",
                        lambda *a, **k: {"exact": False, "why": "off", "generators": [], "seconds": 0.0})
    monkeypatch.setattr(discrete, "_fiber_degrees", lambda fib: [10 ** 2000, 10 ** 2000])
    monkeypatch.setattr(discrete, "_x_bezout", lambda fib: None)
    cert = si.detect(si.Model({"x": "-k^2*x"}, {"y": "x"}, trafo={"x": "x0"}), mode="global",
                     cores=1).info["discrete"]["certificate"]
    assert cert["complete"] == "axiom" and "bits" in cert["count_bound_why"]


def test_msolve_output_over_q_is_read_exactly():
    from fractions import Fraction

    from symident.detection.discrete import parse_rational
    digits = "1" + "0" * 5000
    # quotients past the range of a float and integers past the default digit limit stay exact
    got = parse_rational("[0, [1, " + digits + "/3, -7 / 2^3, " + digits + "]]")
    assert got == [0, [1, Fraction(10 ** 5000, 3), Fraction(-7, 8), 10 ** 5000]]


def test_msolve_output_with_long_integers_is_read_in_linear_time():
    import time

    from symident.detection.discrete import parse_rational
    n = 200000
    t = time.monotonic()
    got = parse_rational("[" + "7" * n + ", -" + "7" * n + " / 2^3]")
    value = 7 * (10 ** n - 1) // 9
    assert got[0] == value and got[1] * 8 == -value
    assert time.monotonic() - t < 10


def test_a_polynomial_in_more_symbols_than_the_recursion_limit_is_read():
    from symident.detection.discrete import _dict
    # the jets of known inputs give a part a thousand symbols and more
    xs = spy.symbols("x0:1100")
    P = _dict(xs[0] * xs[1099] / 2 + 3, xs)
    assert P == {tuple(int(i in (0, 1099)) for i in range(1100)): Fraction(1, 2), (0,) * 1100: Fraction(3)}


def test_a_verified_count_at_a_point_of_thousands_of_digits(monkeypatch):
    from symident.detection import discrete, fieldproof
    monkeypatch.setattr(fieldproof, "exact_fiber",
                        lambda *a, **k: {"exact": False, "why": "off", "generators": [], "seconds": 0.0})
    # the Taylor coefficients at the point pass the default limit of digits of int()
    monkeypatch.setattr(discrete, "_MIN_BITS", 4000)
    cert = si.detect(si.Model({"x": "-k^2*x"}, {"y": "x"}, trafo={"x": "x0"}), mode="global",
                     cores=1).info["discrete"]["certificate"]
    assert cert["complete"] == "verified" and cert["count_bits"] == 4000


def test_a_parameter_read_only_as_a_fourth_power_has_maps_over_q_of_i_from_monodromy():
    # k and -k are rational maps; i*k and -i*k have the output of k as well, over Q(i)
    r = si.detect(si.Model({"x": "-k^4*x"}, {"y": "x"}, trafo={"x": "x0"}), mode="global", cores=1)
    info = r.info["discrete"]
    assert info["count"] == 4 and info["certificate"]["orbit"] == 4 and info["certificate"]["complete"] == "exact"
    assert not any(lim.startswith("k enters only as k^4") for lim in info["limits"])
    over_i = [s for s in r.discrete_symmetries if "monodromy" in s["found_by"]]
    assert sorted((spy.sympify(s["map"]["k"]) / spy.Symbol("k") for s in over_i), key=str) == \
        sorted([spy.I, -spy.I], key=str)
    assert all(s["level"] == "exact" and s["preserves_domain"] is False for s in over_i)
    assert info["monodromy"]["certified"] == 4


def _relaxed(odes, obs, **kw):
    r = si.detect(si.Model(odes, obs, **kw), mode="global")
    return r.info["discrete"], r.discrete_symmetries


def test_an_exponential_of_a_state_is_counted_on_the_relaxed_model():
    info, _ = _relaxed({"x": "-k*exp(x/c)"}, {"y": "x"})
    (lf,) = info["relaxed"]["leaves"]
    assert lf["kind"] == "exp" and same(lf["start"], "exp(x/c)")
    assert info["count"] == 1 and info["certificate"]["complete"] == "exact"


def test_a_free_power_drops_the_coordinate_its_constant_term_determines():
    info, _ = _relaxed({"x": "-k*x^n/(K^n + x^n)"}, {"y": "x"})
    assert list(info["relaxed"]["dropped"]) == ["K"]
    assert info["count"] == 1 and info["certificate"]["complete"] == "exact"


def test_a_logarithm_of_a_state_is_an_independent_leaf():
    info, _ = _relaxed({"x": "-k*x + p*log(x)"}, {"y": "x"})
    assert info["relaxed"]["premise"] is None
    assert info["count"] == 1 and info["certificate"]["complete"] == "exact"


def test_a_map_of_the_relaxed_model_that_keeps_the_leaves_is_a_map_of_the_model():
    info, syms = _relaxed({"x1": "-(k1 + k2)*x1 - a*x1*exp(x1/c)", "x2": "k1*k2*x1"}, {"y1": "x1", "y2": "x2"})
    assert info["count"] == 2 and info["certificate"]["complete"] == "exact"
    (s,) = syms
    assert s["level"] == "exact" and same(s["map"]["k1"], "k2")


def test_a_leaf_the_output_does_not_fix_leaves_the_count_open():
    info, _ = _relaxed({"x1": "-(k1 + k2)*x1*exp(x1/c)", "x2": "k1*k2*x1"}, {"y1": "x1", "y2": "x2"})
    assert info["certificate"]["complete"] == "open" and "new coordinate" in info["relaxed"]["why"]


def test_a_known_input_inside_a_term_becomes_a_state_with_a_known_start():
    info, _ = _relaxed({"x": "-k*x*exp(u/c)"}, {"y": "x"}, known_inputs=["u"])
    assert info["relaxed"]["model"]["trafo"]["u"].startswith("si_u0")
    assert info["count"] == 1 and info["certificate"]["complete"] == "exact"


def test_a_free_power_of_a_state_at_zero_is_counted_on_the_slice_of_its_leading_coefficient():
    # x^a/(g^a + x^a) is N/(P + N) with N = (x/c_x)^a and P = (g/c_x)^a, which determines g
    info, _ = _relaxed({"x": "p - m*x"}, {"y": "x^a/(g^a + x^a)"}, trafo={"x": "0"})
    (lf,) = info["relaxed"]["leaves"]
    assert lf["absorbs"] == "g" and same(lf["start"], "(g/p)^a")
    (sp,) = info["start_powers"]
    assert (sp["state"], sp["exponent"], sp["order"], sp["premises"]) == ("x", "a", 1, [])
    assert info["count"] == 1 and info["certificate"]["complete"] == "verified"


def test_a_free_power_of_a_state_at_zero_in_the_dynamics_of_another_state():
    info, _ = _relaxed({"x": "p - m*x", "z": "x^a/(g^a + x^a) - k*z"}, {"y": "z"}, trafo={"x": "0", "z": "0"})
    (sp,) = info["start_powers"]
    # z = t^(1 + a)/(1 + a) + ...: the lowest exponent other than an integer is at index 2
    assert sp["index"] == 2 and sp["premises"] == []
    assert info["count"] == 1 and info["certificate"]["complete"] == "verified"


def test_a_leading_coefficient_without_a_scaling_is_a_leaf():
    info, _ = _relaxed({"x": "p - m*x"}, {"y": "q*x^a + x"}, trafo={"x": "0"})
    (lf,) = info["relaxed"]["leaves"]
    assert lf["kind"] == "power" and same(lf["start"], "p^a")
    assert info["count"] == 1 and info["certificate"]["complete"] == "verified"


def _start_power_series(g, positive=True):
    from symident.detection.discrete import Series
    seg = {"f": {"x": "p - m*x", "si_N1": "0"}, "g": g, "ic": {"x": "0", "si_N1": "0"}, "t": "0"}
    part = {"segments": [seg], "spow": {"N": "si_N1", "state": "x", "alpha": "a", "k": 1}}
    return Series([part], ["a", "m", "p", "q", "r"], positive)


def test_a_coefficient_without_a_sign_on_the_domain_is_a_premise():
    (sp,) = _start_power_series({"y": "(q - r)*si_N1 + x"}).start_powers
    (why,) = sp["proof"]["premises"]
    assert "not zero on the domain" in why
    (sp,) = _start_power_series({"y": "(q + r)*si_N1 + x"}).start_powers
    assert sp["proof"]["premises"] == [] and sp["proof"]["index"] == 1
    with pytest.raises(ValueError, match="positive"):
        _start_power_series({"y": "si_N1 + x"}, positive=["m", "p", "q", "r"])


def test_the_series_of_a_start_power_has_the_layers_of_the_exponent():
    # y = N = (x/p)^a with x = p t - p m t^2/2: N = S (1 - a m t/2 + ...), S = t^a
    ser = _start_power_series({"y": "si_N1"})
    pt = {spy.Symbol(c): spy.Rational(v) for c, v in zip(ser.coords, (3, 5, 7, 2, 1))}
    (y,) = ser.numeric(pt, 2)
    assert [list(v) for v in y] == [[0, 0, 0], [0, 1, 0], [0, Fraction(-15, 2), 0]]
    assert ser.pins[0][0] == {(1, 0, 0, 0, 0): 1}


def test_the_leading_order_of_a_state_at_zero():
    from symident.detection.startpow import leading
    k, c = leading({"x": "z", "z": "p - m*z + x^a"}, {"x": "0", "z": "0"}, spy.Symbol("x"), spy.Symbol("a"))
    assert k == 2 and same(c, "p/2")


def test_a_power_below_the_order_of_the_base_is_refused():
    # x = q t^(1 + a)/(1 + a) + p t^2/2: below t^2 for a < 1, the expansion has no leading term c t^k
    r = si.detect(si.Model({"x": "z + q*x^a", "z": "p - m*z"}, {"y": "x"}, trafo={"x": "0", "z": "0"}), cores=1)
    with pytest.raises(ValueError, match="below"):
        r.discrete()


def test_a_coordinate_proven_in_the_field_of_the_outputs_pins_the_count(monkeypatch):
    from symident.detection import fieldproof

    def run(proven):
        monkeypatch.setattr(fieldproof, "exact_fiber", lambda *a, **k: {
            "exact": False, "why": "off", "generators": [], "seconds": 0.0, "proven": proven})
        return si.detect(si.Model({"x": "-k^2*x - c*x^3"}, {"y": "x"}, trafo={"x": "x0"}), mode="global",
                         cores=1).info["discrete"]
    free, pinned = run([]), run(["x0", "c"])
    assert free["count"] == pinned["count"] == 2
    # the pins are equations of degree one in place of output coefficients of higher degree
    assert pinned["certificate"]["complete"] == "verified"
    assert pinned["certificate"]["bezout"] < free["certificate"]["bezout"]


def test_a_state_that_starts_at_one_maps_to_a_power_of_itself():
    # z = exp(n k t): n -> -n takes z to 1/z
    r = si.detect(si.Model({"x": "-k*x", "z": "n*k*z"}, {"y1": "x", "y2": "c*(1 + a*z)/(1 + z)"},
                           trafo={"z": "1"}, positive=["x", "k", "c", "a"]), mode="global", cores=1)
    (s,) = r.discrete_symmetries
    assert s["level"] == "exact" and same(s["map"]["n"], "-n") and same(s["map"]["a"], "1/a")
    assert s["state_maps"][0]["z"] == "1/z"


def test_an_equation_that_sets_a_variable_to_a_monomial_eliminates_it():
    from symident.detection import discrete
    p = 1000003
    # 2*x0 - 3*x1*x2 = 0 sets the coordinate x0; x3 is a jet, set to a monomial in a coordinate
    polys = [{((0, 1),): 2, ((1, 1), (2, 1)): p - 3}, {((0, 2),): 1, ((1, 1),): p - 1, ((3, 1),): 1},
             {((3, 1),): 1, ((2, 2),): 7}]
    sat = [{((0, 1), (1, 1)): 1}]
    out, sat2, gone = discrete._monomial_eliminate(polys, sat, 3, p)
    assert gone == {0, 3}
    assert all(v not in gone for P in out + sat2 for mm in P for v, _ in mm)
    # a coordinate does not leave for a monomial that holds a jet
    _, _, gone = discrete._monomial_eliminate([{((0, 1),): 1, ((3, 1), (4, 1)): 1}], [], 3, p)
    assert gone == set()


def test_the_count_after_the_elimination_of_monomial_pins_is_the_same(monkeypatch):
    from symident.detection import fieldproof

    def run(proven):
        monkeypatch.setattr(fieldproof, "exact_fiber", lambda *a, **k: {
            "exact": False, "why": "off", "generators": [], "seconds": 0.0, "proven": proven})
        return si.detect(si.Model({"x1": "-a*b*x1", "x2": "-(a + c)*x2"}, {"y1": "x1 + x2"},
                                  trafo={"x1": "1", "x2": "1"}), mode="global", cores=1).info["discrete"]
    pinned = run(["a*b", "a + c", "b/c"])
    assert pinned["count"] == 2 and pinned["certificate"]["complete"] == "verified"


def test_a_process_past_its_address_space_fails_and_not_the_one_that_waits():
    import sys

    from symident.detection import work
    if sys.platform.startswith("linux"):
        r = work.run([sys.executable, "-c", "x = bytearray(1 << 31)"], memory=1 << 30)
        assert r.returncode != 0


def test_the_symbols_of_the_package_hold_no_clock():
    import pathlib
    import re

    import symident
    # every search and count stops on work caps: no deadline, timeout or CPU clock decides a result
    src = pathlib.Path(symident.__file__).parent
    pat = re.compile(r"\b(deadline|timeout|cpuclock|process_time|RLIMIT_CPU)\b")
    hits = [f"{p.name}:{n}" for p in src.rglob("*.py") for n, line in enumerate(p.read_text().splitlines(), 1)
            if pat.search(line)]
    assert hits == []


def test_a_fiber_past_the_term_limit_is_not_counted(monkeypatch):
    from symident.detection import discrete
    monkeypatch.setattr(discrete, "_FIBER_TERMS", 3)
    info = si.detect(si.Model({"x": "-k^2*x - c*x^3"}, {"y": "x"}, trafo={"x": "x0"}), mode="global",
                     cores=1).info["discrete"]
    assert info["count"] is None
    assert any("have more than 3 terms, the most a count takes" in w for w in info["limits"])


def test_a_fiber_past_the_limit_of_unknowns_is_not_counted(monkeypatch):
    from symident.detection import discrete
    monkeypatch.setattr(discrete, "_FIBER_UNKNOWNS", 3, raising=False)
    info = si.detect(si.Model({"x": "-k^2*x - c*x^3"}, {"y": "x"}, trafo={"x": "x0"}), mode="global",
                     cores=1).info["discrete"]
    assert info["count"] is None
    assert any("unknowns, more than the 3 a count takes" in w for w in info["limits"])


def test_a_power_of_a_series_past_the_term_limit_stops_the_prolongation():
    from symident.detection import discrete
    R, a, b = spy.ring("a,b", spy.QQ)
    xs = [[a + b, a * b + 1, a + 1]]
    assert len(discrete._ser_poly({(3,): 1}, xs, [], 2, R(1), R(0))[2]) > 4
    with pytest.raises(discrete.FiberTooLargeError):
        discrete._ser_poly({(3,): 1}, xs, [], 2, R(1), R(0), limit=4)


def test_a_product_of_the_series_of_two_states_past_the_term_limit_stops_the_prolongation():
    from symident.detection import discrete
    R, a, b, c, d = spy.ring("a,b,c,d", spy.QQ)
    xs = [[a + b, a * b + 1, a + 1], [c + d, c * d + 1, c + 1]]
    assert all(sum(len(t) for t in x) <= 6 for x in xs)
    with pytest.raises(discrete.FiberTooLargeError):
        discrete._ser_poly({(1, 1): 1}, xs, [], 2, R(1), R(0), limit=6)


def test_a_sum_of_series_past_the_term_limit_stops_the_prolongation():
    from symident.detection import discrete
    R, a, b, c, d = spy.ring("a,b,c,d", spy.QQ)
    xs = [[a + b, a * b + 1, a + 1], [c + d, c * d + 1, c + 1]]
    with pytest.raises(discrete.FiberTooLargeError):
        discrete._ser_poly({(1, 0): 1, (0, 1): 1}, xs, [], 2, R(1), R(0), limit=6)


def test_the_term_limit_holds_for_the_coefficients_in_the_gap_variable():
    from symident.detection import discrete
    R, a, b, c, d = spy.ring("a,b,c,d", spy.QQ)
    E = discrete._Eps
    xs = [[E([a + b, a]), E([a * b + 1, b])], [E([c + d, c]), E([c * d + 1, d])]]
    o, z = E([R(1), R(0)]), E([R(0), R(0)])
    with pytest.raises(discrete.FiberTooLargeError):
        discrete._ser_poly({(1, 1): 1}, xs, [], 1, o, z, limit=6)


def test_a_product_within_the_term_limit_is_the_product_without_it():
    from symident.detection import discrete
    R, a, b = spy.ring("a,b", spy.QQ)
    xs = [[(a + b) ** 4, a]]
    # the bound of the terms of the product exceeds the limit, the product itself does not
    assert discrete._ser_poly({(2,): 1}, xs, [], 1, R(1), R(0), limit=20) == \
        discrete._ser_poly({(2,): 1}, xs, [], 1, R(1), R(0))


def test_resting_balances_past_the_bezout_cap_end_with_a_reason():
    m = si.Model({"x": "k1*(1 + u) - k2*x*y + k8*z - k3*x", "y": "k2*x*y - k4*y*z - k5*y + k9*x*z",
                  "z": "k4*y*z - k6*z - k9*x*z - k8*z", "u": "0"}, {"y1": "x", "y2": "y", "y3": "z"}, inputs=["u"],
                 equilibrate=True, events=[{"var": "u", "time": 0, "value": 1, "method": "replace"}])
    with pytest.warns(UserWarning, match="global analysis not done"):
        r = si.detect(m, mode="global", cores=1)
    assert "Bezout number 8 in the states is past the cap rest_bezout of 4" in r.info["discrete"]["unsupported"]


def test_resting_balances_past_the_block_cap_end_with_a_reason():
    from symident.detection import work
    tc = {"x1": "p - (k21 + k01)*x1 + k12*x2", "x2": "k21*x1 - k12*x2"}
    m = si.Model(tc, {"y": "x2"}, events=[{"var": "x1", "time": 1, "value": "D", "method": "add"}], fixed=["D"])
    with work.caps(rest_block=1), pytest.warns(UserWarning, match="global analysis not done"):
        r = si.detect(m, equilibrate=True, mode="global", cores=1)
    assert "largest coupled block of 2 states is past the cap rest_block of 1" in r.info["discrete"]["unsupported"]


def test_a_transcendental_balance_at_rest_is_named_without_a_search():
    from symident.detection import discrete
    with pytest.raises(ValueError, match="balances of x are not rational in the states"):
        discrete.resting_start({"x": "k - x*exp(x)", "u": "0"}, {}, ["u"], {"k"})


def test_a_closed_stage_of_the_fiber_is_solved_first_and_its_points_substituted():
    from symident.detection import discrete
    from symident.gfp import PRIMES
    p = PRIMES[0]
    fib = object.__new__(discrete.Fiber)
    fib.names, fib.allnames = ["x0", "x1", "x2"], ["x0", "x1", "x2"]
    # x0*x1 = 2, x0 + x1 = 3 close on (x0, x1) with two points in GF(p); x2^2 = x0 reads them
    polys = [{((0, 1), (1, 1)): 1, (): p - 2}, {((0, 1),): 1, ((1, 1),): 1, (): p - 3},
             {((2, 2),): 1, ((0, 1),): p - 1}]
    sat = [{((0, 1), (1, 1), (2, 1)): 1}]
    assert discrete._first_stage(polys, [0, 1, 2]) == {0, 1}
    staged = discrete._staged_count(fib, [0, 1, 2], polys, sat, p)
    b = fib._ms_block([0, 1, 2], polys, sat, p)
    assert staged == discrete._gb_count(b["names"], p, b["eqs"], b["sep"]) == 4
    # a block that no stage splits has none
    assert discrete._first_stage(polys[:2], [0, 1]) is None


def test_a_closed_stage_over_q_with_rational_points_gives_the_degree_of_the_whole_block():
    from symident.detection import discrete
    fib = object.__new__(discrete.Fiber)
    fib.names, fib.allnames, fib.sep_mode = ["x0", "x1", "x2"], ["x0", "x1", "x2"], {}
    F = Fraction
    polys = [{((0, 1), (1, 1)): F(1), (): F(-2)}, {((0, 1),): F(1), ((1, 1),): F(1), (): F(-3)},
             {((2, 2),): F(1), ((0, 1),): F(-1)}]
    sat = [{((0, 1), (1, 1), (2, 1)): F(1)}]
    pts = discrete._stage_points(fib, {0, 1}, polys[:2], [], 0)
    assert sorted((pt[0], pt[1]) for pt in pts) == [(1, 2), (2, 1)]
    whole = discrete._block_count(fib, [0, 1, 2], polys, sat, 0)
    assert discrete._staged_count(fib, [0, 1, 2], polys, sat, 0) == whole == 4


def test_a_system_past_the_caps_of_msolve_is_not_solved_and_the_cap_is_named():
    from symident.detection import discrete, work
    from symident.gfp import PRIMES
    p = PRIMES[0]
    fib = object.__new__(discrete.Fiber)
    fib.names, fib.allnames, fib.sep_mode = ["x0", "x1"], ["x0", "x1"], {}
    polys = [{((0, 1), (1, 1)): 1, (): p - 2}, {((0, 1),): 1, ((1, 1),): 1, (): p - 3}]
    discrete._FAILURES.clear()
    assert discrete._block_count(fib, [0, 1], polys, [], p) == 2
    with work.caps(msolve_terms=4):
        assert discrete._block_count(fib, [0, 1], polys, [], p) is None
        why = discrete.no_count_reason(3)
    assert "a system past the caps of msolve (400 unknowns, 4 terms)" in why
    assert work.msolve_too_large(["x0", "x1"], ["x0*x1-2", "x0+x1-3"]) is None
    with work.caps(msolve_unknowns=1):
        assert "past the cap of 1 unknowns of msolve" in work.msolve_too_large(["x0", "x1"], ["x0*x1-2"])


def test_msolve_ends_at_the_cap_on_the_entries_of_its_matrices():
    from symident.detection import discrete, work
    from symident.gfp import PRIMES
    p = PRIMES[0]
    fib = object.__new__(discrete.Fiber)
    fib.names, fib.allnames, fib.sep_mode = ["x0", "x1"], ["x0", "x1"], {}
    polys = [{((0, 1), (1, 1)): 1, (): p - 2}, {((0, 1),): 1, ((1, 1),): 1, (): p - 3}]
    b = fib._ms_block([0, 1], polys, [], p)
    discrete._FAILURES.clear()
    assert discrete._gb_count(b["names"], p, b["eqs"], b["sep"]) == 2
    with work.caps(msolve_entries=1):
        assert discrete._gb_count(b["names"], p, b["eqs"], b["sep"]) is None
        why = discrete.no_count_reason(3)
    assert "msolve reached the cap msolve_entries of 1 nonzero entries of the matrices of F4" in why
    with work.local_caps(msolve_entries=1):
        assert discrete._gb_count(b["names"], p, b["eqs"], b["sep"]) is None
    assert discrete._gb_count(b["names"], p, b["eqs"], b["sep"]) == 2


def test_the_memory_of_msolve_is_a_setting_of_the_global_analysis(monkeypatch):
    from symident.detection import work
    seen, orig = [], work._limiter

    def record(memory):
        if memory is not None:
            seen.append(memory)
        return orig(memory)
    monkeypatch.setattr(work, "_limiter", record)
    default = work.MSOLVE_MEMORY
    r = si.detect(si.Model({"x": "-k^2*x"}, {"y": "x"}, trafo={"x": "x0"}), mode="global", cores=1,
                  global_options={"memory": 3, "points": 1})
    assert r.info["discrete"]["count"] == 2
    assert seen and set(seen) == {3 << 30} and work.MSOLVE_MEMORY == default


def test_the_orders_of_the_count_rise_a_fixed_number_of_times(monkeypatch):
    import types

    from symident.detection import discrete
    given = []
    monkeypatch.setattr(discrete, "Fiber", lambda series, orders, positive, gap=0: types.SimpleNamespace(
        orders=orders, too_large=None))
    monkeypatch.setattr(discrete, "complex_count", lambda fib, point, p: given.append(fib.orders))
    series = types.SimpleNamespace(output_orders=lambda cap, p: [1, 3], coords=["a"], labels=["y", "z"])
    discrete.choose_order(series, True, 1, 20)
    # every observable starts one order above the highest order of the rank criterion, and the orders rise together
    assert given[:2] == [[4, 4], [5, 5]] and len(given) == discrete._ORDER_RAISES + 1


def test_a_count_past_the_caps_of_msolve_ends_the_rise_of_the_orders(monkeypatch):
    import types

    from symident.detection import discrete
    given = []

    def count(fib, point, p):
        given.append(fib.orders)
        discrete._failed(kind="size")
    monkeypatch.setattr(discrete, "Fiber", lambda series, orders, positive, gap=0: types.SimpleNamespace(
        orders=orders, too_large=None))
    monkeypatch.setattr(discrete, "complex_count", count)
    series = types.SimpleNamespace(output_orders=lambda cap, p: [1, 3], coords=["a"], labels=["y", "z"])
    assert discrete.choose_order(series, True, 1, 20)[2] is None and given == [[4, 4]]


def test_a_count_that_msolve_ends_out_of_memory_is_named_so(monkeypatch):
    from symident.detection import fieldproof
    monkeypatch.setattr(fieldproof, "exact_fiber", lambda *a, **k: {"exact": False, "why": "off", "proven": []})
    info = si.detect(si.Model({"x": "-k^2*x"}, {"y": "x"}, trafo={"x": "x0"}), mode="global", cores=1,
                     global_options={"memory": 0.05, "points": 1}).info["discrete"]
    assert info["count"] is None
    (lim,) = [w for w in info["limits"] if w.startswith("msolve found no finite count")]
    assert "memory" in lim and "positive dimension" not in lim


def _pinned_series(f, g, ic, coords, pins):
    from symident.detection import discrete
    series = discrete.Series([{"segments": [{"f": f, "g": g, "ic": ic, "t": "0", "label": ""}]}], coords)
    series.pins = discrete._pins(pins, series)
    return series


def _chain_series():
    from symident.detection import discrete
    return discrete.Series([{"segments": [{"f": {"x": "-k1*x", "y": "k1*x - k2*y"}, "g": {"o": "y"},
                                           "ic": {"x": "x0", "y": "0"}, "t": "0", "label": ""}]}], ["k1", "k2", "x0"])


def test_the_taylor_series_of_the_outputs_stops_at_the_cap_of_its_meter():
    from symident.detection import work
    from symident.gfp import PRIMES
    series = _chain_series()
    series.meter = work.Meter("the Taylor series of the outputs", 10, "term products")
    assert series.output_orders(40, PRIMES[0]) is None
    assert series.capped is not None and series.meter.over
    series = _chain_series()
    series.meter = work.Meter("the Taylor series of the outputs", 10 ** 9, "term products")
    assert series.output_orders(40, PRIMES[0]) is not None and 0 < series.meter.spent < 10 ** 9


def test_a_taylor_series_past_the_cap_is_named_in_the_limits():
    from symident.detection import work
    m = si.Model({"x": "-k1*x - k2*x^2"}, {"y": "x"}, trafo={"x": "x0"})
    with work.caps(series_products=10):
        info = si.detect(m, mode="global", cores=1).info["discrete"]
    (lim,) = [w for w in info["limits"] if w.startswith("the Taylor coefficients of the outputs up to order")]
    assert "the Taylor series of the outputs reached the cap of 10 term products" in lim


def test_the_candidates_stop_at_their_cap_with_the_cap_named():
    from symident.detection import work
    m = si.Model({"x1": "-k1*x1", "x2": "-k2*x2"}, {"y": "x1 + x2"}, trafo={"x1": "a1", "x2": "a2"})
    info = si.detect(m, mode="global", cores=1).info["discrete"]
    assert info["candidates"]["complete_search"] and info["candidates"]["found"] >= 1
    with work.caps(candidates=0):
        info = si.detect(m, mode="global", cores=1).info["discrete"]
    assert info["candidates"]["complete_search"] is False and info["candidates"]["found"] == 0
    assert "the candidates were not all tested: the search of candidates reached the cap of 0 permutations" in \
        info["limits"]


def test_the_reconstruction_of_the_maps_stops_at_its_cap_of_points():
    from symident.detection import discrete, work
    m = si.Model({"x": "-k^2*x"}, {"y": "x"}, trafo={"x": "x0"})
    r = si.detect(m, cores=1, reconstruct=True)
    parts, _ = r._fiber_parts(6, "discrete()")
    fib = discrete.Fiber(discrete.Series(parts, ["k", "x0"]), [3])
    k, maps = discrete.rational_maps(fib)
    assert k == 2 and len(maps) == 1
    meter = work.Meter("the reconstruction of the maps", 0, "points of the fiber")
    assert discrete.rational_maps(fib, meter) == (None, None)
    assert meter.reason() == "the reconstruction of the maps reached the cap of 0 points of the fiber"


def test_a_share_of_a_meter_ends_with_either_cap():
    from symident.detection import work
    m = work.Meter("the search", 10, "entries")
    s = m.share("the dense relations", 3)
    assert not s.spend(3) and s.spend(1) and not m.over and m.spent == 4
    assert s.reason() == "the dense relations reached the cap of 3 entries"
    t = m.share("the rest", 100)
    assert t.spend(7) and m.over and t.reason() == "the search reached the cap of 10 entries"


def test_the_rows_of_the_rank_take_the_orders_by_doubling():
    from symident.gfp import PRIMES
    series = _chain_series()
    rows = series.rank_rows(40, PRIMES[0])
    # the series stops at the first doubled order where the rows reach the rank, far below the cap
    assert rows is not None and series._jac[(PRIMES[0], 0)][0] < 40
    assert rows == _chain_series().rank_rows(max(k for _, k in rows), PRIMES[0])


def test_proven_functions_lower_the_orders_of_the_rank_criterion():
    from symident.detection import discrete
    series = _pinned_series({"x": "-k1*x - k2*x^2"}, {"y": "x"}, {"x": "x0"}, ["k1", "k2", "x0"], ["k1", "k2"])
    p = discrete.PRIMES[0]
    assert series.output_orders(6, p) == [2]
    # the gradients of the pins and of y(0) reach the rank of the three coordinates
    assert series.pinned_orders(6, p) == [0]
    # the pins of least degree that raise the rank and as many more come first, then all
    pins = ["k1^7", "k1^3 + k2^5", "k1^2*k2", "k1*k2", "k2", "k1"]
    series = _pinned_series({"x": "-k1*x - k2*x^2"}, {"y": "x"}, {"x": "x0"}, ["k1", "k2", "x0"], pins)
    first, every = discrete._pin_sets(series, p)
    assert sorted(pins[i] for i in first) == sorted(["k1", "k2", "k1*k2", "k1^2*k2"]) and every == list(range(6))


def test_the_count_of_the_proven_functions_verifies_where_the_search_stops(monkeypatch):
    from symident.detection import fieldproof
    monkeypatch.setattr(fieldproof, "exact_fiber", lambda *a, **k: {
        "exact": False, "why": "off", "generators": [], "seconds": 0.0, "proven": ["k1", "k2"]})
    r = si.detect(si.Model({"x": "-k1*x - k2*x^2"}, {"y": "x"}, trafo={"x": "x0"}), mode="global", cores=1)
    info = r.info["discrete"]
    cert = info["certificate"]
    assert info["count"] == 1 and info["count_final"] and cert["complete"] == "verified"
    pin = cert["pinned"]
    # one order above the criterion of the pins, below the order 2 of the outputs alone
    assert pin["functions"] == ["k1", "k2"] and pin["orders"] == {"y": 1} and pin["count"] == 1
    assert cert["count_bound"] == pin["count_bound"] and 0 < pin["count_bound"] < 1e-6
    assert "counted on 2 functions proven" in repr(r)


def test_a_count_of_the_proven_functions_past_twice_the_orbit_certifies_nothing():
    from symident.detection import discrete
    # the swap of k1 and k2 with x0 -> k1 x0 / k2 keeps the pins: two points where the orbit of the proven maps is one
    series = _pinned_series({"x": "-k1*x", "z": "k1*x - k2*z"}, {"y": "z"}, {"x": "x0", "z": "0"},
                            ["k1", "k2", "x0"], ["k1 + k2", "k1*k2", "k1*x0"])
    out = discrete.pinned_count(series, True, 1)
    assert out["count"] == 2 and out["cert"] is None and "not below 2" in out["why"]


def _toy_projection(series, syms):
    """The supports of the projections of the series, as fieldproof.exact_fiber() reports those past ioproj.TERMS."""
    import random

    from symident.detection import fieldproof, ioproj
    from symident.gfp import all_primes
    rel = fieldproof._relations(series, syms)
    rng = random.Random(3)
    p = all_primes()[0]
    pos = [rel.names.index(str(s)) for s in syms if s in set(series.params)]
    h = fieldproof._profiles(rel, rng)[0]
    pr = ioproj.Projection.output(rel, h, 0, pos, rng)
    S = pr.search([rng.randrange(1, p) for _ in pos], p, math.inf)
    return [{"profile": list(h), "observable": 0, "support": [tuple(m) for m in S]}]


def test_random_primes_for_the_trials_of_a_verified_relation():
    import random

    from symident.detection import fewcoef
    rng = random.Random(1)
    assert all(fewcoef._is_prime(n) == spy.isprime(n) for n in range(2, 3000))
    q = fewcoef.random_prime(rng, 1 << 30)
    assert spy.isprime(q) and 1 << 29 <= q <= 1 << 30
    # Rosser and Schoenfeld: more than 3.8e7 primes from 2^29 to the largest engine prime
    assert fewcoef.primes_at_least(1 << 29, 1518500183) > 3.8e7


def test_coefficients_of_a_projection_over_a_monomial_one_are_verified():
    import random

    from symident.detection import fewcoef
    series = _pinned_series({"x": "-k1*x - k2*x^2"}, {"y": "x"}, {"x": "x0"}, ["k1", "k2", "x0"], [])
    syms = list(series.cs)
    log = []
    out = fewcoef.verified_functions(series, syms, _toy_projection(series, syms), math.inf, random.Random(2), log)
    # y' + k1 y + k2 y^2 = 0 over the coefficient 1 of y'
    assert sorted(out["functions"]) == ["k1", "k2"] and out["why"] is None
    assert 0 < out["exist"] < 1e-30 and all(0 < b < 1e-30 for b in out["checks"].values())


def test_a_verified_relation_enters_the_count_and_its_bound(monkeypatch):
    from symident.detection import fieldproof

    def search(series, orders, grp, syms, *a, **k):
        return {"exact": False, "why": "off", "generators": [], "seconds": 0.0, "proven": [],
                "large": _toy_projection(series, syms)}
    monkeypatch.setattr(fieldproof, "exact_fiber", search)
    r = si.detect(si.Model({"x": "-k1*x - k2*x^2"}, {"y": "x"}, trafo={"x": "x0"}), mode="global", cores=1)
    cert = r.info["discrete"]["certificate"]
    pin = cert["pinned"]
    assert cert["complete"] == "verified" and sorted(pin["functions"]) == ["k1", "k2"]
    assert sorted(pin["relation"]["functions"]) == ["k1", "k2"] and pin["relation_bound"] > 0
    assert cert["count_bound"] == pin["count_bound"] == pin["count_bound_q"] + pin["relation_bound"]
    assert "verified relation" in cert["bounded"]


def test_a_state_read_only_through_its_logarithm_is_a_log_coordinate():
    info, _ = _relaxed({"x1": "x1*(a1 - b1*log(x1) - c*log(x2))", "x2": "x2*(a2 - b2*log(x2))"},
                       {"y1": "x1", "y2": "x2^2"})
    assert info["relaxed"]["dropped"] == {"x1": "si_P1", "x2": "si_P2"}
    assert info["relaxed"]["model"]["observables"] == {"y1": "si_L1", "y2": "2*si_L2"}
    assert info["count"] == 1 and info["certificate"]["complete"] == "exact"


def test_a_state_set_to_numbers_by_events_is_a_number_per_segment():
    m = si.Model({"x": "a*exp(v/k)*(1 - x) - b*x", "v": "0"}, {"y": "x"}, inputs=["v"], equilibrate=True,
                 events=[{"var": "v", "time": 0, "value": 10, "method": "replace"}], positive=["a", "b", "k", "x"])
    info = si.detect(m, mode="global", cores=1).info["discrete"]
    kinds = {rw["rewrite"]: rw for rw in info["rewrites"]}
    assert kinds["constant"]["state"] == "v" and kinds["coordinate"]["from"] == "exp(10/k)"
    assert info["count"] == 1 and info["certificate"]["complete"] == "exact"


def test_the_resting_chart_starts_at_rest():
    from symident.detection.restchart import rest_chart
    c = rest_chart({"V": "I - g*(V - E) - c*w*(V - F)", "w": "a*exp(V/k)*(1 - w) - b*w", "I": "0"}, {"y": "V"},
                   ["I"], ["g", "E", "c", "F", "a", "k", "b"], ["a", "b", "k", "w", "g", "c"])
    # E, signed, is exchanged for the resting value of V; the start solves every balance
    assert c["exchanged"].keys() == {"E"} and c["rest"] == {"V": "si_V_rest1"}
    from symident.sym.parse import model_sympify
    at = {spy.Symbol(x): model_sympify(v) for x, v in c["trafo"].items()}
    for e in c["odes"].values():
        assert spy.simplify(model_sympify(e).xreplace(at).xreplace({spy.Symbol("I"): 0})) == 0
    assert "E" not in c["positive"] and "si_V_rest1" in c["coordinates"]


def test_a_resting_state_inside_an_exponential_is_counted_in_the_resting_chart():
    m = si.Model({"V": "I - g*(V - E) - c*w*V", "w": "a*exp(V/k)*(1 - w) - b*w", "I": "0"}, {"y": "V"},
                 inputs=["I"], equilibrate=True, events=[{"var": "I", "time": 0, "value": 1, "method": "replace"}],
                 positive=["a", "b", "k", "w", "g", "c"])
    info = si.detect(m, mode="global", cores=1).info["discrete"]
    assert info["relaxed"]["rest_chart"]["rest"] == {"V": "si_V_rest1"}
    assert info["count"] == 1 and info["certificate"]["complete"] in ("exact", "verified")


def test_the_term_bound_holds_the_terms_of_the_expansion():
    from symident.detection.discrete import _term_bound
    a, b, c = spy.symbols("a b c")
    for e in [(a + b) * (a + c), (a + b + c) ** 3, (a + b) / (a + c) ** 2 + c]:
        assert _term_bound(e) >= len(spy.Add.make_args(spy.expand(spy.fraction(spy.together(e))[0])))


def test_a_map_of_the_states_past_the_cap_on_terms_is_not_solved_and_the_cap_is_named(monkeypatch):
    from symident.detection import discrete, work
    monkeypatch.setattr(discrete, "_block_map", lambda *a: None)
    f, g = {"x": "-a*x", "y": "-b*y"}, {"o": "x + y"}
    a, b = spy.symbols("a b")
    sig = {a: b, b: a}
    assert discrete._state_map(f, g, None, sig, ["x", "y"]) is not None
    discrete._CAPPED.clear()
    with work.caps(state_map_terms=1):
        assert discrete._state_map(f, g, None, sig, ["x", "y"]) is None
    assert any("past the cap state_map_terms of 1" in n for n in discrete._CAPPED)


def test_a_result_of_reduce_cq_is_counted_on_the_level_sets_of_the_totals():
    f = {"A": "-k1^2*A + k2*B", "B": "k1^2*A - k2*B"}
    r = si.detect(si.Model(f, {"y1": "A", "y2": "B"}), reduce_cq=True)
    out = r.discrete()
    (s,) = out["symmetries"]
    assert r.info["coordinates"] == ["B", "k1", "k2", "total1"] and out["info"]["eliminated"] == {"A": "(-B + total1)"}
    assert s["level"] == "exact" and s["map"] == {"k1": "-k1"} and out["info"]["count"] == 2
    assert out["info"]["certificate"]["complete"] == "exact" and out["info"]["count_domain"] == 1
    # at rest, with the total set by the trafo
    f = {"A": "-k1^2*u*A + k2*B", "B": "k1^2*u*A - k2*B", "u": "0"}
    m = si.Model(f, {"y1": "A", "y2": "B"}, inputs=["u"], equilibrate=True, reduce_cq=True, trafo={"total1": "A0"},
                 events=[{"var": "u", "time": 0, "value": 1, "method": "replace"}])
    r = si.detect(m, mode="global")
    i = r.info["discrete"]
    assert r.info["coordinates"] == ["A0", "k1", "k2"] and [s["map"] for s in r.discrete_symmetries] == [{"k1": "-k1"}]
    assert i["count"] == 2 and i["count_domain"] == 1 and i["certificate"]["complete"] == "exact"

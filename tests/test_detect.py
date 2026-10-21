"""Behaviour of detect(): ranks, verdicts and directions on small models."""

import re
import warnings
from fractions import Fraction

import numpy as np
import pytest
import sympy as spy

import symident as si
from symident import state
from symident.detection.cq import equil_zero_states
from symident.sym.msolve import MsolveMissingError, _msolve_binary
from symident.sym.parse import model_sympify


def _has_msolve():
    try:
        _msolve_binary()
    except Exception:
        return False
    return True


needs_msolve = pytest.mark.skipif(not _has_msolve(), reason="needs msolve")

_MODEL_KEYS = ("reactions", "trafo", "conditions", "events", "inputs", "fixed", "parameters", "known_inputs")


def detect(odes=None, observables=None, **kw):
    """Build a Model from the model keywords and run si.detect with the rest."""
    model = si.Model(odes, observables, **{k: kw.pop(k) for k in _MODEL_KEYS if k in kw})
    kw.setdefault("cores", 1)
    return si.detect(model, **kw)


def network(*reactions):
    """Reaction network from (educts, products, rate) triples; "" or "0" is no species."""
    def side(s):
        return [x.strip() for x in s.split("+") if x.strip() not in ("", "0")]
    species = []
    for ed, pr, _ in reactions:
        for s in side(ed) + side(pr):
            if s not in species:
                species.append(s)
    S = []
    for ed, pr, _ in reactions:
        row = [0] * len(species)
        for s in side(ed):
            row[species.index(s)] -= 1
        for s in side(pr):
            row[species.index(s)] += 1
        S.append(row)
    return {"species": species, "stoichiometry": S, "rates": [r for _, _, r in reactions]}


def event(var, time, value, method="replace"):
    return {"var": var, "time": time, "value": value, "method": method}


def grid(rows, **cols):
    return {"rows": list(rows), "cols": {k: list(v) for k, v in cols.items()}}


def canonical():
    return network(("A", "B", "k1 * A"), ("B", "A", "k2 * B"))


def sym(x):
    """Expression string as sympy; every name not called as a function is a symbol."""
    x = str(x).replace("^", "**")
    names = set(re.findall(r"\b([A-Za-z_]\w*)\b(?!\s*\()", x))
    return spy.sympify(x, locals={n: spy.Symbol(n) for n in names})


def expr_equal(a, b):
    """Exact symbolic equality of two expression strings."""
    return spy.simplify(sym(a) - sym(b)) == 0


def tangent(d, pt, coords):
    """The generator components of a direction evaluated at the point `pt`."""
    v = dict.fromkeys(coords, 0.0)
    subs = {spy.Symbol(k): x for k, x in pt.items()}
    for k, e in d["generator"].items():
        v[k] = float(sym(e).subs(subs))
    return v


def supports(res):
    return {s for d in res.symmetries for s in d["support"]}


def dirset(res):
    return sorted("+".join(sorted(d["support"])) for d in res.symmetries)


def generator_strings(res):
    return [str(e) for d in res.symmetries for e in (d["generator"] or {}).values()]


def chain():
    return network(("", "A", "kin"), ("A", "B", "k1 * A"), ("B", "C", "k2 * B"),
                   ("C", "D", "k3 * C"), ("D", "", "k4 * D"))


def hill_michaelis():
    return network(("0", "FB", "k_pr_FB"), ("FB", "0", "d_FB * FB"),
                   ("0", "x", "k_pr_x * K^n / (K^n + FB^n)"), ("x", "0", "d_x * x"))


def pool_model():
    """Two messenger pools gating one channel with a shared affinity K."""
    f = {"P1": "k1*(T1 - P1) - (l1 + c1*u)*P1", "P2": "k2*(T2 - P2) - (l2 + c2*u)*P2", "u": "0"}
    g = {"y1": "P1^2/(K^2 + P1^2)", "y2": "P2^2/(K^2 + P2^2)"}
    return f, g, grid(["ctrl", "stim"], u=[0, 1])


def test_trafo_renames_rate():
    res = detect(reactions=canonical(), observables={"Aobs": "alpha * A"}, trafo={"k1": "kf"})
    assert res.identifiable is False
    supp = supports(res)
    assert {"A", "B", "alpha"} <= supp
    assert "k1" not in supp


def test_canonical_non_identifiability():
    res = detect(reactions=canonical(), observables={"Aobs": "alpha * A"})
    assert res.identifiable is False
    assert res.rank < res.dim
    assert any({"A", "B", "alpha"} <= set(d["support"]) for d in res.symmetries)


def test_fully_observed_identifiable():
    res = detect(reactions=canonical(), observables={"o1": "A", "o2": "B"})
    assert res.identifiable is True
    assert res.rank == res.dim


def test_scalings_only_canonical():
    res = detect(reactions=canonical(), observables={"Aobs": "alpha * A"}, scalings_only=True)
    assert len(res.symmetries) == 1
    v = res.symmetries[0]["weights"]
    assert {"A", "B", "alpha"} <= set(v)
    assert int(v["A"]) == int(v["B"])
    assert int(v["alpha"]) == -int(v["A"])


def test_deep_chain():
    res = detect(reactions=chain(), observables={"yD": "scale * D"})
    assert res.identifiable is False
    assert res.rank < res.dim
    assert len(res.symmetries) >= 1


def test_non_rational_observable_rejected():
    # sqrt(A) is rational only in the log chart of a positive A
    with pytest.raises(Exception, match="rational"):
        detect(reactions=canonical(), observables={"Aobs": "sqrt(A)"}, positive=False)


def test_reconstruct_exact_scaling():
    res = detect(reactions=canonical(), observables={"Aobs": "alpha * A"}, reconstruct=True)
    assert res.identifiable is False
    assert res.info["engine"] == "modular"
    assert all(isinstance(d["explicit"], bool) and len(d["support"]) > 0 for d in res.symmetries)
    scal = [d for d in res.symmetries if {"A", "B", "alpha"} <= set(d["generator"] or {})]
    assert len(scal) >= 1
    d = scal[0]
    assert d["type"] == "scaling"
    w = {k: int(v) for k, v in d["weights"].items()}
    assert w["A"] == w["B"]
    assert w["alpha"] == -w["A"]


def test_closed_form_and_support_verdicts_agree():
    ana = detect(reactions=canonical(), observables={"Aobs": "alpha * A"}, reconstruct=True)
    sup = detect(reactions=canonical(), observables={"Aobs": "alpha * A"}, reconstruct=False)
    assert ana.rank == sup.rank
    assert ana.dim == sup.dim
    # without reconstruct only the scalings come in closed form
    general = [d for d in sup.symmetries if d["type"] != "scaling"]
    assert not any(d["explicit"] for d in general)


def test_reconstruct_deep_chain():
    res = detect(reactions=chain(), observables={"yD": "scale * D"}, reconstruct=True)
    assert res.identifiable is False
    assert len(res.symmetries) >= 1
    assert any(d["explicit"] for d in res.symmetries)


def test_fixed_parameter_removes_symmetry():
    free = detect(reactions=canonical(), observables={"Aobs": "alpha * A"}, reconstruct=True)
    fixed = detect(reactions=canonical(), observables={"Aobs": "alpha * A"}, reconstruct=True,
                   fixed=["alpha"])
    assert fixed.dim == free.dim - 1
    assert len(fixed.symmetries) < len(free.symmetries)
    assert "alpha" not in supports(fixed)


def test_steady_state_initial_condition():
    # started at rest x* = b/a without a dose, only s*b/a is observed
    res = detect({"x": "b - a*x"}, {"y": "s*x"}, trafo={"x": "b/a"}, reconstruct=True)
    assert res.identifiable is False
    assert res.rank == 1
    assert res.dim == 3


def test_pre_equilibrated_known_dose():
    def ident(method):
        return detect({"x": "b - a*x"}, {"y": "s*x"}, trafo={"x": "b/a"},
                      events=[event("x", 0, "dose", method)],
                      conditions=grid(["stim"], dose=[2])).identifiable
    # an absolute dose fixes the scale, a relative one does not
    assert ident("replace") is True
    assert ident("add") is True
    assert ident("multiply") is False


def test_equilibrate_matches_explicit_steady_state():
    res = detect({"x": "b - a*x"}, {"y": "s*x"}, equilibrate=True, reconstruct=True)
    assert res.identifiable is False
    assert res.rank == 1
    assert res.dim == 3
    # every direction is a null direction of grad(s*b/a) in (a, b, s)
    vals = {"a": 2, "b": 3, "s": 5}
    grad = {"a": -vals["s"] * vals["b"] / vals["a"] ** 2, "b": vals["s"] / vals["a"],
            "s": vals["b"] / vals["a"]}
    for d in res.symmetries:
        v = tangent(d, vals, ["a", "b", "s"])
        assert sum(grad[k] * v[k] for k in grad) == pytest.approx(0, abs=1e-9)


def test_equilibrate_irrational_steady_state():
    # x* = sqrt(b/a) has no rational form
    res = detect({"x": "b - a*x^2"}, {"y": "s*x"}, equilibrate=True, reconstruct=True)
    assert res.identifiable is False
    assert res.rank == 1
    assert res.dim == 3


def test_equilibrate_ignores_state_trafo():
    with pytest.warns(UserWarning, match="ignored"):
        res = detect({"x": "b - a*x"}, {"y": "s*x"}, equilibrate=True, trafo={"x": "x0"})
    assert res.identifiable is False
    assert res.rank == 1
    assert res.dim == 3
    # a parameter-only trafo is a substitution
    res = detect({"x": "b - a*x"}, {"y": "s*x"}, equilibrate=True, trafo={"a": "k1*k2"})
    assert res.identifiable is False
    assert res.rank == 1
    assert res.dim == 4


def test_equilibrate_t0_dose():
    res = detect({"x": "b - a*x"}, {"y": "s*x"}, equilibrate=True,
                 events=[event("x", 0, "dose", "replace")], conditions=grid(["stim"], dose=[2]))
    assert res.identifiable is True
    assert res.rank == 3
    res0 = detect({"x": "b - a*x"}, {"y": "s*x"}, equilibrate=True)
    assert res0.identifiable is False
    assert res0.rank == 1


def test_only_drained_species_rest_at_zero():
    # a conserved pair beside a degraded species keeps its total at rest
    rx = network(("A", "B", "k1 * A"), ("B", "A", "k2 * B"), ("C", "", "kc * C"))
    assert equil_zero_states(rx["stoichiometry"], rx["species"], rx["rates"], []) == ["C"]
    res = detect(reactions=rx, observables={"Aobs": "alpha * A", "Cobs": "s * C"}, equilibrate=True)
    assert "A" in res.info["coordinates"] and ["alpha"] not in [d["support"] for d in res.symmetries]
    # a pair that drains through one of its members empties as a whole
    rx = network(("A", "B", "k1 * A"), ("B", "A", "k2 * B"), ("B", "", "k3 * B"))
    assert set(equil_zero_states(rx["stoichiometry"], rx["species"], rx["rates"], [])) == {"A", "B"}


def test_condition_zero_state_held_at_zero():
    # the equilibrate verdict matches the explicit steady state given per condition
    rx = network(("", "A", "k1"), ("A", "", "d1*A"), ("", "B", "k2*A"), ("B", "", "d2*B"),
                 ("", "X", "kp*plasmid"), ("X", "", "dx*X"), ("", "A", "kx*X"))
    g = {"yA": "s*(A + X)", "yB": "B"}
    ev = [event("A", 0, "dose", "add")]
    eq = detect(reactions=rx, observables=g, conditions=grid(["ctrl", "oe"], plasmid=[0, 1], dose=[1, 1]),
                events=ev, equilibrate=True, reduce_cq=True)

    def ss(pl):
        return {"X": f"kp*{pl}/dx", "A": f"(k1 + kx*kp*{pl}/dx)/d1",
                "B": f"k2*(k1 + kx*kp*{pl}/dx)/(d1*d2)", "plasmid": str(pl)}
    ex = detect(reactions=rx, observables=g, trafo=[ss(0), ss(1)], events=ev,
                conditions=grid(["ctrl", "oe"], dose=[1, 1]))
    assert eq.rank == ex.rank
    assert eq.identifiable == ex.identifiable
    assert dirset(eq) == dirset(ex)


def test_kernel_thread_count_invariance():
    f = {"A": "-(k1 + u*k2)*A + b", "u": "0"}
    ev = [event("A", 0, "dose", "add"), event("u", -1, "var_u", "replace")]
    cg = grid(["c1", "c2", "c3", "c4"], var_u=[0, 1, 0, 1], dose=[1, 1, 2, 2])

    def run(cores):
        return detect(f, {"y": "s*A"}, events=ev, conditions=cg, inputs=["u"], reconstruct=True,
                      cores=cores)
    serial, parallel = run(1), run(2)
    assert serial.rank == parallel.rank
    assert serial.identifiable == parallel.identifiable
    assert len(serial.symmetries) == len(parallel.symmetries)


def test_michaelis_menten_enzyme():
    # the directions annihilate the observable invariants s*S, s*Km, s*kcat*Etot
    res = detect({"S": "-kcat*Etot*S/(Km + S)"}, {"y": "s*S"}, reconstruct=True)
    assert res.identifiable is False
    assert res.rank == 3
    assert res.dim == 5
    assert len(res.symmetries) == 2
    assert all(d["explicit"] for d in res.symmetries)
    pt = {"S": 2, "Etot": 3, "Km": 5, "s": 7, "kcat": 11}
    for d in res.symmetries:
        v = tangent(d, pt, list(pt))
        dinv = [pt["s"] * v["S"] + pt["S"] * v["s"],
                pt["s"] * v["Km"] + pt["Km"] * v["s"],
                pt["kcat"] * pt["Etot"] * v["s"] + pt["s"] * pt["Etot"] * v["kcat"]
                + pt["s"] * pt["kcat"] * v["Etot"]]
        assert dinv == pytest.approx([0, 0, 0], abs=1e-9)


def test_transcription_translation_curve():
    res = detect({"m": "ktx - dm*m", "p": "ktl*m - dp*p"}, {"y": "p"}, reconstruct=True)
    assert res.identifiable is False
    assert res.rank == 5
    assert res.dim == 6
    assert len(res.symmetries) == 1
    d = res.symmetries[0]
    assert d["explicit"] is True
    assert set(d["generator"]) == {"m", "ktx", "ktl"}
    pt = {"m": 2, "p": 3, "ktx": 5, "ktl": 7, "dm": 11, "dp": 13}
    v = tangent(d, pt, list(pt))
    # d(ktx*ktl) and d(ktl*m) vanish along the direction
    assert pt["ktl"] * v["ktx"] + pt["ktx"] * v["ktl"] == pytest.approx(0, abs=1e-9)
    assert pt["m"] * v["ktl"] + pt["ktl"] * v["m"] == pytest.approx(0, abs=1e-9)


def test_scalings_peeled_past_interpolation_cap():
    f = {"x": "-k * x"}
    g = {"y": "s*a*b*c*d*e*x"}
    res = detect(f, g, reconstruct=True)
    assert res.identifiable is False
    assert res.dim == 8
    assert res.rank == 2
    assert len(res.symmetries) == 6
    assert all(d["type"] == "scaling" and d["explicit"] for d in res.symmetries)
    # each scaling leaves the product s*a*b*c*d*e*x invariant
    factors = ["s", "a", "b", "c", "d", "e", "x"]
    for d in res.symmetries:
        assert abs(sum(float(sym(d["weights"].get(k, 0))) for k in factors)) < 1e-9
    sup = detect(f, g)
    assert all(d["type"] == "scaling" and d["generator"] is not None for d in sup.symmetries)
    assert len(sup.symmetries) == 6


def test_two_moieties_scaling_and_general():
    rx = network(("A1", "B1", "k1 * A1"), ("B1", "A1", "k2 * B1"),
                 ("A2", "B2", "k3 * A2"), ("B2", "A2", "k4 * B2"))
    res = detect(reactions=rx, observables={"y1": "s * A1", "y2": "s * A2"}, reconstruct=True)
    assert res.rank == 6
    assert res.dim == 9
    assert len(res.symmetries) == 3
    types = [d["type"] for d in res.symmetries]
    assert types.count("scaling") == 1
    assert types.count("general") == 2
    assert all(d["explicit"] for d in res.symmetries)
    # each conserved direction shifts B_i along the total A_i + B_i
    for d in res.symmetries:
        if d["type"] != "general":
            continue
        b = [k for k in d["generator"] if k in ("B1", "B2")]
        assert b
        assert expr_equal(d["generator"][b[0]], "A1 + B1" if "B1" in d["generator"] else "A2 + B2")


def test_equilibrate_free_hill_exponent():
    ev = [event("u", -1, "1", "replace")]
    cond = grid(["stim"], kin=["kin"])
    r = detect({"p": "kpr/(1+kinh*p^nhill) - dp*p + kin*u", "u": "0"}, {"y": "s*p"},
               equilibrate=True, inputs=["u"], events=ev, conditions=cond, reconstruct=False)
    assert "nhill" not in supports(r)
    assert r.dim == 6
    # a fractional-root steady state: a scaling with weight 1 - q on dp
    r2 = detect({"x": "kpr - dp*x^q + kin*u", "u": "0"}, {"y": "s*x"},
                equilibrate=True, inputs=["u"], events=ev, conditions=cond, reconstruct=True)
    assert len(r2.symmetries) == 1
    d2 = r2.symmetries[0]
    assert d2["explicit"] is True
    assert d2["type"] == "scaling"
    assert "q" not in d2["support"]
    assert expr_equal(d2["weights"]["dp"], "1 - q")
    assert expr_equal(d2["weights"]["kpr"], "1")
    assert expr_equal(d2["weights"]["s"], "-1")
    w2 = [str(x) for x in d2["weights"].values()]
    assert not any("log(" in x for x in w2)
    assert not any("_E_" in x or "_L_" in x for x in w2)
    # without equilibrate the transient recast gives a scaling weighted by nhill
    r3 = detect({"p": "kpr/(1+kinh*p^nhill) - dp*p"}, {"y": "s*p"}, reconstruct=True)
    assert r3.dim == 6
    assert r3.rank == 5
    assert len(r3.symmetries) == 1
    d3 = r3.symmetries[0]
    assert d3["explicit"] is True
    assert "nhill" not in d3["support"]
    assert set(d3["support"]) == {"kinh", "kpr", "p", "s"}
    gen3 = [str(x) for x in d3["generator"].values()]
    assert any("nhill" in x for x in gen3)
    assert not any("_E_" in x or "_L_" in x for x in gen3)
    assert not any("log(" in x for x in gen3)


def test_equilibrate_hill_closed_form_recast():
    ev = [event("u", -1, "var_u", "replace")]
    r = detect({"p": "kpr/(1+kinh*p^nhill) - dp*p + kin*u", "u": "0"}, {"y": "s*p"},
               equilibrate=True, events=ev, conditions=grid(["ctrl", "stim"], var_u=[0, 1]),
               reconstruct=True)
    assert all(d["explicit"] for d in r.symmetries)
    exprs = generator_strings(r)
    assert any("p^nhill" in x for x in exprs)
    assert any("log(p)" in x for x in exprs)
    assert not any("_E_" in x or "_L_" in x for x in exprs)
    # generators use the ^ power syntax and are plain name-to-string maps
    assert not any("**" in x for x in exprs)
    assert all(isinstance(d["generator"], dict) and all(isinstance(v, str) for v in d["generator"].values())
               for d in r.symmetries)


def test_michaelis_hill_exponent_non_identifiable():
    # K under the exponent is a coordinate and non-identifiable, so dim is 7
    r = detect(reactions=hill_michaelis(), observables={"xobs": "scale * x"}, equilibrate=True)
    assert r.identifiable is False
    assert r.dim == 7
    supp = supports(r)
    assert "K" in supp
    assert "n" in supp


def test_michaelis_hill_reconstruct_log_factor():
    r = detect(reactions=hill_michaelis(), observables={"xobs": "scale * x"}, equilibrate=True,
               reconstruct=True)
    assert all(d["explicit"] for d in r.symmetries)
    assert any("log(" in x for x in generator_strings(r))


def test_equilibrate_held_variable_moiety():
    # reduce_cq=False holds the pivot's resting value under the pivot's name
    rx = network(("A", "B", "k1 * A * (1 + kin*u)"), ("B", "A", "k2 * B"), ("0", "u", "0"))
    obs = {"y": "scale * A"}
    ev = [event("u", 0, "dose", "replace")]
    args = dict(reactions=rx, observables=obs, equilibrate=True, inputs=["u"], events=ev,
                conditions=grid(["d1", "d2"], dose=[1, 2]), reconstruct=True)
    r_t = detect(**args, reduce_cq=True)
    r_f = detect(**args, reduce_cq=False)
    assert r_f.rank == r_t.rank
    assert r_f.dim == r_t.dim
    assert len(r_f.symmetries) == len(r_t.symmetries)
    supp_f, supp_t = supports(r_f), supports(r_t)
    assert "A" in supp_f
    assert not any(s.startswith("total") for s in supp_f)
    assert any(s.startswith("total") for s in supp_t)
    assert "A" not in supp_t
    # the pool scaling lands on the pivot's initial value with weight +1
    assert all(d["explicit"] for d in r_f.symmetries)
    pool = [d for d in r_f.symmetries if "A" in d["support"]]
    assert len(pool) >= 1
    assert expr_equal(pool[0]["generator"]["A"], "A")
    # a single condition with a rank-deficient f = 0
    r1 = detect(reactions=rx, observables=obs, equilibrate=True, inputs=["u"], events=ev,
                conditions=grid(["d1"], dose=[1]), reconstruct=True)
    assert "A" in supports(r1)


def test_sparse_reconstruction_matches_dense():
    res = detect(reactions=canonical(), observables={"Aobs": "alpha * A"}, reconstruct=True,
                 control=si.reconst_control(relevance_cap=0))
    d = [x for x in res.symmetries
         if x["explicit"] and {"B", "k1", "k2"} <= set(x["generator"] or {})]
    assert len(d) >= 1
    assert expr_equal(d[0]["generator"]["B"], "A + B")
    assert expr_equal(d[0]["generator"]["k1"], "k2")


def test_multi_condition_switch_gated_rate():
    multi = detect({"A": "-(k1 + u*k2)*A", "u": "0"}, {"y": "A"},
                   events=[event("u", -1, "var_u", "replace")],
                   conditions=grid(["ctrl", "stim"], var_u=[0, 1]))
    assert multi.identifiable is True
    assert multi.info["conditions"] == 2
    assert not any("u" in d["support"] for d in multi.symmetries)


def test_multi_condition_union_parameter_space():
    res = detect({"A": "-k*A"}, {"y": "A"}, conditions=grid(["ctrl", "knd"], k=["k", "k_knd"]))
    assert res.dim == 3
    assert res.identifiable is True


def test_multi_condition_common_confounder():
    res = detect({"A": "-k*A", "u": "0"}, {"y": "s*A"}, events=[event("u", -1, "var_u", "replace")],
                 conditions=grid(["a", "b"], var_u=[0, 1]))
    assert res.identifiable is False
    supp = supports(res)
    assert {"A", "s"} <= supp
    assert "u" not in supp


def test_gap_dose_resolved_in_own_condition():
    # the same model with the grid column named apart from or after its first cell
    def run(colname):
        ev = [event("A", 0, "0", "replace"), event("A", 1, colname, "add")]
        return detect({"A": "-k*A"}, {"y": "A"}, events=ev,
                      conditions={"rows": ["c1", "c2"], "cols": {colname: ["d", "d2"]}})
    neutral = run("dose")
    colliding = run("d")
    assert colliding.dim == neutral.dim
    assert colliding.rank == neutral.rank
    assert colliding.identifiable == neutral.identifiable
    assert neutral.identifiable is True
    assert neutral.dim == 3


def test_per_condition_observables():
    f = {"A": "-k*A", "B": "-q*B"}
    cg = grid(["elisa", "wb"], q=["q1", "q2"])
    pooled = detect(f, {"yA": "A", "yB": "B"}, conditions=cg)
    assert pooled.identifiable is True
    assert len(pooled.symmetries) == 0
    split = detect(f, [{"yA": "A"}, {"yB": "B"}], conditions=cg)
    assert split.identifiable is False
    assert split.info["conditions"] == 2
    assert split.dim == pooled.dim
    assert split.rank < pooled.rank
    assert len(split.symmetries) == 1
    assert split.symmetries[0]["support"] == ["q1"]


def test_scaling_lattice_intersection_per_condition():
    f = {"A": "-k*A", "B": "-k*B"}
    cg = grid(["relative", "calibrated"], s=["s", "1"])
    pooled = detect(f, {"yA": "s*A", "yB": "s*B"}, scalings_only=True, conditions=cg)
    assert len(pooled.symmetries) == 0
    split = detect(f, [{"yA": "s*A"}, {"yB": "s*B"}], scalings_only=True, conditions=cg)
    assert len(split.symmetries) == 1
    assert split.symmetries[0]["support"] == ["A", "s"]
    obs = detect(f, [{"yA": "s*A"}, {"yB": "s*B"}], conditions=cg)
    assert len(obs.symmetries) == 1
    assert obs.symmetries[0]["support"] == ["A", "s"]


def test_post_t0_event_second_segment():
    ev = [event("A", 0, "d0", "add"), event("A", 1, "d1", "add")]
    r = detect({"A": "-k*A"}, {"y": "s*A"}, events=ev, conditions=grid(["c1"], d0=[1], d1=[1]),
               reconstruct=True)
    assert r.info["conditions"] == 1
    assert r.info["segments"] == 2
    assert r.dim == 3
    assert r.identifiable is True


def test_direction_depending_on_gap():
    # only z(1) = z*exp(-k2) reaches the output, so z and k2 move in a gap-dependent ratio
    ev = [event("u", 0, "0", "replace"), event("u", 1, "1", "replace")]
    r = detect({"x": "-k1*x + u*z", "z": "-(1-u)*k2*z", "u": "0"}, {"y": "x"}, events=ev,
               reconstruct=True)
    assert r.identifiable is False
    assert r.rank == 3
    assert len(r.symmetries) == 1
    gen = r.symmetries[0]["generator"]
    assert set(gen) == {"z", "k2"}
    assert expr_equal(gen["z"], f"z*({gen['k2']})")
    r2 = detect({"x": "-k1*x + u*k2*z", "z": "-(1-u)*z", "u": "0"}, {"y": "x"}, events=ev)
    assert r2.rank == 3
    assert r2.symmetries[0]["explicit"] is True


def test_event_time_parameter_is_coordinate():
    ev = [event("u", 0, "0", "replace"), event("u", "tau", "1", "replace")]
    r = detect({"x": "-k1*x + u*z", "z": "-(1-u)*k2*z", "u": "0"}, {"y": "x"}, events=ev,
               reconstruct=True)
    assert "tau" in r.info["coordinates"]
    assert r.rank == 4
    gen = r.symmetries[0]["generator"]
    assert set(gen) == {"z", "k2"}
    assert expr_equal(gen["z"], f"tau*z*({gen['k2']})")
    # a jump or a kink in the output reveals the time of a dose
    evd = [event("A", "tau", "d", "add")]
    rj = detect({"A": "-k*A"}, {"y": "A"}, events=evd, trafo={"A": "0"})
    assert rj.identifiable is True
    rk = detect({"D": "-ka*D", "A": "ka*D - k*A"}, {"y": "A"}, events=[event("D", "tau", "d", "add")],
                trafo={"D": "0", "A": "0"})
    assert rk.identifiable is True
    # a switch the output does not feel leaves its time free
    ri = detect({"A": "-k*A", "u": "0"}, {"y": "A"},
                events=[event("u", 0, "0"), event("u", "tau", "1")], reconstruct=True)
    assert ri.rank == 2
    assert list(ri.symmetries[0]["generator"]) == ["tau"]
    bad = [event("A", 0, "d", "add"), event("A", 5, "d", "add"), event("A", "tau", "d", "add"),
           event("A", 2, "d", "add")]
    with pytest.raises(Exception, match="ascending"):
        detect({"A": "-k*A"}, {"y": "A"}, events=bad)
    with pytest.raises(Exception, match="modular"):
        detect({"A": "-k*A"}, {"y": "A"}, events=evd, engine="symbolic")


def test_log_parameters_and_log_observables():
    f = {"A": "-k1*A + k2*B", "B": "k1*A - k2*B"}
    tr = {"k1": "exp(lk1)", "k2": "exp(lk2)", "s": "exp10(ls)", "B": "10^lB"}
    lin = detect(f, {"y": "s*A"}, reconstruct=True)
    lg = detect(f, {"y": "log10(s*A)"}, trafo=tr, reconstruct=True)
    assert lg.rank == lin.rank
    assert set(lg.info["coordinates"]) == {"A", "lB", "lk1", "lk2", "ls"}
    # the scaling of B and s is a translation of lB and ls at rate 1/log(10)
    sc = next(d for d in lg.symmetries if set(d["generator"] or {}) == {"A", "lB", "ls"})["generator"]
    assert expr_equal(sc["lB"], f"({sc['A']})/(A*log(10))")
    assert expr_equal(sc["ls"], f"-({sc['A']})/(A*log(10))")
    sy = detect(f, {"y": "log10(s*A)"}, trafo=tr, engine="symbolic")
    assert sy.rank == lg.rank
    # an offset on a log observable is a translation; a scale on it needs A > 0
    off = detect(f, {"y": "log(A) + c"}, reconstruct=True)
    assert off.rank == 3
    with pytest.raises(Exception, match=r"a\*log\(h\)"):
        detect(f, {"y": "s*log(A)"}, positive=False)
    detect(f, {"y": "s*log(A)"})
    # a parameter that also enters outside an exponent stays a coordinate
    both = detect({"A": "-exp(lk)*A + lk"}, {"y": "A"})
    assert both.identifiable is True
    assert set(both.info["coordinates"]) == {"A", "lk"}


def test_single_segment_without_later_events():
    r = detect({"A": "-k*A"}, {"y": "s*A"}, events=[event("A", 0, "d0", "add")],
               conditions=grid(["c1"], d0=[1]))
    assert r.info["segments"] == 1
    assert r.dim == 3
    assert r.identifiable is False


def test_transient_channel_parameter_identified():
    # kinh acts only on the boundary state at t = 0, reached through the propagated gap
    f = {"x": "kpr/(1 + kinh*inh) - kdeg*x + kstim*stim", "inh": "0", "stim": "0"}
    ev = [event("inh", -30, "1", "replace"), event("stim", 0, "1", "replace")]
    r30 = detect(f, {"y": "s*x"}, equilibrate=True, events=ev, conditions=grid(["c1"]),
                 inputs=["inh", "stim"])
    assert r30.info["segments"] == 2
    assert r30.info["gap_order_used"] >= 1
    assert "kinh" not in supports(r30)


def test_equilibrate_seeds_later_segment():
    ev = [event("x", 0, "dose", "replace"), event("x", 1, "d1", "add")]
    r = detect({"x": "b - a*x"}, {"y": "s*x"}, equilibrate=True, events=ev,
               conditions=grid(["stim"], dose=[2], d1=[1]))
    assert r.info["conditions"] == 1
    assert r.info["segments"] == 2
    assert r.identifiable is True


def test_free_hill_feedback_weighted_scaling():
    # k_fb * k_pf^nh is invariant: a scaling of k_pf compensated by k_fb with weight -nh
    f = {"R": "k_pr/(1 + k_fb*FB^nh) - k_dg*R + k_stim*u", "FB": "k_pf*R - k_df*FB", "u": "0"}
    r = detect(f, {"R_obs": "s*R"}, equilibrate=True, inputs=["u"],
               events=[event("u", 0, "dose", "replace")], conditions=grid(["c1"], dose=[1]),
               reconstruct=True)
    assert all(d["explicit"] for d in r.symmetries)
    hill = [d for d in r.symmetries if set(d["support"]) == {"k_fb", "k_pf"}]
    assert len(hill) == 1
    assert hill[0]["explicit"] is True
    assert "nh" in " ".join(str(w) for w in hill[0]["weights"].values())


def test_implicit_steady_state_multi_condition():
    f = {"x1": "v1 - k*x1/(K + x1)", "x2": "v2 - k*x2/(K + x2)"}
    r = detect(f, {"y": "s*(x1 + x2)"}, equilibrate=True, conditions=grid(["c1", "c2"], k=["k1", "k2"]),
               reconstruct=False)
    assert r.identifiable is False
    assert r.dim == 6
    assert r.rank == 2
    assert len(r.symmetries) == 4


def test_per_condition_trafo_matches_grid():
    f, g = {"A": "-p*A"}, {"y": "s*A"}
    r_grid = detect(f, g, conditions=grid(["c1", "c2"], p=["pa", "pb"]))
    r_traf = detect(f, g, trafo=[{"p": "pa"}, {"p": "pb"}])
    assert r_traf.info["conditions"] == 2
    assert r_traf.rank == r_grid.rank
    assert r_traf.dim == r_grid.dim
    r_bake = detect(f, g, trafo=[{"p": "1/2"}, {"p": "3/2"}])
    r_bgrid = detect(f, g, conditions=grid(["c1", "c2"], p=[0.5, 1.5]))
    assert r_bake.rank == r_bgrid.rank
    assert r_bake.dim == r_bgrid.dim


def test_explicit_steady_state_matches_equilibrate():
    # the resting state x* = b/a as initial value versus the f = 0 constraint
    rx = network(("", "x", "b"), ("x", "", "a*x"))
    g = {"y": "s*x"}
    r_trafo = detect(reactions=rx, observables=g, trafo={"x": "b/a"}, reconstruct=True)
    r_equil = detect(reactions=rx, observables=g, equilibrate=True, reconstruct=True)
    assert r_trafo.rank == r_equil.rank
    assert r_trafo.dim == r_equil.dim
    assert r_trafo.identifiable == r_equil.identifiable
    assert sorted(supports(r_trafo)) == sorted(supports(r_equil))


def test_scalings_with_events_and_conditions():
    g = {"Aobs": "alpha * A"}
    base = detect(reactions=canonical(), observables=g, scalings_only=True)
    assert len(base.symmetries) == 1
    # a known dose pins the absolute value of A
    pinned = detect(reactions=canonical(), observables=g, scalings_only=True,
                    events=[event("A", 0, "2", "replace")])
    assert len(pinned.symmetries) == 0
    # a second condition fixing alpha drops the shared scaling
    drop = detect(reactions=canonical(), observables=g, scalings_only=True,
                  trafo=[{"k1": "k1"}, {"alpha": "1"}])
    assert len(drop.symmetries) == 0
    keep = detect(reactions=canonical(), observables=g, scalings_only=True,
                  conditions=grid(["c1", "c2"], k1=["ka", "kb"]))
    assert len(keep.symmetries) == 1


def test_toric_peel_hill_weight():
    # k_inh * p^nhill is invariant under p -> lam*p, k_inh -> lam^(-nhill)*k_inh
    r = detect({"p": "kpr/(1 + kinh*p^nhill) - dp*p + kin*u", "u": "0"}, {"y": "s*p"},
               equilibrate=True, events=[event("u", -1, "var_u", "replace")],
               conditions=grid(["ctrl", "stim"], var_u=[0, 1]), reconstruct=True)
    hill = [d for d in r.symmetries if d["type"] == "scaling" and "kinh" in (d["generator"] or {})]
    assert len(hill) == 1
    assert hill[0]["explicit"] is True
    assert expr_equal(hill[0]["weights"]["kinh"], "-nhill")


def test_constant_shift_degree_zero():
    # only the total A + B is observed: a constant translation, not a scaling
    r = detect({"A": "-k1*A + k2*B", "B": "k1*A - k2*B"}, {"y": "A + B"}, reconstruct=True)
    ab = [d for d in r.symmetries if {"A", "B"} <= set(d["generator"] or {})]
    assert len(ab) >= 1
    assert ab[0]["type"] == "general"
    assert int(ab[0]["degree"]) == 0
    assert expr_equal(ab[0]["generator"]["A"], "1")
    assert expr_equal(ab[0]["generator"]["B"], "-1")


def test_state_symbol_in_trafo_is_initial_value():
    ab = {"A": "-k1*A + k2*B", "B": "k1*A - k2*B"}
    tr = {"k1": "k2*B/A"}
    # started at rest and left alone only alpha*A is identifiable
    r = detect(ab, {"Aobs": "alpha*A"}, trafo=tr, reconstruct=True)
    assert set(r.info["coordinates"]) == {"A", "B", "alpha", "k2"}
    assert r.rank == 1
    assert r.dim == 4
    # a dose on A moves the resting state, leaving one direction
    r2 = detect(ab, {"Aobs": "alpha*A"}, trafo=tr, events=[event("A", 0, "1", "add")], reconstruct=True)
    assert set(r2.info["coordinates"]) == {"A", "B", "alpha", "k2"}
    assert r2.rank == 3
    assert len(r2.symmetries) == 1
    names = [x for d in r2.symmetries for k, v in (d["generator"] or {}).items() for x in (k, str(v))]
    assert not any("_init" in x for x in names)


def test_grid_column_naming_state_fixes_initial_value():
    f, g = {"A": "-k*A"}, {"y": "s*A"}
    cg = grid(["c1", "c2"], A=[1, 2])
    r = detect(f, g, conditions=cg, reconstruct=True)
    assert r.identifiable is True
    assert set(r.info["coordinates"]) == {"k", "s"}
    r2 = detect(f, g, reconstruct=True, trafo=[{"A": "1"}, {"A": "2"}])
    assert r.rank == r2.rank
    assert r.dim == r2.dim
    rs = detect(f, g, scalings_only=True, conditions=cg)
    assert not any("A" in (d["generator"] or {}) for d in rs.symmetries)


def test_lie_order_saturates_per_condition():
    # the sparse condition reads x1 alone and needs five orders to reach b
    f = {"x1": "x2", "x2": "x3", "x3": "x4", "x4": "x5", "x5": "-(a + u*b)*x1"}
    g = [{"y1": "x1", "y2": "x2", "y3": "x3", "y4": "x4", "y5": "x5", "y6": "a*x1"}, {"w": "x1"}]
    cg = grid(["rich", "sparse"], u=[0, 1])
    res = detect(f, g, conditions=cg)
    assert res.identifiable is True
    assert res.rank == res.dim
    assert res.info["lie_order_used"] == 5
    # zero-based index of the sparse condition
    assert res.info["lie_order_driver"] == 1
    with state.switches(LIEPLATEAU_BLOCK="0"), warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        stacked = detect(f, g, conditions=cg)
    assert not rec
    assert stacked.identifiable is True
    assert stacked.info["lie_order_used"] >= 5


def test_lie_order_certified_by_rank_bound():
    free = detect({"A": "-k1*A", "B": "k1*A - k2*B"}, [{"y": "s*A"}, {"y": "s*B"}],
                  conditions=grid(["c1", "c2"]))
    assert free.info["lie_certified"] is True
    # a steady state as initial value
    ss = detect({"x": "b - a*x", "z": "a*x - c*z"}, {"y": "s*x"}, trafo={"x": "b/a", "z": "b/c"},
                conditions=grid(["c1", "c2"], b=[1, 2]))
    assert ss.info["lie_certified"] is True
    # a fixed initial value at which the lower rows vanish
    pin = detect({"x": "th"}, {"y": "x^5"}, trafo={"x": "0", "th": "th"})
    assert pin.info["lie_certified"] is True
    assert pin.identifiable is True
    assert pin.info["lie_order_used"] == 5
    # a state that stays at zero
    zs = detect({"x1": "-k1*x1", "x2": "k1*x1 - k2*x2 + a*x3", "x3": "-k3*x3"}, {"y": "x2"},
                trafo={"x1": "1", "x2": "0", "x3": "0", "k1": "k1", "k2": "k2", "k3": "k3", "a": "a"})
    assert zs.info["lie_certified"] is True
    assert zs.rank == 2
    # a start state on an invariant affine space
    li = detect({"x1": "a - (d + f)*x1", "x2": "b - (d + f)*x2", "f": "-c*f"}, {"y": "x1 + x2"},
                trafo={"x1": "a/d", "x2": "b/d", "f": "1", "a": "a", "b": "b", "c": "c", "d": "d"})
    assert li.info["lie_certified"] is True
    assert li.rank == 3
    # exact scalings prove the rank; a scaling that holds only at rest does not
    sw = detect({"A": "-k1*A", "B": "k1*A - k2*B"}, {"y": "s*B"},
                trafo={"A": "A0", "B": "0", "k1": "k1", "k2": "k2", "s": "s"})
    assert sw.info["rank_proven"] == 1
    rs = detect({"x": "k1 - k2*x"}, {"y": "x"}, trafo={"x": "k1/k2", "k1": "k1", "k2": "k2"})
    assert rs.info.get("rank_proven") is None
    # an event time among the coordinates
    td = detect({"A": "-k*A"}, {"y": "A"}, events=[event("A", "tau", "d", "add")],
                trafo={"A": "0", "k": "k", "d": "d", "tau": "tau"})
    assert td.info["lie_certified"] is True
    assert td.identifiable is True


def test_complete_generator_flow():
    r = detect(reactions=canonical(), observables={"y": "s * B"}, reconstruct=True)
    sc = next(d for d in r.symmetries if d["type"] == "scaling")
    gn = next(d for d in r.symmetries if d["type"] == "general")
    assert sc["complete_generator"] == sc["generator"]
    assert sc["factor"] == "1"
    # complete_generator = factor * generator, the factor positive on the orthant
    for k in gn["generator"]:
        assert expr_equal(gn["complete_generator"][k], f"({gn['factor']})*({gn['generator'][k]})")
    rng = np.random.default_rng(1)
    pts = [dict(zip(["A", "B", "k1", "k2"], np.exp(rng.normal(0, 3, 4)))) for _ in range(50)]
    syms = spy.symbols("A B k1 k2")
    fac = spy.lambdify(syms, sym(gn["factor"]))
    assert all(fac(*pt.values()) > 0 for pt in pts)
    # in log coordinates the speed is at most 1/2, so the flow exists for all s
    comp = {k: spy.lambdify(syms, sym(v)) for k, v in gn["complete_generator"].items()}
    speed = [np.sqrt(sum((fn(*pt.values()) / pt[k]) ** 2 for k, fn in comp.items())) for pt in pts]
    assert all(s <= 0.5 + 1e-12 for s in speed)
    # with nothing declared positive the bound holds in linear coordinates
    r0 = detect(reactions=canonical(), observables={"y": "s * B"}, reconstruct=True, positive=False)
    g0 = next(d for d in r0.symmetries if d["type"] == "general")
    comp0 = [spy.lambdify(syms, sym(v)) for v in g0["complete_generator"].values()]
    speed0 = [np.sqrt(sum(fn(*pt.values()) ** 2 for fn in comp0)) for pt in pts]
    assert all(s <= 0.5 + 1e-12 for s in speed0)
    # neither repr() nor summary() shows the factor
    out = repr(r) + "\n" + str(r.summary())
    assert gn["factor"] not in out


_AUTOCRINE_STEADY_STATE = {
    "m": "ktx*(C + b0)/dm",
    "P": "ktl*ktx*(C + b0)*(Km + R1 + R2 + R3)/(dm*(Km*ksec + R1*ksec + R2*ksec + R3*ksec + kdg))",
    "kint": "ksec*ktl*ktx*r_C_1*r_L_1*(C + b0)*(Km + R1 + R2 + R3)/(C*dm*(r_C_1*r_L_1 + r_C_1 + 1)"
            "*(Km*ksec + R1*ksec + R2*ksec + R3*ksec + kdg))",
    "koff": "ksec*ktl*ktx*r_L_1*(C + b0)*(Km + R1 + R2 + R3)/(C*dm*(r_C_1*r_L_1 + r_C_1 + 1)"
            "*(Km*ksec + R1*ksec + R2*ksec + R3*ksec + kdg))",
    "ksR": "(R*kdR*r_C_1*r_L_1 + R*kdR*r_C_1 + R*kdR*r_L_1 + R*kdR + ksec*ktl*ktx*r_C_1*r_L_1**2*(C + b0)"
           "*(Km + R1 + R2 + R3)/(dm*(r_C_1*r_L_1 + r_C_1 + 1)*(Km*ksec + R1*ksec + R2*ksec + R3*ksec + kdg))"
           " + ksec*ktl*ktx*r_C_1*r_L_1*(C + b0)*(Km + R1 + R2 + R3)/(dm*(Km*ksec + R1*ksec + R2*ksec"
           " + R3*ksec + kdg)))/(r_C_1*r_L_1 + r_C_1 + r_L_1 + 1)",
    "kon": "ksec*ktl*ktx*r_L_1*(C + b0)*(r_L_1/(r_C_1*r_L_1 + r_C_1 + 1) + 1)*(Km + R1 + R2 + R3)"
           "/(L*R*dm*(r_L_1 + 1)*(Km*ksec + R1*ksec + R2*ksec + R3*ksec + kdg))",
    "kL": "ksec*ktl*ktx*(C + b0)*(r_L_1/(r_C_1*r_L_1 + r_C_1 + 1) + 1)*(Km + R1 + R2 + R3)"
          "/(L*dm*(r_L_1 + 1)*(Km*ksec + R1*ksec + R2*ksec + R3*ksec + kdg))",
}


def test_sum_of_leaves_translation_group():
    # the curved direction trading kdg against ksec has the entry ksec*(Km + R1 + R2 + R3)^2;
    # the trafo is the steady state of the network solved for some rates
    rx = network(("", "m", "ktx*(b0 + C)"), ("m", "", "dm*m"), ("", "P", "ktl*m"),
                 ("P", "", "kdg*P/(Km + R1 + R2 + R3)"), ("P", "L", "ksec*P"), ("L", "", "kL*L"),
                 ("", "R", "ksR"), ("R", "", "kdR*R"), ("L + R", "C", "kon*L*R"),
                 ("C", "L + R", "koff*C"), ("C", "", "kint*C"))
    g = {"ym": "sm*m", "yC": "sC*C", "yL": "L", "yR1": "R1", "yR2": "R2", "yR3": "R3"}
    args = dict(reactions=rx, observables=g, trafo=_AUTOCRINE_STEADY_STATE,
                events=[event("L", 0, "dose", "add")], conditions=grid(["ctrl", "stim"], dose=[0, 1]),
                reconstruct=True)
    with state.switches(NOGROUP=False):
        res = detect(**args)
    assert all(d["explicit"] for d in res.symmetries)
    wide = next(d for d in res.symmetries if "ksec" in (d["generator"] or {}))
    assert expr_equal(f"({wide['generator']['Km']})/({wide['generator']['ksec']})",
                      "(Km + R1 + R2 + R3)**2/kdg")
    with state.switches(NOGROUP=True):
        res0 = detect(**args)
    w0 = next(d for d in res0.symmetries if "ksec" in (d["generator"] or {}))
    for v in wide["generator"]:
        assert expr_equal(wide["generator"][v], w0["generator"][v])


def test_translation_groups_test_only_pairs_alike_under_single_moves():
    import numpy as np

    from symident.detection.groups import translation_groups
    P = 1000003
    calls = []

    def kb(pts, primes, nt):
        calls.append(len(pts))
        return [{"ok": True, "pivots": [0], "v": np.array([(z[0] + z[1]) % P, z[2] * z[3] % P, z[4] % P])}
                for z in pts]
    base = [2, 3, 5, 7, 11]
    ref = kb([base], [P], 4)[0]
    rel = []
    for li in range(5):
        z = list(base)
        z[li] += 1
        rel.append({"rp": kb([z], [P], 4)[0]})
    sc = {"point0": base, "pool": lambda idx: np.atleast_1d(idx).astype(np.int64) * 7 + 3, "pool_next": 0,
          "nt_used": 4}
    calls.clear()
    groups = translation_groups(rel, ref, lambda o, p: o["v"], kb, sc, P, 5)
    assert [(g["type"], g["rep"], g["members"]) for g in groups] == [("add", 0, [1]), ("mul", 2, [3])]
    # one call at the point, then per pass the single moves and the pairs alike under them
    assert len(calls) == 5 and sum(calls) <= 1 + 5 + 1 + 3 + 1


def test_per_condition_observables_as_tuple():
    # per-condition observables given as a tuple match the list form
    f = {"A": "-k1*A", "B": "-k2*B"}
    g = ({"yA": "sA*A"}, {"yB": "sB*B"})
    res = detect(f, g, conditions=grid(["elisa", "wb"]))
    res0 = detect(f, list(g), conditions=grid(["elisa", "wb"]))
    assert res.rank == res0.rank
    assert res.dim == res0.dim


def test_symbolic_engine_free_hill_exponent():
    ss = {"FB": "k_pr_FB/d_FB", "x": "k_pr_x*K^n/(d_x*(K^n + (k_pr_FB/d_FB)^n))"}
    sres = detect(reactions=hill_michaelis(), observables={"xobs": "scale * x"}, engine="symbolic",
                  trafo=ss)
    assert any("n" in d["support"] for d in sres.symmetries)


def test_base_point_generic_beyond_prime_pool():
    # kdg + ksec = kt is where the two decay rates cannot be told apart
    f = {"P": "ktl - (kdg + ksec)*P", "T1": "ksec*P - kt*T1"}
    res = detect(f, {"y": "T1"}, trafo={"P": "0", "T1": "0"})
    assert res.rank == 3
    assert detect(f, {"y": "T1"}, trafo={"P": "0", "T1": "0"}, engine="symbolic").rank == 3


def test_transit_chain_readout_not_saturated_early():
    # the output vanishes up to Lie order n + 1
    n = 8
    f = {"P": "ktl - (kdg + ksec)*P", "T1": "ksec*P - kt*T1"}
    for i in range(2, n + 1):
        f[f"T{i}"] = f"kt*T{i - 1} - kt*T{i}"
    res = detect(f, {"y": f"T{n}"}, trafo=dict.fromkeys(f, "0"), reconstruct=True)
    assert res.rank == 3
    assert res.info["lie_order_used"] >= n + 3
    assert set(res.symmetries[0]["support"]) == {"kdg", "ksec", "ktl"}


def test_summary_suggests_gauge():
    f, g, cg = pool_model()
    res = detect(f, g, conditions=cg)
    gs = res.info.get("gauge_suggestion")
    assert gs is not None
    assert gs["gauge"] == ["K"]
    assert max(gs["sizes"]) <= 3
    assert "gauge: fixing K" in str(res.summary())


def test_coupling_gauge_preference_warns():
    f, g, cg = pool_model()
    with pytest.warns(UserWarning, match="leaves general directions on up to"):
        res = detect(f, g, conditions=cg, gauge=["T1"], reconstruct=True)
    assert res.gauge == ["T1"]
    res2 = detect(f, g, conditions=cg, gauge=None, reconstruct=True)
    assert res2.gauge == ["K"]
    gen = [d for d in res2.symmetries if d["type"] == "general"]
    assert all(len(d["generator"]) <= 3 for d in gen)


def test_stacked_rank_proven_by_fields():
    f, g, cg = pool_model()
    ref = detect(f, g, conditions=cg, reconstruct=True)
    with state.options(stacked_min=0):
        st = detect(f, g, conditions=cg, reconstruct=True)
    assert st.rank == ref.rank
    assert st.info["rank_proof"] == "proven symmetries"
    assert len(st.symmetries) == len(ref.symmetries)
    # the gauged detection takes rank and proof from its second step
    gd = detect(f, g, conditions=cg, gauge=None, reconstruct=True)
    assert gd.rank == ref.rank
    assert gd.info.get("rank_proven") is not None
    assert gd.gauge == ["K"]


def test_constant_state_events_per_segment():
    rx = network(("", "R", "kR"), ("R", "", "dR*R"), ("", "G", "kG*R*(b + u)"), ("G", "", "dG*G"),
                 ("", "u", "0"))
    g = {"yG": "sG*G"}
    ev = [event("u", 0, "dose", "add")]
    cg = grid(["c0", "stim"], dose=[0, 1])
    # a constant state is not zero at rest
    r0 = detect(reactions=rx, observables=g, equilibrate=True)
    assert r0.rank == 1
    # the dose relaxes G at rate dG, so b + u and dG are identifiable
    r1 = detect(reactions=rx, observables=g, conditions=cg, events=ev, equilibrate=True, reconstruct=True)
    assert r1.rank == 3
    assert "b+u" in dirset(r1)
    ss = {"R": "kR/dR", "G": "kG*kR/dR*(b + u)/dG", "u": "u"}
    r2 = detect(reactions=rx, observables=g, trafo=[ss, ss], conditions=cg, events=ev, reconstruct=True)
    assert r2.rank == 3
    assert "b+u" in dirset(r2)


def test_static_subsystem_proves_rank():
    rx = network(("", "m", "km*plasmid"), ("m", "", "dm*m"), ("", "E", "kt*m"), ("E", "", "dE*E"),
                 ("E", "X", "kex*E"), ("", "X", "kx"), ("X", "", "dX*X*(1 + L)"),
                 ("", "L", "kL"), ("L", "", "dL*L"))
    g = {"yX": "sX*(X + E)", "ym": "sm*m", "yL": "L"}
    ev = [event("L", 0, "dose", "add"), event("L", 5, "dose", "add")]
    r = detect(reactions=rx, observables=g, conditions=grid(["ctrl", "oe"], plasmid=[0, 1], dose=[1, 1]),
               events=ev, equilibrate=True, reduce_cq=True, gauge=None, reconstruct=True)
    assert r.rank == 7
    assert r.info["rank_proof"] == "exact scalings and static subsystems"
    # a subsystem that no event reaches stays at rest and has exact directions
    assert {d.level for d in r.symmetries} == {"exact"}
    st = [d for d in r.symmetries if d.get("route") == "static subsystem"]
    assert sorted("+".join(sorted(d["support"])) for d in st) == ["dE+kt", "dm+km"]
    g_e = next(d for d in st if "dE" in d["support"])["generator"]
    assert expr_equal(g_e["dE"], "dE + kex")


def test_log_coordinate_read_off_depends_on_own_coordinate():
    rx = network(("", "x1", "kb"), ("", "x1", "kin*u"), ("x1", "x2", "k1*x1"), ("x1", "", "kl*x1"),
                 ("x2", "", "k2*x2"))
    rx["species"].append("u")
    rx["stoichiometry"] = [row + [0] for row in rx["stoichiometry"]]
    r = detect(reactions=rx, observables={"y": "s*x2"}, inputs=["u"],
               events=[event("u", 0, "dose", "replace")], conditions=grid(["low", "high"], dose=[1, 10]),
               equilibrate=True, reduce_cq=True, reconstruct=True)
    assert r.rank == 4
    gen = next(d for d in r.symmetries if d.type == "general").generator
    assert set(gen) == {"k1", "kb", "kin", "kl"}
    ratio = {k: spy.simplify(spy.sympify(v.replace("^", "**")) / spy.sympify(gen["k1"].replace("^", "**")))
             for k, v in gen.items()}
    assert ratio == {"k1": 1, "kb": spy.sympify("-kb/k1"), "kin": spy.sympify("-kin/k1"), "kl": -1}


def test_lie_order_certified_per_condition_from_a_generic_state():
    m = {"odes": {"x1": "-x2", "x2": "k"}, "observables": {"y": "x1"}, "trafo": {"x2": "0"}}
    one = detect(**m)
    two = detect(**m, conditions=grid(["a", "b"], k=["k", "k"]))
    assert (one.rank, two.rank) == (2, 2)
    assert two.info["lie_order_used"] == 2


def test_failed_relevance_probe_does_not_freeze_a_rate():
    rx = network(("A", "B", "kf0*u*A"), ("B", "A", "kr0*B"), ("B", "C", "kf1*B"), ("C", "B", "kr1*C"),
                 ("D", "E", "V*C*D/(K + D)"), ("E", "D", "kr2*E"), ("E", "F", "kf3*E"), ("F", "E", "kr3*F"))
    r = detect(reactions=rx, observables={"y0": "s0*C", "y1": "s1*B"}, conditions=grid(["a", "b"], u=[1, 4]),
               equilibrate=True, reduce_cq=True, reconstruct=True)
    gens = [d.generator for d in r.symmetries if d.generator]
    assert {"kf0": "kf0", "kr0": "kr0"} in gens and {"kf1": "kf1", "kr1": "kr1"} in gens
    for g in gens:
        for v in g.values():
            assert all(abs(int(n)) < 1000 for n in re.findall(r"\b[0-9]+\b", v)), g


def test_constant_state_takes_its_initial_value_from_trafo():
    r = detect({"x1": "-a*x1", "x2": "0"}, {"y": "s*(x1 + x2)"}, trafo={"x2": "1"}, reconstruct=True)
    assert "x2" not in r.info["coordinates"]
    assert (r.rank, r.dim) == (3, 3)


def test_directions_span_the_kernel_without_duplicates():
    rx = network(("x1", "x2", "k1*x1"), ("x2", "", "k2*x2"), ("x3", "", "k3*x3"))
    r = detect(reactions=rx, observables={"y1": "x2"}, equilibrate=True, reduce_cq=True, reconstruct=True)
    assert (r.rank, r.dim) == (0, 3)
    gens = [tuple(sorted(d.generator)) for d in r.symmetries]
    assert len(gens) == 3 and len(set(gens)) == 3


def test_forward_closed_form_has_no_resting_state():
    rx = network(("x1", "x2", "V3*x1/(K4 + x1)"), ("", "x1", "k5"), ("", "x2", "k6"), ("x1", "", "k7*x1"),
                 ("x2", "", "k8*x2"), ("", "x1", "k10*u"))
    rx["species"].append("u")
    rx["stoichiometry"] = [row + [0] for row in rx["stoichiometry"]]
    r = detect(reactions=rx, observables={"y2": "x2"}, inputs=["u"], events=[event("u", 0, "dose", "replace")],
               conditions=grid(["c1"], dose=[5]), equilibrate=True, reduce_cq=True, reconstruct=True)
    gen = next(d for d in r.symmetries if d.type == "general").generator
    assert all("x1" not in str(v) for v in gen.values())
    ratio = spy.simplify(spy.sympify(gen["k5"].replace("^", "**")) / spy.sympify(gen["K4"].replace("^", "**")))
    assert spy.simplify(ratio + (spy.Symbol("K4") * spy.Symbol("k7") + spy.Symbol("V3")) / spy.Symbol("K4")) == 0


def test_unobserved_sink_state_leaves_the_resting_system():
    rx = network(("x1", "", "k1*x1"), ("x1", "x2", "k2*x1"), ("", "x3", "k3"), ("x3", "", "k4*x3"))
    r = detect(reactions=rx, observables={"y1": "x3"}, equilibrate=True, reduce_cq=True, reconstruct=True)
    assert (r.rank, r.dim) == (1, 4)
    assert sorted(tuple(sorted(d.generator)) for d in r.symmetries) == [("k1",), ("k2",), ("k3", "k4")]


def test_hill_resting_state_does_not_stall_static_fields():
    rx = network(("x1", "", "V2*x1^3/(K1^3 + x1^3)"), ("x1", "", "k3*x1"), ("", "x1", "k5"))
    r = detect(reactions=rx, observables={"y1": "s1*x1"}, conditions=grid(["c1", "c2"], K1=[7, 1]),
               equilibrate=True, reduce_cq=True, reconstruct=True)
    assert (r.rank, r.dim) == (2, 4)


def test_small_network_at_rest_finishes_quickly():
    rx = network(("A", "B", "k1*A"), ("B", "A", "k2*B"), ("B", "C", "k3*B"), ("C", "B", "k4*C"))
    r = detect(reactions=rx, observables={"y": "B"}, equilibrate=True, reduce_cq=True)
    assert (r.rank, r.dim) == (1, 5)


@pytest.mark.parametrize("odes, obs, rank_dim", [
    ({"x": "-k*x"}, {"y": "exp2(s*x) + acsch(s*x)"}, (2, 3)),
    ({"x": "-k*atanh(x)"}, {"y": "x"}, (2, 2)),
    ({"x": "-k*acot(x) - c*x"}, {"y": "x"}, (3, 3)),
    ({"x": "-k*asech(x)"}, {"y": "s*x"}, (3, 3)),
    ({"x": "-k*x"}, {"y": "acoth(1 + s*x)"}, (2, 3)),
])
def test_the_functions_of_the_backend_expressions(odes, obs, rank_dim):
    r = detect(odes, obs)
    assert (r.rank, r.dim) == rank_dim


def test_steps_at_every_integer_are_rejected():
    for fn in ("floor", "ceiling", "round"):
        with pytest.raises(ValueError, match=rf"{fn}\(\) in"):
            detect({"x": f"b - k*{fn}(x)"}, {"y": "x"})


def test_euler_number_is_no_coordinate():
    r = detect({"x": "b - k*exp(1)*abs(x)"}, {"y": "s*x"})
    assert set(r.info["coordinates"]) == {"x", "b", "k", "s"}


def test_pi_is_the_number():
    assert set(detect({"x": "-pi*k*x + E"}, {"y": "s*x"}).info["coordinates"]) == {"x", "k", "s", "E"}
    with pytest.raises(ValueError, match="pi is the number"):
        detect({"pi": "-k*pi"}, {"y": "s*pi"})


def test_fractional_power_of_a_positive_parameter():
    assert (detect({"x": "-sqrt(a)*x"}, {"y": "x"}).rank, detect({"x": "-a^(3/2)*x + b"}, {"y": "x"}).rank) == (2, 3)
    r = detect({"x": "-sqrt(a*b)*x"}, {"y": "x"}, reconstruct=True)
    assert [d.generator for d in r.symmetries] == [{"a": "a", "b": "-b"}]


@pytest.mark.parametrize("odes, obs, trafo, rank_dim", [
    ({"x": "-k*atan(x)"}, {"y": "x"}, None, (2, 2)),
    ({"x": "-k*log(1 + x^2) - d*x"}, {"y": "s*x"}, None, (4, 4)),
    ({"x": "-a*atan(b*x)"}, {"y": "x"}, None, (3, 3)),
    ({"x": "-k*x"}, {"y": "atan(s*x)"}, None, (2, 3)),
    ({"x": "-k*sqrt(1 + x^2)"}, {"y": "x"}, {"x": "0"}, (1, 1)),
    ({"x": "-k*asinh(x) + c"}, {"y": "x"}, {"x": "0"}, (2, 2)),
    ({"x": "-sqrt(a + b)*x"}, {"y": "x"}, None, (2, 3)),
    ({"x": "-k*log(x, 3)"}, {"y": "x"}, None, (2, 2)),
    ({"x": "-k*x + a*atan2(x, 2)"}, {"y": "x"}, None, (3, 3)),
    ({"x": "-k*atanh(x/(1 + x))"}, {"y": "x"}, None, (2, 2)),
])
def test_elementary_functions_agree_with_the_symbolic_engine(odes, obs, trafo, rank_dim):
    mod = detect(odes, obs, trafo=trafo)
    sym = detect(odes, obs, trafo=trafo, engine="symbolic")
    assert (mod.rank, mod.dim) == (sym.rank, sym.dim) == rank_dim


@pytest.mark.parametrize("odes, obs, trafo, rank_dim", [
    ({"x": "-k*sqrt(1 + x^2)"}, {"y": "x"}, None, (2, 2)),
    ({"x": "-k*x"}, {"y": "s*sqrt(1 + x^2)"}, None, (3, 3)),
    ({"x": "-k*sqrt(1 + exp(c*x))"}, {"y": "x"}, None, (3, 3)),
    ({"x": "-k*asin(x/(1 + x))"}, {"y": "x"}, None, (2, 2)),
    ({"x": "-k*sqrt(1 + x^2) + a"}, {"y": "s*x"}, {"x": "2"}, (3, 3)),
    ({"x": "-k*(1 + x^2)^(1/3)"}, {"y": "x"}, None, (2, 2)),
    ({"x": "-k*(1 + x^2)^(2/3) + a"}, {"y": "s*x"}, None, (4, 4)),
    ({"x": "-k*(1 + x^2)^(1/6)"}, {"y": "x"}, None, (2, 2)),
    ({"x": "-k*(1 + x^2)^(1/5)"}, {"y": "x"}, {"x": "1"}, (1, 1)),
])
def test_rational_powers_of_initial_values_are_exact(odes, obs, trafo, rank_dim):
    from symident import gfp
    mod = detect(odes, obs, trafo=trafo)
    sym = detect(odes, obs, trafo=trafo, engine="symbolic")
    assert (mod.rank, mod.dim) == (sym.rank, sym.dim) == rank_dim
    assert gfp.PRIMES[0] == 1518500183 and gfp.verify_prime() == 1518500141


def test_odd_root_rerun_restores_the_primes():
    from symident import gfp
    rx = detect({"x": "-k*(1 + x^2)^(1/3)"}, {"y": "x"})
    assert rx.rank == 2
    assert list(gfp.PRIMES) == [1518500183, 1518500173, 1518500171, 1518500143]
    assert gfp.holds_roots([], 1518500171, [3]) and not gfp.holds_roots([], 1518500173, [3])


def test_a_constant_with_an_engine_prime_as_denominator_reruns_over_other_primes():
    r = detect({"x": "-k*x/1518500183"}, {"y": "s*x"},
               conditions=grid(["a", "b", "c"], k=["k1", "k2", "k3"]), cores=4)
    assert (r.rank, r.dim) == (4, 5)


def test_use_primes_switches_every_prime_together():
    from symident import gfp
    before = gfp.all_primes()
    with gfp.use_primes([], [3]):
        run = gfp.prime_run(12)
        assert len(set(gfp.all_primes())) == 8 and len(set(run)) == 12
        assert all(gfp.holds_roots([], q, [3]) for q in gfp.all_primes() + run)
        assert gfp.verify_prime() not in run and run[:4] == list(gfp.PRIMES)
    assert gfp.all_primes() == before


def test_forward_route_moves_its_solved_rate():
    rx = network(("x1", "", "k2*x1^2"), ("x1", "", "k3*x1"), ("", "x1", "k5"))
    r = detect(reactions=rx, observables={"y1": "s1*x1"}, equilibrate=True, reduce_cq=True, reconstruct=True)
    assert (r.rank, r.dim) == (1, 4)
    assert not any(set(d.generator or {}) in ({"k3"}, {"k5"}) for d in r.symmetries)
    supports = sorted(tuple(sorted(d.support)) for d in r.symmetries)
    assert ("k2", "k3") in supports and ("k2", "k5") in supports


def test_sink_behind_a_state_at_zero_leaves_the_resting_system():
    rx = network(("x1", "x2", "k1*x1"), ("x1 + x2", "", "k2*x1*x2"))
    r = detect(reactions=rx, observables={"y1": "x1"}, equilibrate=True, reduce_cq=True, reconstruct=True)
    assert (r.rank, r.dim) == (0, 2)
    assert sorted(tuple(d.generator) for d in r.symmetries) == [("k1",), ("k2",)]


def test_directions_stay_independent_under_several_resting_conditions():
    rx = network(("M0s2", "M0s1", "kr01*M0s2"), ("M1s0", "M1s1", "kf10*u*M1s0"), ("M1s1", "M1s0", "kr10*M1s1"),
                 ("M1s1", "M1s2", "kf11*u*M1s1"), ("M1s2", "M1s1", "kr11*M1s2"), ("", "P", "kin*M1s2"),
                 ("P", "", "kdeg*P"))
    r = detect(reactions=rx, observables={"y1": "s1*P"}, conditions=grid(["a", "b"], u=[1, 4]),
               equilibrate=True, reduce_cq=True, reconstruct=True)
    assert len(r.symmetries) == r.dim - r.rank


@pytest.mark.parametrize("odes, obs, rank_dim", [
    ({"x": "-(k1 + k2)*x"}, {"y": "x"}, (0, 2)),
    ({"x": "-k1*x"}, {"y": "x"}, (0, 1)),
    ({"x1": "x2 - x1 + k4", "x2": "-k1*x2 - k2*x2"}, {"y": "x1"}, (1, 3)),
])
def test_ode_state_resting_at_zero(odes, obs, rank_dim):
    r = detect(odes, obs, equilibrate=True, reconstruct=True)
    assert (r.rank, r.dim) == rank_dim


def test_ode_zero_rest_with_an_input_pulse():
    r = detect({"x1": "x2 - x1 + k4", "x2": "-k1*x2 - k2*x2 + u", "u": "-u"}, {"y": "x1"}, inputs=["u"],
               events=[event("u", 0, "dose", "replace")], conditions=grid(["c"], dose=[3]),
               equilibrate=True, reconstruct=True)
    assert (r.rank, r.dim) == (2, 3)
    assert [d.generator for d in r.symmetries] == [{"k1": "1", "k2": "-1"}]


def test_small_root_finder_matches_sympy():
    import random

    from sympy.polys import galoistools as gft
    from sympy.polys.domains import ZZ

    from symident.sym.gfpoly import _x_pow_mod
    p, rng = 1518500183, random.Random(3)
    for n in range(1, 9):
        f = [1] + [rng.randrange(p) for _ in range(n)]
        assert _x_pow_mod(f, p) == gft.gf_strip(gft.gf_pow_mod([1, 0], p, f, p, ZZ))


def test_unobserved_production_branch_keeps_independent_directions():
    rx = network(("A", "B", "kf*u*A"), ("B", "A", "kr*B"), ("", "P", "kin*B"), ("P", "", "kdeg*P"))
    for gauge in (None, False):
        r = detect(reactions=rx, observables={"y": "A + B"}, conditions=grid(["a", "b"], u=[1, 4]),
                   equilibrate=True, reduce_cq=True, reconstruct=True, gauge=gauge)
        assert (r.rank, r.dim, len(r.symmetries)) == (1, 5, 4)
        assert r.reduce().remaining == []


def test_modular_values_take_decimals_exactly_and_refuse_irrationals():
    from symident.sym.modp import eval_rational_mod
    assert eval_rational_mod("0.5*k", ["k"], [3], 7) == 3 * pow(2, 5, 7) % 7
    with pytest.raises(TypeError):
        eval_rational_mod("sqrt(k)", ["k"], [2], 7)


def test_direction_check_fails_where_a_leaf_component_does_not_evaluate():
    from symident.detection.verify import verify_direction
    znames, leaves = ["a", "b", "x_c1"], ["a", "b"]

    def kcall(pt, q, nt):
        return {"ok": True}

    def residues(rq, q, z):
        return np.array([1, 2, 5])
    ok = {"vector": {"a": "1", "b": "2", "x_c1": "x_c1^2"}}
    assert verify_direction(ok, 0, znames, leaves, [3, 4], 1, kcall, residues)
    pole = {"vector": {"a": "1", "b": "2 + 1/(a - a)"}}
    assert not verify_direction(pole, 0, znames, leaves, [3, 4], 1, kcall, residues)
    wrong = {"vector": {"a": "1", "b": "3"}}
    assert not verify_direction(wrong, 0, znames, leaves, [3, 4], 1, kcall, residues)


def test_rate_zero_test_is_exact_for_rational_rates():
    from symident.detection.cq import _expr_is_zero
    assert not _expr_is_zero("1e-10*x", set())
    assert _expr_is_zero("k*x*y", {"y"}) and _expr_is_zero("k1*x - k1*x", set())
    assert not _expr_is_zero("k/(1 + y)", {"y"})


def test_rate_zero_test_is_exact_for_other_rates():
    from symident.detection.cq import _expr_is_zero
    assert _expr_is_zero("k*exp(x)*y", {"y"}) and _expr_is_zero("log(1 + y)*k", {"y"})
    assert _expr_is_zero("sqrt(k)*x - k^(1/2)*x", set())
    assert not _expr_is_zero("1e-12*exp(x)", set()) and not _expr_is_zero("exp(y) - 1 + x", {"y", "z"})


def test_the_sink_weights_solve_the_lp_exactly():
    from symident.detection.cq import _exact_lp, _sink_cluster
    # rows are reactions, columns species
    assert _sink_cluster(np.array([[-1.0, 0.0], [0.0, 1.0]])) == [0]
    assert _sink_cluster(np.array([[-1.0, 1.0], [1.0, -1.0]])) == []
    # a weight ratio with a denominator past what a rounded float recovers
    p, q = 1000003, 999983
    M = [[-p, q], [p, -q], [-1, 0]]
    assert _sink_cluster(np.array(M, dtype=float)) == [0]
    assert _exact_lp([[Fraction(a) for a in row] for row in M], [Fraction(-1), Fraction(0)]) == \
        [Fraction(q, p), Fraction(1)]


def test_scalings_only_reports_terms_left_out():
    r = detect({"x": "-k*x^n"}, {"y": "s*x"}, scalings_only=True)
    assert any(lim.startswith("scalings:") for lim in r.info["limits"])
    r = detect({"x": "-k*x"}, {"y": "s*x"}, scalings_only=True)
    assert r.info["limits"] == []


def test_names_that_sympy_holds_as_functions_or_constants_are_symbols():
    r = detect({"S": "b - k*S", "N": "k*S - d*N"}, {"y": "s*N"}, reconstruct=True)
    assert (r.rank, r.dim) == (5, 6)
    gens = [g.sympy() for g in r.symmetries]
    assert all(isinstance(k, str) and v.free_symbols <= {spy.Symbol(n) for n in ("S", "N", "b", "k", "d", "s")}
               for g in gens for k, v in g.items())


def _kernel_ref(vectors, nz, p):
    """The reduced rows of a matrix whose kernel the integer `vectors` span."""
    from symident.gfp import rref_modp
    K = spy.Matrix(vectors).T
    A = [list(v.T) for v in K.T.nullspace()]
    A = [[int(x * spy.lcm([spy.fraction(y)[1] for y in row])) for x in row] for row in A]
    rr = rref_modp(np.array(A, dtype=np.int64) % p, p)
    return {"R": rr["R"][:rr["rank"]], "pivots": rr["piv"], "rank": rr["rank"], "dim": nz, "ok": True}


def test_minimal_support_beyond_six_coordinates():
    from symident.gfp import PRIMES, minsupport_gauge
    p, nz = PRIMES[0], 14
    wide = [1, 2, 3, 4, 5, 6, 7, 8, 9] + [0] * 5
    narrow = [0] * 9 + [1, 1, 0, 0, 0]
    scaling = [0] * 10 + [1, 0, 1, 1]
    ref = _kernel_ref([wide, narrow, scaling], nz, p)
    free = [c for c in range(nz) if c not in set(int(x) for x in ref["pivots"])]
    resid = [fc for fc in free][:2]
    ms = minsupport_gauge(resid, np.array([scaling]) % p, p, nz, {"ref": ref}, free, cand_cap=2000)
    supports = sorted(len(np.nonzero(ms["vectors"][:, j] % p)[0]) for j in range(ms["vectors"].shape[1]))
    assert supports == [2, 9]


def test_entry_degrees_along_lines():
    from symident.detection.interp import entry_degrees
    from symident.gfp import PRIMES
    p = PRIMES[0]

    def kbatch(pts, primes, nt):
        return [{"ok": True, "pivots": [0], "pt": pt} for pt in pts]

    def nv(rp, q, z):
        x1, x2, x3 = (int(rp["pt"][i]) for i in (1, 2, 3))
        inv = lambda a: pow(a % p, p - 2, p)  # noqa: E731
        return [1, (x1 * x1 + x2) * inv(x1 + 3) % p, x1 * x2 * x3 % p, inv(pow(x1, 5, p))]
    rel = {"relevant": [1, 2, 3], "support_cols": [1, 2, 3], "rel_by_entry": [[1, 2], [1, 2, 3], [1]]}
    ctrl = si.reconst_control()
    # an entry past degree_cap is found when the bound doubles, and stays over below it
    assert entry_degrees(0, [0], [5, 7, 11, 13], nv, lambda pt: None, rel, kbatch, 4, ctrl) == [2, 3, 5]
    ctrl = si.reconst_control(degree_cap_max=4)
    assert entry_degrees(0, [0], [5, 7, 11, 13], nv, lambda pt: None, rel, kbatch, 4, ctrl) == [2, 3, "over"]


def test_several_resting_branches_are_recorded():
    f = {"x": "c + a*x - b*x^3 + kin*u - d*x*z", "z": "e*x - g*z", "u": "0"}
    m = si.Model(f, {"y": "s*x"}, inputs=["u"],
                 events=[{"var": "u", "time": 0, "value": "1", "method": "replace"}])
    assert si.detect(m, equilibrate=True).info["branches"] == []
    with si.switches(NOFWDSEED=True):
        r = si.detect(m, equilibrate=True)
    assert r.info["branches"] == ["resting states x, z: several interior roots modulo p, the first is taken"]
    assert any(line.startswith("  branch: resting states x, z") for line in str(r.summary()).split("\n"))


def test_a_start_on_a_curved_invariant_set_leaves_the_lie_order_unproven():
    m = si.Model({"x1": "-w*x2 - k*x1*(x1^2 + x2^2 - 1)", "x2": "w*x1 - k*x2*(x1^2 + x2^2 - 1)"}, {"y": "s*x1"},
                 trafo={"x1": "1", "x2": "0"})
    r = si.detect(m, reconstruct=True)
    assert (r.rank, r.dim) == (2, 3) and [s.support for s in r.symmetries] == [["k"]]
    assert r.info["lie_certified"] is False
    assert r.info["rank_status"] == {"status": "lower bound", "why": "the Lie order is not proven sufficient",
                                     "probability": None}


def _autocrine(times, **kw):
    f = {"L": "ksec*m/(V*vf) - kon*L*R + koff*C - kdL*L", "R": "kR - kon*L*R + koff*C - kdR*R",
         "C": "kon*L*R - koff*C - kc*C", "m": "kb + ka*C - dm*m"}
    return detect(f, {"y": "s*m", "yC": "C"}, conditions=grid(["a", "b"], vf=[1, 1.5], dose=[1, 1]),
                  events=[event("L", t, "dose", "add") for t in times], equilibrate=True, reconstruct=True,
                  **kw)


@pytest.mark.parametrize("times", [[0], [0, 10]])
def test_the_diagonal_start_lifts_coupled_conditions_without_a_resting_solve(monkeypatch, times):
    from symident.detection.observability import rest

    with si.switches(NODIAG=True):
        ref = _autocrine(times)

    def no_solve(*a, **k):
        raise AssertionError("resting state solved at a point")
    monkeypatch.setattr(rest, "solve_steady_state_modular", no_solve)
    r = _autocrine(times)
    assert (r.rank, r.dim) == (ref.rank, ref.dim)
    assert sorted(map(tuple, (s.support for s in r.symmetries))) == \
        sorted(map(tuple, (s.support for s in ref.symmetries)))


def test_the_diagonal_start_zeroes_the_secretion_that_couples_the_conditions():
    from symident.detection.observability.start import DiagonalStart

    m = ["L = ksec*m/(V*vf) - kon*L*R + koff*C - kdL*L", "R = kR - kon*L*R + koff*C - kdR*R",
         "C = kon*L*R - koff*C - kc*C", "m = kb + ka*C - dm*m"]
    models = [[s.replace("vf", "1") for s in m], [s.replace("vf", "1.5") for s in m]]
    ds = DiagonalStart(models, [0, 1], 0, lambda ci: [], ["ksec", "kon", "koff", "kdL", "kR", "kdR", "kc", "kb",
                                                          "ka", "dm", "V"])
    assert ds.ok and ds.Z == ["ksec"] and ds.S == ["C", "L"]
    bad = DiagonalStart(models, [0, 1], 0, lambda ci: [], ["V"])
    assert not bad.ok


def test_an_event_after_the_start_leaves_the_start_at_zero():
    f = {"x": "-k*x + u*(1 - s)", "s": "0"}
    ev = [{"var": "s", "time": 5, "value": "1", "method": "replace"}]
    r = si.detect(si.Model(f, {"y": "c*x"}, trafo={"x": "0", "s": "0", "u": "1"}, events=ev))
    assert (r.rank, r.dim) == (2, 2) and r.info["rank_status"]["status"] == "proven"


def test_names_sympy_reserves_stay_coordinates_in_closed_forms():
    from symident.detection.closedform import simplify
    assert spy.simplify(model_sympify(simplify("I**2*k")) - model_sympify("I^2*k")) == 0
    assert "log(E)" in simplify("log(E)*k")
    r = detect({"C": "Ef+Eb+Ep-k*C", "C13": "Ef*rf+Eb*rb+Ep*rp-a*k*C13"}, {"CH4": "C", "ratio": "C13/C"},
               fixed=["rf", "rb", "rp", "a"], reconstruct=True)
    assert not r.info["errors"]


def test_a_state_named_e_keeps_its_closed_form():
    r = detect({"E": "b*(al*P+I+al*A)-E/De", "P": "E/De-P/Dp", "I": "r*P/Dp-I/Di", "A": "(1-r)*P/Dp-A/Di"},
               {"y": "r*P/Dp"}, fixed=["De", "Dp", "Di"], reconstruct=True)
    assert not r.info["errors"] and all("exp" not in str(s.generator) for s in r.symmetries)
    curved = [s.generator for s in r.symmetries if "E" in s.generator]
    assert curved and all(spy.simplify(model_sympify(g["E"]) / model_sympify(g["P"]) - spy.Symbol("E") /
                                       spy.Symbol("P")) == 0 for g in curved)


def test_a_premature_lie_order_is_extended():
    from symident.detection.core import _past_premature_order
    calls = []

    def oam():
        calls.append(state.get("nt_floor"))
        return {"verification": {"ok": True}}
    out = _past_premature_order(oam, {"verification": {"ok": False, "grow_at": 14}})
    assert out["verification"]["ok"] and calls == [14 + int(state.env_value("LIEPLATEAU", "3"))]


def test_an_event_value_reads_the_state_at_the_event():
    ev = [{"var": "z", "time": 1, "value": "x", "method": "replace"}]
    r = detect({"x": "-k*x", "z": "0"}, {"y": "z"}, events=ev, trafo={"z": "0"}, reconstruct=True)
    assert (r.rank, r.dim) == (1, 2)
    g = r.symmetries[0].generator
    assert spy.simplify(model_sympify(g["x"]) / model_sympify(g["k"]) - spy.Symbol("x")) == 0
    ev = [{"var": "x", "time": 1, "value": "x+c", "method": "replace"}]
    assert detect({"x": "-k*x"}, {"y": "x"}, events=ev, trafo={"x": "1"}).identifiable


def test_free_exponents_of_states_with_given_initial_values():
    assert detect({"x": "-k*x^n"}, {"y": "x"}, trafo={"x": "1"}, positive=["k", "n"]).identifiable
    assert detect({"x": "a*x^(3/4)-x/K^(1/4)"}, {"y": "x"}).identifiable


def test_a_kernel_in_log_coordinates_is_renamed_back():
    ev = [{"var": "P", "time": i, "value": str(v), "method": "replace"} for i, v in enumerate([3, 0, 8])] + \
        [{"var": "PET", "time": i, "value": str(v), "method": "replace"} for i, v in enumerate([1, 2, 1])]
    odes = {"SM": "P - P*SM^BETA/FC^BETA - PET*min(SM/(LP*FC),1)", "SUZ": "P*SM^BETA/FC^BETA - K1*SUZ",
            "P": "0", "PET": "0"}
    assert detect(odes, {"Q": "K1*SUZ"}, inputs=["P", "PET"], events=ev).identifiable


def test_a_size_cap_in_the_gap_bound_leaves_no_bound(monkeypatch):
    from symident.detection import gapcert

    def boom(*a):
        raise OverflowError("a Lie derivative grows past the size cap")
    monkeypatch.setattr(gapcert, "_gap_bound", boom)
    assert gapcert.gap_bound([], [], [], 1, 7, None)["bound"] is None


def test_the_combined_lift_solves_the_square_subsystem():
    from symident.detection.fields import _square_solve
    a, b = spy.symbols("a b")
    A = spy.Matrix([[a, 1], [2 * a, 2], [1, b]])
    c = _square_solve(A, A * spy.Matrix([b, a]))
    assert [spy.cancel(x) for x in c] == [b, a]
    from symident.sym.direction import _canon_int_vector
    assert _canon_int_vector([spy.E * a], [a]) is None


def test_a_signed_net_uptake_rests_at_zero():
    rx = network(("", "X", "ki*(lev*u - X)"), ("X", "Y", "kf*X"), ("", "Y", "kb"), ("Y", "", "kd*Y"))
    rx = dict(rx, species=rx["species"] + ["u"], stoichiometry=[r + [0] for r in rx["stoichiometry"]])
    res = detect(reactions=rx, observables={"y": "s*Y"}, inputs=["u"], events=[event("u", 0, "1")],
                 equilibrate=True)
    ex = detect({"X": "ki*(lev*u - X)", "Y": "kf*X + kb - kd*Y", "u": "0"}, {"y": "s*Y"}, inputs=["u"],
                events=[event("u", 0, "1")], trafo={"X": "0", "Y": "kb/kd"})
    assert res.rank == ex.rank


def test_a_known_input_is_a_generic_known_function():
    r = detect({"x": "-k*x + b*T"}, {"y": "x"}, known_inputs=["T"], trafo={"x": "0"})
    assert r.identifiable and set(r.info["coordinates"]) == {"k", "b"}
    assert r.info["timing"][-1]["stage"] == "done"


def test_a_repeated_reduction_is_a_copy_of_the_first(monkeypatch):
    r = detect({"x": "-k*x"}, {"y": "s*x"}, reconstruct=True)
    calls, run = [], si.api.reduce
    monkeypatch.setattr(si.api, "reduce", lambda *a, **k: calls.append(1) or run(*a, **k))
    a, b = r.reduce(), r.reduce()
    assert a is not b and a.to_dict() == b.to_dict() and len(calls) == 1


def test_exponentials_of_parameters_outside_the_resting_model_under_equilibrate():
    ev = [event("u", 1, "exp(5*(1-c)/c)")]
    eq = detect({"x": "b - k*x + u", "u": "0"}, {"y": "s*x"}, inputs=["u"], events=ev, equilibrate=True,
                reconstruct=True)
    ex = detect({"x": "b - k*x + u", "u": "0"}, {"y": "s*x"}, inputs=["u"], events=ev, trafo={"x": "b/k"},
                reconstruct=True)
    assert (eq.rank, eq.dim) == (ex.rank, ex.dim) and "_ew" not in str(eq.info["coordinates"])


def test_a_state_idle_at_rest_keeps_a_resting_value():
    r = detect({"x": "b - k*x + u", "z": "kz*u", "u": "0"}, {"y": "s*x + z"}, inputs=["u"],
               events=[event("u", 0, "1")], equilibrate=True)
    assert "z" in r.info["coordinates"]


@needs_msolve
def test_a_discrete_symmetry_is_counted():
    r = detect({"x1": "x2", "x2": "-(p+q)*x2 - p*q*x1"}, {"y": "x1"}, trafo={"x1": "1", "x2": "0"})
    assert r.identifiable and r.solution_count()["count"] == 2


@needs_msolve
def test_the_count_reads_the_clock_and_known_inputs():
    r = detect({"x": "p*time + q"}, {"y": "x"}, trafo={"x": "0"})
    assert r.identifiable and r.solution_count()["count"] == 1
    r = detect({"x": "b*u - c*u^2"}, {"y": "x"}, trafo={"x": "0"}, known_inputs=["u"])
    assert r.identifiable and r.solution_count()["count"] == 1


def test_the_count_refuses_a_resting_start():
    r = detect({"x": "b - x"}, {"y": "x"}, equilibrate=True)
    with pytest.raises(ValueError, match="equilibrate"):
        r.solution_count()


def test_a_missing_msolve_names_its_installation(monkeypatch):
    monkeypatch.setattr("symident.sym.msolve.find_msolve", lambda: (None, None))
    r = detect({"x1": "x2", "x2": "-(p+q)*x2 - p*q*x1"}, {"y": "x1"}, trafo={"x1": "1", "x2": "0"})
    with pytest.raises(MsolveMissingError, match="install_msolve"):
        r.solution_count()


def test_a_spike_train_with_state_dependent_events():
    ev = []
    for t in [f"{k}/10" for k in range(5, 20)]:
        ev += [event("u", t, "u+U*(1-u)"), event("Is", f"{t}+1/1000", "Is+A*u*x"),
               event("x", f"{t}+2/1000", "x-u*x")]
    r = detect({"x": "(1-x)/tD", "u": "(U-u)/tF", "Is": "-Is/ts"}, {"y": "Is"}, events=ev,
               trafo={"x": "1", "u": "0", "Is": "0"})
    assert r.identifiable


def brief(r):
    return r.rank, r.dim, r.info["coordinates"], sorted((s.type, s.level) for s in r.symmetries)


def test_a_compound_replacement_binds_as_one_operand():
    from symident.rexpr import replace_symbols
    assert replace_symbols(["k"], ["a+b"], "-k*x") == "-(a+b)*x"
    assert replace_symbols(["k"], ["-1"], "k") == "-1"
    assert replace_symbols(["k"], ["exp(l)"], "-k*x") == "-exp(l)*x"


@pytest.mark.parametrize("kw", [
    dict(odes={"x": "-k*x"}),
    dict(odes={"x": "-c*x"}, trafo={"x": "1"}, conditions=grid(["p", "q"], c=["1", "c*k"])),
    dict(odes={"x": "-c*x"}, trafo={"x": "1"}, events=[event("x", "1", "c*k", "add")]),
    dict(odes={"x": "-c*x"}, trafo=[{"x": "1"}, {"x": "2"}], events=[event("x", "1", "c*k", "add")]),
    dict(odes={"x": "u - k*x", "u": "0"}, trafo=[{}, {}], inputs=["u"], events=[event("u", "0", "1")],
         equilibrate=True),
])
def test_a_compound_trafo_value_equals_its_parenthesized_form(kw):
    def run(v):
        tr = kw.get("trafo")
        tr = [dict(t, k=v) for t in tr] if isinstance(tr, list) else dict(tr or {}, k=v)
        return detect(observables={"y": "x"}, **dict(kw, trafo=tr))
    assert brief(run("a+b")) == brief(run("(a+b)"))


@pytest.mark.parametrize("clock, plain", [
    (dict(trafo={"x": "a + time"}), dict(trafo={"x": "a"})),
    (dict(trafo={"x": "a + time"}, events=[event("x", "-1", "c", "add")]),
     dict(trafo={"x": "a - 1"}, events=[event("x", "-1", "c", "add")])),
    (dict(trafo={"x": "1"}, events=[event("x", "2", "c*time", "add")]),
     dict(trafo={"x": "1"}, events=[event("x", "2", "2*c", "add")])),
    (dict(trafo={"k": "a*time"}), dict(odes={"x": "-a*time*x"})),
    (dict(trafo=[{"x": "1", "k": "a*time"}, {"x": "2", "k": "a*time"}]),
     dict(odes={"x": "-a*time*x"}, trafo=[{"x": "1"}, {"x": "2"}])),
])
def test_time_outside_the_right_hand_sides_is_the_clock(clock, plain):
    def run(kw):
        return detect(**{"odes": {"x": "-k*x"}, "observables": {"y": "x"}, **kw}, reconstruct=True)
    r = run(clock)
    assert brief(r) == brief(run(plain)) and "time" not in r.info["coordinates"]


def test_the_clock_under_equilibrate_is_an_input_from_the_start():
    f = {"x": "kb + kin*u - k*x", "u": "0"}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rest = detect(f, {"y": "x + c*time"}, inputs=["u"], events=[event("u", "0", "1")], equilibrate=True)
    told = detect(f, {"y": "x + c*time"}, trafo={"x": "kb/k", "u": "1"})
    assert brief(rest) == brief(told)
    # a start before zero: the resting state of the model at that time
    f = {"x": "kb + c*time + kin*u - k*x", "u": "0"}
    rest = detect(f, {"y": "x"}, inputs=["u"], events=[event("u", "-1", "1")], equilibrate=True)
    told = detect(f, {"y": "x"}, trafo={"x": "(kb - c)/k"}, events=[event("u", "-1", "1")])
    assert brief(rest) == brief(told)
    with pytest.raises(ValueError, match="clock"):
        detect({"x": "kb + kin*u*exp(-a*time) - k*x", "u": "0"}, {"y": "x"}, inputs=["u"],
               events=[event("u", "0", "1")], equilibrate=True)


def _scaling_rows(r):
    """The weights of the scalings of `r` over its coordinates, a zero row without one."""
    c = r.info["coordinates"]
    return spy.Matrix([[int(s.weights.get(z, 0)) for z in c] for s in r.symmetries if s.type == "scaling"] or
                      [[0] * len(c)])


@pytest.mark.parametrize("odes, obs, kw", [
    ({"x": "-d*x"}, {"y": "s*x"}, dict(trafo={"x": "1"})),
    ({"x": "-d*x"}, {"y": "s*x"}, dict(events=[event("x", "0", "1")])),
    ({"x": "-d*x"}, {"y": "s*x"}, dict(conditions=grid(["c1"], s=[2]))),
    ({"x": "-d*x"}, {"y": "s*x"}, dict(conditions=grid(["c1", "c2"], s=[2, 3]))),
    ({"x": "-k*x + b*u"}, {"y": "x"}, dict(known_inputs=["u"])),
    ({"x": "-d*x"}, {"y": "s*x"}, dict(trafo=[{"x": "1"}, {"x": "2"}])),
    ({"x": "-d*x"}, {"y": "s*x"}, dict(events=[event("x", "1", "c", "add")])),
    ({"x": "-d*x"}, {"y": "s*x"}, dict(events=[event("x", "tau", "c", "add")])),
    ({"x": "kb + kin*u - k*x", "u": "0"}, {"y": "s*x"}, dict(inputs=["u"], events=[event("u", "0", "dose")],
                                                           equilibrate=True)),
])
def test_scalings_only_holds_the_values_the_detection_holds(odes, obs, kw):
    full = detect(odes, obs, reconstruct=True, **kw)
    sc = detect(odes, obs, scalings_only=True, **kw)
    assert sc.info["coordinates"] == full.info["coordinates"]
    a, b = _scaling_rows(sc), _scaling_rows(full)
    assert a.rank() == b.rank() == spy.Matrix.vstack(a, b).rank()


@pytest.mark.parametrize("kw", [
    dict(conditions=grid(["a", "b"], tau=[0, 1])),
    dict(trafo=[{"x": "1", "tau": "0"}, {"x": "1", "tau": "1"}]),
])
def test_an_event_time_resolved_per_condition(kw):
    ev = [event("x", "tau", "c", "add")]
    kw = dict({"trafo": {"x": "1"}}, **kw)
    r = detect({"x": "-k*x"}, {"y": "x"}, events=ev, **kw)
    ref = detect({"x": "-k*x"}, {"y": "x"}, trafo=[{"x": "1"}, {"x": "1"}],
                 events=[dict(ev[0], time="0", cond=[0]), dict(ev[0], time="1", cond=[1])])
    assert brief(r) == brief(ref)
    one = detect({"x": "-k*x"}, {"y": "x"}, events=ev, trafo={"x": "1"}, conditions=grid(["a"], tau=[0]))
    assert brief(one) == brief(detect({"x": "-k*x"}, {"y": "x"}, events=ev, trafo={"x": "1", "tau": "0"}))


def test_the_segments_of_a_condition_follow_its_resolved_event_times():
    from symident.detection.conditions import Grid, as_events, resolve_conditions
    res = resolve_conditions(Grid(["a", "b"], {"tau": [0, 1]}), as_events([event("x", "tau", "c", "add")]),
                             {"x": "1"}, ["x", "k", "c", "tau"], ["x"])
    assert res["chain_of"] == [1, 2, 2] and res["times"] == ["0", "0", "1"]
    assert res["ic0"][0] == {"x": "(1) + (c)"} and res["seg_events"][2] == [
        {"var": "x", "method": "add", "value": "c"}]


def test_reduce_cq_reads_the_totals_of_an_ode_model():
    ev = [event("u", "0", "1")]
    ode = si.Model({"A": "-k1*A*(1+u) + k2*B", "B": "k1*A*(1+u) - k2*B", "u": "0"}, {"y": "s*B"}, inputs=["u"],
                   events=ev, totals={"T": "A + B"})
    rx = si.Model(reactions={"species": ["A", "B", "u"], "stoichiometry": [[-1, 1, 0], [1, -1, 0]],
                             "rates": ["k1*A*(1+u)", "k2*B"]}, observables={"y": "s*B"}, inputs=["u"], events=ev,
                  totals={"T": "A + B"})
    for eq in (False, True):
        a, b = si.detect(ode, equilibrate=eq, reduce_cq=True), si.detect(rx, equilibrate=eq, reduce_cq=True)
        assert brief(a) == brief(b) and "T" in a.info["coordinates"]
    leak = si.Model({"A": "-k1*A + k2*B", "B": "k1*A - k2*B - d*B"}, {"y": "s*B"}, totals={"T": "A + B"})
    with pytest.raises(ValueError, match="T = A \\+ B is not conserved"):
        si.detect(leak, reduce_cq=True)


@pytest.mark.parametrize("kw, name", [
    (dict(known_inputs=["U"]), "U"),
    (dict(known_inputs=["x"]), "x"),
    (dict(inputs=["U"]), "U"),
    (dict(events=[event("X", "1", "c", "add")]), "X"),
    (dict(conditions=grid(["lo", "hi"], Dose=[1, 10])), "Dose"),
    (dict(fixed=["S"]), "S"),
    (dict(parameters=["p"]), "p"),
])
def test_a_declared_name_outside_the_model_is_refused(kw, name):
    with pytest.raises(ValueError, match=f"\\b{name}\\b"):
        detect({"x": "-k*x + b*u"}, {"y": "s*x"}, **kw)


def test_names_in_the_trafo_events_cells_and_totals_are_declared():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        detect({"x": "-k*x"}, {"y": "x"}, trafo={"k": "exp(lk)", "x": "x0"}, fixed=["x0"], positive=["lk"])
        detect({"x": "-k*x"}, {"y": "x"}, events=[event("x", "1", "c", "add")], fixed=["c"])
        detect({"x": "-k*x"}, {"y": "s*x"}, conditions=grid(["a", "b"], s=["s1", "s2"]), fixed=["s1"])
        detect(reactions=canonical(), observables={"y": "A"}, reduce_cq=True, fixed=["total1"])
    with pytest.warns(UserWarning, match="`positive` names K"):
        detect({"x": "-k*x"}, {"y": "x"}, positive=["k", "K", "1 - x"])


@pytest.mark.parametrize("kw", [
    dict(trafo={"x": "0", "u": "w"}),
    dict(trafo=[{"x": "0", "u": "1"}, {"x": "0", "u": "2"}]),
    dict(trafo={"x": "0"}, conditions=grid(["a", "b"], u=[1, 2])),
])
def test_an_input_takes_no_initial_value(kw):
    with pytest.raises(ValueError, match="input u starts at zero"):
        detect({"x": "-k*x + b*u", "u": "-d*u"}, {"y": "x"}, inputs=["u"], **kw)


@pytest.mark.parametrize("start", [{}, {"u": "0"}, {"u": "u"}])
def test_an_event_at_the_start_sets_an_input(start):
    f = {"x": "-k*x + b*u", "u": "-d*u"}
    r = detect(f, {"y": "x"}, inputs=["u"], trafo={"x": "0", **start}, events=[event("u", "0", "w")])
    assert brief(r) == brief(detect(f, {"y": "x"}, trafo={"x": "0", "u": "w"}))


def test_a_rate_set_to_zero_by_the_trafo_rests_like_a_constant_state():
    ev = [event("u", "0", "1")]
    zero = detect({"z": "-k*u*z", "w": "-j*u*w", "u": "0"}, {"y": "z*w"}, inputs=["u"], events=ev,
                  trafo={"k": "0"}, equilibrate=True)
    idle = detect({"z": "0", "w": "-j*u*w", "u": "0"}, {"y": "z*w"}, inputs=["u"], events=ev, equilibrate=True)
    assert brief(zero) == brief(idle)
    r = detect({"z": "-k*u*z", "w": "-j*u*w", "u": "0"}, {"y": "z*w"}, inputs=["u"], events=ev, equilibrate=True,
               reconstruct=True)
    assert si.reduce(r, allow_checked=True).checks["loop"]["ok"] is not None


@needs_msolve
def test_the_count_substitutes_the_trafo_into_the_initial_values():
    chained = detect({"x": "-k^2*x"}, {"y": "x"}, trafo={"x": "k", "k": "a"})
    written = detect({"x": "-a^2*x"}, {"y": "x"}, trafo={"x": "a"})
    assert chained.solution_count()["count"] == written.solution_count()["count"] == 1
    assert detect({"x": "-b*x"}, {"y": "x"}, trafo={"x": "k", "k": "a^2"}).solution_count()["count"] == 2
    one = detect({"x": "-k^2*x"}, [{"y": "s*x"}], trafo=[{"x": "k", "k": "a"}], conditions=grid(["c"], s=[1]))
    assert one.solution_count()["count"] == 1


def test_the_count_of_a_model_that_is_not_rational_is_none():
    for f, tr in [({"x1": "x2", "x2": "-(p+q)*x2 - p*q*abs(x1)"}, {"x1": "1", "x2": "0"}),
                  ({"x": "-k*x^n"}, {"x": "2"})]:
        r = detect(f, {"y": "x1" if "x1" in f else "x"}, trafo=tr)
        assert r.identifiable and r.solution_count() is None
    r = detect({"x1": "x2", "x2": "-(p+q)*x2 - p*q*x1 + piecewise(p, time > 1, 0)"}, {"y": "x1"},
               trafo={"x1": "1", "x2": "0"})
    with pytest.raises(ValueError, match="switches"):
        r.solution_count()


@needs_msolve
def test_the_count_reads_the_model_as_detected():
    m = si.Model({"x1": "x2", "x2": "-(p+q)*x2 - p*q*x1"}, {"y": "x1"}, trafo={"x1": "1", "x2": "0"})
    r = si.detect(m)
    m.observables = {"y": "x1", "y2": "p*x2"}
    assert r.solution_count()["count"] == 2

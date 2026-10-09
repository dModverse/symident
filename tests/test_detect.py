"""Behaviour of detect(): ranks, verdicts and directions on small models."""

import re
import time
import warnings

import numpy as np
import pytest
import sympy as spy

import symident as si
from symident import state
from symident.sym.msolve import _msolve_binary


def _has_msolve():
    try:
        _msolve_binary()
    except Exception:
        return False
    return True


needs_msolve = pytest.mark.skipif(not _has_msolve(), reason="needs msolve")

_MODEL_KEYS = ("reactions", "trafo", "conditions", "events", "inputs", "fixed", "parameters")


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
    cond = grid(["stim"], var=[1])
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


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
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
    t0 = time.monotonic()
    sres = detect(reactions=hill_michaelis(), observables={"xobs": "scale * x"}, engine="symbolic",
                  trafo=ss)
    assert time.monotonic() - t0 < 120
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
    assert st.info["rank_proof"] == "invariant fields"
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


@pytest.mark.slow
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


@pytest.mark.slow
def test_small_network_at_rest_finishes_quickly():
    rx = network(("A", "B", "k1*A"), ("B", "A", "k2*B"), ("B", "C", "k3*B"), ("C", "B", "k4*C"))
    t0 = time.monotonic()
    r = detect(reactions=rx, observables={"y": "B"}, equilibrate=True, reduce_cq=True)
    assert (r.rank, r.dim) == (1, 5)
    assert time.monotonic() - t0 < 40


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


@pytest.mark.slow
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

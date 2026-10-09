"""Behaviour of the reduction: invariants, sections, charts and zero limits.

Assertions are mathematical: invariants are checked by X(I) = 0, charts by evaluating
them, reparametrisations by detecting again with the emitted trafo.
"""

import math
import re
import time
import warnings

import pytest
import sympy as spy

import symident as si

_FUNS = {"exp": math.exp, "log": math.log, "sqrt": math.sqrt, "atan": math.atan, "abs": abs,
         "exp10": lambda x: 10.0 ** x, "log10": math.log10, "log1p": math.log1p}
_IDENT = re.compile(r"[A-Za-z_.][A-Za-z0-9_.]*")


def det(f, g, *, trafo=None, fixed=(), reactions=None, **kw):
    kw.setdefault("cores", 1)
    model = si.Model(f, g, reactions=reactions, trafo=trafo, fixed=fixed)
    return si.detect(model, **kw)


def identifiable(f, g, **kw):
    return det(f, g, **kw).identifiable


def red(obj, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return si.reduce(obj, **kw)


def mkdir(gen, type="polynomial", weights=None):
    # a synthetic direction with an explicit generator
    return {"type": type, "generator": gen, "weights": weights, "support": list(gen), "explicit": True}


def mkobj(dirs, coords):
    return {"identifiable": False, "rank": 1, "dim": len(coords), "symmetries": dirs,
            "info": {"engine": "modular", "settings": {}, "coordinates": list(coords)}}


def network(*reactions):
    # reactions as (educts, products, rate) with educts and products as "A + B"
    species = []
    for lhs, rhs, _ in reactions:
        for s in (lhs + " + " + rhs).split("+"):
            if s.strip() and s.strip() not in species:
                species.append(s.strip())
    S = [[0] * len(species) for _ in reactions]
    for i, (lhs, rhs, _) in enumerate(reactions):
        for s in lhs.split("+"):
            S[i][species.index(s.strip())] -= 1
        for s in rhs.split("+"):
            S[i][species.index(s.strip())] += 1
    return {"species": species, "stoichiometry": S, "rates": [r for _, _, r in reactions]}


def ev(expr, pt):
    # numeric value of an R expression; comparisons give a bool
    return eval(str(expr).replace("^", "**"), {"__builtins__": {}}, {**_FUNS, **pt})


def variables(expr):
    return [m.group(0) for m in _IDENT.finditer(str(expr))
            if not str(expr)[m.end():].lstrip().startswith("(")]


def expr_equal(a, b):
    d = spy.sympify(f"({a}) - ({b})".replace("^", "**"))
    return spy.simplify(d) == 0


def lie_at(gen, inv, pt, eps=1e-7):
    # X(I) at a point, by forward differences of I
    f0 = ev(inv, pt)
    out = 0.0
    for v, xi in gen.items():
        zp = dict(pt)
        zp[v] += eps
        out += ev(xi, pt) * (ev(inv, zp) - f0) / eps
    return out


def replace_symbols(expr, table):
    return re.sub(r"(?<![A-Za-z0-9_.])[A-Za-z_.][A-Za-z0-9_.]*(?![A-Za-z0-9_.])",
                  lambda m: table.get(m.group(0), m.group(0)), expr)


def compose(tr, trafo):
    # the outer trafo substituted into the inner one
    table = {k: f"({v})" for k, v in trafo.items()}
    return {k: replace_symbols(v, table) for k, v in tr.items()}


def blocks_of(r, type):
    return [b for b in r.blocks if b["type"] == type]


def lines(text):
    return str(text).split("\n")


def grep(pattern, out):
    return any(re.search(pattern, s) for s in out)


def zero_rows(r):
    return {row["coordinates"]: row for row in r.to_dict()["zero_compatibility"]}


F_MRNA = {"m": "ktx - dm*m", "p": "ktl*m - dp*p"}
F_AB = {"A": "-k1*A + k2*B", "B": "k1*A - k2*B"}
F_OHM = {"V": "(gL*(EL - V) + gt*(Es - V))/C"}


def test_one_scaling_pin_family_identifiable():
    res = det(F_MRNA, {"y": "p"}, reconstruct=True)
    r = red(res)
    b = r.blocks[0]
    assert b["type"] == "scaling"
    assert b["status"] == "reduced"
    # the invariant lattice is spanned by ktx*ktl and ktx/m
    assert any(expr_equal(iv, "ktx*ktl") or expr_equal(iv, "1/(ktx*ktl)") for iv in b["invariants"])
    # every listed transversal is one coordinate of the support
    adm = r.to_dict()["family"][0]["admissible"]
    assert all(t in b["support"] for T in adm for t in T)
    assert identifiable(F_MRNA, {"y": "p"}, trafo=r.trafo)


def test_fixed_rank_verdict_redundancy_unknown_names():
    W = {"x": "1", "y": "-1", "z": "-1"}
    obj = mkobj([mkdir({"x": "x", "y": "-y", "z": "-z"}, "scaling", W)], ["x", "y", "z"])
    with pytest.warns(UserWarning, match="no effect"):
        si.reduce(obj, fixed=["nothere"])
    r = red(obj, fixed=["x", "y"])
    b = r.blocks[0]
    assert b["status"] == "fixed"
    assert b["removed_by_fixed"] == 1
    assert b["redundant_fixed"] == ["y"]
    # fixed coordinates are the user's: the trafo is the identity
    assert all(v == k for k, v in r.trafo.items())


def test_scaling_overlapping_curved_direction_gauged_outside_support():
    g = {"y": "alpha*A"}
    res = det(F_AB, g, reconstruct=True)
    # with the search caps off the curved block stays open; the grouping is the same
    split = red(res, d_poly=0, d_darboux=0, d_exp=0)
    types = [b["type"] for b in split.blocks]
    assert sorted(types) == ["curved", "scaling"]
    sb = blocks_of(split, "scaling")[0]
    cb = blocks_of(split, "curved")[0]
    assert sb["status"] == "reduced"
    assert not any(p in cb["support"] for p in sb["pins"])
    assert len(cb["labels"]) == 1
    # with the default caps both reduce and the reduced model is identifiable
    r0 = red(res)
    assert not r0.remaining
    assert identifiable(F_AB, g, trafo=r0.trafo)
    # a fixed readout leaves one curved direction, which reduces end-to-end
    r = red(res, fixed=["alpha"])
    assert {b["status"] for b in r.blocks} == {"fixed", "reduced"}
    cb = next(b for b in r.blocks if b["status"] == "reduced")
    assert len(cb["invariants"]) == 2
    assert identifiable(F_AB, g, fixed=["alpha"], trafo=r.trafo)


def test_darboux_rational_invariant_and_degree_cap_certificate():
    obj = mkobj([mkdir({"x": "x**2", "y": "y**2"})], ["x", "y"])
    # quadrature and rational stages off, so the factor stages answer on their own
    b0 = red(obj, d_poly=0, d_darboux=0, d_exp=0, separable=False).blocks[0]
    assert b0["status"] == "unresolved"
    assert any("coordinate and xi factors only" in c for c in b0["certificates"])
    # the exp stage resolves the block through exp(1/x - 1/y)
    b_e = red(obj, d_poly=0, d_darboux=0, separable=False).blocks[0]
    assert b_e["stage"] == "exp"
    assert len(b_e["invariants"]) > 0
    assert any("exp(" in iv for iv in b_e["invariants"])
    # cap 1: the extactic factors x, y, x - y give (x - y)/(x*y)
    b1 = red(obj, d_poly=0, d_darboux=1, separable=False).blocks[0]
    assert b1["stage"] == "darboux"
    assert len(b1["invariants"]) == 1
    assert abs(lie_at({"x": "x^2", "y": "y^2"}, b1["invariants"][0], {"x": 1.31, "y": 0.77})) < 1e-4


def test_module_reduction_collapses_pair():
    X12 = {"C": "Ci*rr + Ci", "Ci": "-(Ci*rr - Ci)", "k2i": "-(k2*rr - k2 + k2i*rr + k2i)",
           "r1": "r1*rr + r1", "r2": "r2*rr + r2 + rr"}
    X13 = {"C": "Ci*r2*rr + Ci*r2", "Ci": "-(Ci*r2*rr - Ci*r2)",
           "k2i": "-(k2*r2*rr - k2*r2 + k2i*r2*rr + k2i*r2)", "r1": "-r1*rr",
           "rr": "-(r2*rr**3 - 2*r2*rr**2 - r2*rr - rr**3 - rr**2)"}
    obj = mkobj([mkdir(X12), mkdir(X13)], ["C", "Ci", "k2", "k2i", "r1", "r2", "rr"])
    b = red(obj).blocks[0]
    # the combination (-r2)*X1 + X2 lives on 3 coordinates
    assert any("support 3" in c for c in b["module_combos"])


def test_multi_generator_block_common_invariants_only():
    # X1 scales (a, b), X2 scales (b, c): only a*b*c is invariant under both
    obj = mkobj([mkdir({"a": "a", "b": "-b"}, "general"), mkdir({"b": "b", "c": "-c"}, "general")],
                ["a", "b", "c"])
    b = red(obj).blocks[0]
    assert len(b["invariants"]) == 1
    assert expr_equal(b["invariants"][0], "a*b*c") or expr_equal(b["invariants"][0], "1/(a*b*c)")


def test_support_only_directions_and_missing_coordinates():
    d = {"type": "general", "generator": None, "weights": None, "support": ["p", "q"], "explicit": False}
    obj = mkobj([d], ["p", "q", "r"])
    b = red(obj).blocks[0]
    assert b["status"] == "unresolved"
    assert "reconstruct=True" in b["reason"]
    # without a coordinate list the support union is used, with a warning
    del obj["info"]["coordinates"]
    with pytest.warns(UserWarning, match="no coordinate list"):
        r2 = si.reduce(obj)
    assert set(r2.trafo) == {"p", "q"}


def test_trafo_entries_parse_with_integer_powers():
    res = det(F_AB, {"y": "alpha*A"}, reconstruct=True)
    r = red(res, fixed=["alpha"])
    assert isinstance(r.trafo, dict)
    assert set(r.trafo) == set(r.coordinates)
    for v in r.trafo.values():
        spy.sympify(v.replace("^", "**"))
        assert not re.search(r"\^\s*\(?\s*[0-9]*\.[0-9]", v)


def test_identifiable_result_reduces_to_nothing():
    res = det({"x": "-k*x"}, {"y": "x"})
    r = red(res)
    assert r.blocks == []
    assert r.trafo is None
    assert "Nothing to reduce" in repr(r)


def test_print_lean_summary_carries_block_report():
    res = det(F_MRNA, {"y": "p"}, reconstruct=True)
    r = red(res)
    out = lines(repr(r))
    assert grep(r"^Reduced 1 of 1 direction", out)
    assert grep("Trafo", out)
    # the monomial of a scaling survivor is not printed; it stays on the object
    assert not grep(r"ktx\*ktl", out)
    assert any("ktx*ktl" in v for v in r.blocks[0]["survivor_meaning"].values())
    assert not grep(r"Scaling block|transversal:|admissible|\[", out)
    assert len(out) < 12
    # one line per block; a reduced block repeats no invariants and no certificate
    rep = lines(r.summary())
    assert grep(r"^Blocks$", rep)
    assert grep(r"\{X[^}]+\} scaling, reduced \| transversal ktl = 1", rep)
    assert grep("admissible", rep)
    assert not grep(r"ktx\*ktl", rep)
    assert not grep(r"invariants  |certificate|certified", rep)
    assert len(rep) < 20
    # verbose adds the raw invariants, still no certificates
    vrb = lines(r.summary(verbose=True))
    assert grep("invariants  ", vrb)
    assert not grep(r"\[transversal certified|\[verified", vrb)
    assert len(vrb) > len(rep)
    assert grep(r"admissible  \{", vrb)


@pytest.mark.slow
def test_egf_mek_erk_cascade_reduced_end_to_end():
    reactions = network(("EGF + EGFR", "EGF_EGFR", "k_bind * EGF * EGFR"),
                        ("EGF_EGFR", "EGF + EGFR", "k_unbind * EGF_EGFR"),
                        ("MEK", "pMEK", "k_phos_MEK * EGF_EGFR * MEK"),
                        ("pMEK", "MEK", "k_dephos_MEK * pMEK"),
                        ("ERK", "pERK", "k_phos_ERK * pMEK * ERK"),
                        ("pERK", "ERK", "k_dephos_ERK * pERK"))
    obs = {"pMEK_obs": "scale_pMEK * pMEK", "pERK_obs": "scale_pERK * pERK"}
    egf = det(None, obs, reactions=reactions, reduce_cq=False, reconstruct=True)
    r = red(egf)
    # three unit scalings and the rational confounder alone in its curved block
    assert sorted(b["type"] for b in r.blocks) == ["curved", "scaling", "scaling", "scaling"]
    cb = blocks_of(r, "curved")[0]
    assert len(cb["labels"]) == 1
    assert len(cb["invariants"]) == 2
    # on the section EGF_EGFR = 1 the confounder invariant reads (EGF + 1)*(EGFR + 1)
    assert any(expr_equal(iv, "EGF*EGFR + EGF + EGFR") for iv in cb["invariants"])
    # a positive chart requires an offset
    assert cb["status"] == "reduced"
    assert not r.remaining
    assert any("carrier offset" in c for c in cb["certificates"])
    # the chart stays positive for every positive outer value
    vals = [0.7, 2.3, 1.9, 0.31, 1.7, 0.45, 1.1, 0.9]
    outer = list(dict.fromkeys(r.coordinates + [v for e in r.trafo.values() for v in variables(e)]))

    def u_at(scale):
        return {nm: scale * vals[i % len(vals)] for i, nm in enumerate(outer)}

    def chart(u):
        return {nm: ev(e, u) for nm, e in r.trafo.items()}

    for scale in (1e-3, 1, 1e3):
        z = chart(u_at(scale))
        assert all(math.isfinite(v) and v > 0 for v in z.values())
    # survivor_meaning at the chart point returns the outer value
    u = u_at(1)
    z = chart(u)
    for nm, m in cb["survivor_meaning"].items():
        assert ev(m, z) == pytest.approx(u[nm], rel=1e-8)
    # the invariants behind the new parameters are positive at arbitrary inner points
    for z0 in ({"EGF": 1.3, "EGFR": 0.2, "EGF_EGFR": 2.5, "k_bind": 0.8, "k_phos_MEK": 0.3, "k_unbind": 1.9},
               {"EGF": 1e-4, "EGFR": 1e-4, "EGF_EGFR": 50, "k_bind": 3, "k_phos_MEK": 2, "k_unbind": 1e-3}):
        for m in cb["survivor_meaning"].values():
            assert ev(m, z0) > 0


def test_exponential_factor_invariant_solved_and_reported():
    gen = {"x": "x", "y": "x + y"}
    obj = mkobj([mkdir(gen)], ["x", "y"])
    r = red(obj)
    b = r.blocks[0]
    assert b["stage"] == "exp"
    assert b["status"] == "reduced"
    assert len(b["invariants"]) == 1
    assert abs(lie_at(gen, b["invariants"][0], {"x": 1.7, "y": 0.9})) < 1e-4
    # I(trafo(u)) is u or 1/u
    assert list(b["survivor_meaning"]) == ["q_1"]
    z = {nm: ev(e, {"x": 5, "y": 3, "q_1": 3}) for nm, e in r.trafo.items()}
    Iv = ev(b["invariants"][0], z)
    assert abs(Iv - 3) < 1e-10 or abs(Iv - 1 / 3) < 1e-10
    assert any("exp stage" in c for c in b["certificates"])
    # d_exp = 0 switches the stage off with a certificate
    b0 = red(obj, d_exp=0).blocks[0]
    assert b0["status"] == "unresolved"
    assert any("exp stage skipped" in c for c in b0["certificates"])
    txt = lines(r.summary(verbose=True))
    assert grep(r"\[exp\]", txt)
    assert grep("gauge pin", txt)


def test_section_from_product_of_two_summand_balances():
    f = {"m": "ktx - dm*m", "P": "-k_p*s*ktx*P + k_d*pP", "pP": "k_p*s*ktx*P - k_d*pP"}
    g = {"yA": "s*m", "yB": "pP"}
    r = red(det(f, g, reconstruct=True))
    assert not r.remaining
    curved = blocks_of(r, "curved")[0]
    assert curved["status"] == "reduced"
    assert "certified positive" in curved["gauge_note"]
    # the pinned scaling coordinate is out of the ansatz and of the block report
    assert not any("ktx" in s for s in curved["invariants"] + (curved.get("section") or []))
    z = {nm: ev(e, {"q_1": 2.3, "q_2": 0.7, "pP": 1.4, "s": 0.9, "dm": 1.1, "m": 0.5})
         for nm, e in r.trafo.items()}
    assert all(z[k] > 0 for k in ("P", "k_p", "k_d"))
    assert identifiable(f, g, trafo=r.trafo)


def test_invariants_verified_against_other_blocks():
    f = {"P1": "-k1*P1 + k2*pP1", "pP1": "k1*P1 - k2*pP1",
         "P2": "-k3*P2 + k4*pP2", "pP2": "k3*P2 - k4*pP2"}
    g = {"y1": "pP1", "y2": "pP2"}
    r = red(det(f, g, reconstruct=True))
    curved = blocks_of(r, "curved")
    assert len(curved) == 2
    assert all(any("every other block" in c for c in b["certificates"]) for b in curved)
    assert identifiable(f, g, trafo=r.trafo)


@pytest.mark.slow
def test_positive_declares_certificate_domain():
    f = {"P": "-k_p*P + k_d*pP", "pP": "k_p*P - k_d*pP"}
    g = {"y": "s*pP"}
    res = det(f, g, reconstruct=True)

    def curved(r):
        return next(b for b in r.blocks if b["type"] == "curved" and b.get("invariants"))

    full = red(res, fixed=["s"])
    assert not full.remaining
    assert curved(full)["status"] == "reduced"
    none = red(res, fixed=["s"], positive=False)
    assert none.remaining
    assert curved(none)["status"] == "invariant_only"
    assert "declared domain" in curved(none)["reason"]
    # the invariants are unchanged up to their form, only the chart is gone
    assert len(curved(none)["invariants"]) == len(curved(full)["invariants"])
    assert all(any(expr_equal(iv, jv) for jv in curved(full)["invariants"]) for iv in curved(none)["invariants"])
    # k_d left real holds the real invariant k_p + k_d; the chart still holds
    short = red(res, fixed=["s"], positive=["P", "pP", "k_p"])
    assert curved(short)["status"] == "reduced"
    assert "real" in curved(short)["carrier_domain"].values()
    assert identifiable(f, g, trafo=short.trafo, fixed=["s"])
    with pytest.warns(UserWarning, match="not a coordinate"):
        si.reduce(res, positive=["P", "nonesuch"])


def test_root_carrier_quadratic_invariant():
    # the rotation has the invariant a^2 + b^2; the balance section a = b meets every circle
    obj = mkobj([mkdir({"a": "b", "b": "-a"})], ["a", "b"])
    r = red(obj)
    b = r.blocks[0]
    assert b["status"] == "reduced"
    assert "gauge section" in b["gauge_note"]
    assert "certified positive" in b["gauge_note"]
    assert all(re.search(r"sqrt|\^\(1/2\)", e) for e in r.trafo.values())
    z = {nm: ev(e, {"a": 1, "b": 1, "q_1": 9}) for nm, e in r.trafo.items()}
    assert z["a"] ** 2 + z["b"] ** 2 == pytest.approx(9, rel=1e-8)
    z2 = {nm: ev(e, {"a": 1, "b": 1, "q_1": 0.01}) for nm, e in r.trafo.items()}
    assert all(v > 0 for v in z2.values())


def test_fixed_removes_curved_directions_moving_it():
    obj = mkobj([mkdir({"a": "a", "b": "-b"}), mkdir({"b": "b", "cc": "cc"})], ["a", "b", "cc"])
    r = red(obj, fixed=["a"])
    st = [b["status"] for b in r.blocks]
    assert "fixed" in st
    fb = next(b for b in r.blocks if b["status"] == "fixed")
    assert fb["labels"] == ["X₁"]
    kb = next(b for b in r.blocks if b["status"] != "fixed")
    assert kb["status"] == "reduced"
    assert "a" not in kb["support"]
    assert set(r.removed) == {"X₁", "X₂"}
    # a single direction moving the fixed coordinate disappears entirely
    r2 = red(mkobj([mkdir({"a": "a^2", "b": "a*b"})], ["a", "b"]), fixed=["a"])
    assert r2.blocks[0]["status"] == "fixed"
    assert all(v == k for k, v in r2.trafo.items())


def test_extactic_basis_over_moved_coordinates():
    s = "(k1 + k2 + k3 + k4)"
    obj = mkobj([mkdir({"x": f"{s}*x**2", "y": f"{s}*y**2"})], ["x", "y", "k1", "k2", "k3", "k4"])
    # quadrature and rational stages off, so the extactic basis is built
    b = red(obj, d_poly=0, d_darboux=1, separable=False).blocks[0]
    assert any("extactic complete" in c for c in b["certificates"])
    assert b["stage"] == "darboux"
    assert len(b["invariants"]) > 0
    gen = {"x": f"{s}*x^2", "y": f"{s}*y^2"}
    pt = {"x": 1.31, "y": 0.77, "k1": 0.2, "k2": 0.3, "k3": 0.15, "k4": 0.4}
    assert abs(lie_at(gen, b["invariants"][0], pt)) < 1e-4


def test_separable_characteristics_by_quadrature():
    # support 3 and atan invariants: only the quadrature stage reaches these
    obj = mkobj([mkdir({"x": "1 + x**2", "y": "1 + y**2", "z": "1 + z**2"})], ["x", "y", "z"])
    b = red(obj).blocks[0]
    assert b["stage"] == "separable"
    assert len(b["invariants"]) == 2
    gen = {"x": "1 + x^2", "y": "1 + y^2", "z": "1 + z^2"}
    for iv in b["invariants"]:
        assert abs(lie_at(gen, iv, {"x": 0.7, "y": 1.3, "z": 2.1})) < 1e-4
    # atan is no trafo entry
    assert b["status"] == "invariant_only"
    # a rational quadrature reduces end-to-end
    b2 = red(mkobj([mkdir({"x": "x**2", "y": "y**2", "z": "z**2"})], ["x", "y", "z"])).blocks[0]
    assert b2["stage"] == "separable"
    assert b2["status"] == "reduced"
    assert any(expr_equal(iv, "(x - y)/(x*y)") for iv in b2["invariants"])
    # a coupled generator is declined
    b3 = red(mkobj([mkdir({"x": "x", "y": "x + y"})], ["x", "y"])).blocks[0]
    assert b3.get("stage") != "separable"
    # the switch turns the stage off with a certificate
    b4 = red(mkobj([mkdir({"x": "x**2", "y": "y**2"})], ["x", "y"]), separable=False).blocks[0]
    assert b4.get("stage") != "separable"
    assert any("separable-characteristics stage skipped" in c for c in b4["certificates"])


def test_integrating_factor_liouvillian_integral():
    # damped oscillator, integrating factor 1/(x^2 + x*y + y^2)
    gen = {"x": "y", "y": "-x - y"}
    b = red(mkobj([mkdir(gen)], ["x", "y"])).blocks[0]
    assert b["stage"] == "intfactor"
    assert b["status"] == "invariant_only"
    assert len(b["invariants"]) == 1
    assert any("solved by quadrature" in c for c in b["certificates"])
    assert any(re.search(r"verified: X\(I\) = 0", c) for c in b["certificates"])
    assert abs(lie_at(gen, b["invariants"][0], {"x": 0.83, "y": 0.41})) < 1e-4


def test_sign_changing_carrier_real_with_covering_pin():
    # a - b is the only invariant and takes both signs on the positive orthant
    r = red(det({"x": "-(a - b)*x"}, {"y": "x"}, reconstruct=True), verbose=False)
    b1 = r.blocks[0]
    assert b1["status"] == "reduced"
    assert b1["coverage"] == "total"
    assert b1["carrier_domain"]["q_1"] == "real"
    # positive for every real carrier value, and on the orbit the carrier names
    for q in (-50, -3, -1, -0.4, 0, 0.4, 1, 3, 50):
        z = {k: ev(r.trafo[k], {"q_1": q}) for k in ("a", "b")}
        assert z["a"] > 0 and z["b"] > 0
        assert ev(b1["survivor_meaning"]["q_1"], z) == pytest.approx(q, abs=1e-12)


def test_certified_positive_carrier_keeps_positive_domain():
    # the moiety total A + B is positive on the orthant by inspection
    r = red(det(F_AB, {"y": "A + B"}, reconstruct=True, reduce_cq=False), verbose=False)
    dom = [v for b in r.blocks for v in (b.get("carrier_domain") or {}).values()]
    assert all(v == "positive" for v in dom)


def test_zero_limits_conditional():
    reactions = network(("P", "pP", "k_p*P"), ("pP", "P", "k_d*pP"))
    res = det(None, {"y": "s*pP"}, reactions=reactions, reconstruct=True)
    r = red(res, fixed=["s"], zero_limits=True)
    v = zero_rows(r)

    # flow of X2: k_p*exp(-e), k_d + k_p*(1 - exp(-e)), (P + pP)*exp(e) - pP
    def kd_zero(z):
        return -math.log1p(z["k_d"] / z["k_p"])

    def flow_P(e, z):
        return (z["P"] + z["pP"]) * math.exp(e) - z["pP"]

    # k_p*(P + pP) is a positive invariant, so k_p never vanishes
    assert v["k_p"]["verdict"] == "no"
    assert [v["P"]["verdict"], v["k_d"]["verdict"]] == ["if", "if"]
    # neither face is invariant: finite eps, not a limit
    assert not any(row["limit"] for row in v.values() if row["verdict"] == "if")
    assert all(row["certain"] for row in v.values())
    t1 = {"P": 1, "pP": 0.2, "k_p": 0.3, "k_d": 0.05, "s": 1}
    t2 = {"P": 0.2, "pP": 1, "k_p": 0.05, "k_d": 0.3, "s": 1}
    assert flow_P(kd_zero(t1), t1) > 0
    assert flow_P(kd_zero(t2), t2) < 0
    # the conditions agree with the flow at both points
    assert ev(v["k_d"]["condition"], t1)
    assert not ev(v["P"]["condition"], t1)
    assert not ev(v["k_d"]["condition"], t2)
    assert ev(v["P"]["condition"], t2)
    # the reported degenerate point is where the flow lands
    p_at = re.search(r"\bP = ([^,]+)", v["k_d"]["at"]).group(1)
    assert ev(p_at, t1) == pytest.approx(flow_P(kd_zero(t1), t1), rel=1e-8)
    out = lines(repr(r))
    assert grep(r"k_d = 0 +where +P\*k_p > k_d\*pP", out)
    assert grep(r"P += 0 +where +k_d\*pP > P\*k_p", out)
    assert not grep(r"^ +k_p += 0", out)
    assert not grep("everywhere", out)


def test_zero_limits_unconditional_and_switch():
    # the invariant is k1 + u*k2: either term moves into the other from any point
    res = det({"A": "k1 + u*k2 - kdeg*A"}, {"y": "A"}, fixed=["u"], reconstruct=True)
    r = red(res, zero_limits=True)
    v = r.to_dict()["zero_compatibility"]
    assert {row["coordinates"] for row in v} == {"k1", "k2"}
    assert all(row["verdict"] == "yes" for row in v)
    assert all(not row["condition"] for row in v)
    # the degenerate points keep the invariant unchanged
    pt = {"k1": 0.7, "k2": 1.3, "u": 2, "kdeg": 0.4, "A": 1.1}

    def inv(z):
        return z["k1"] + z["u"] * z["k2"]

    for row in v:
        z = dict(pt)
        for e in row["at"].split(", "):
            k, val = e.split(" = ")
            z[k] = ev(val, pt)
        assert inv(z) == pytest.approx(inv(pt), rel=1e-12)
        assert all(z[row2["coordinates"]] >= 0 for row2 in v)
    assert grep("k1 = 0 +everywhere", lines(repr(r)))
    assert red(res).to_dict()["zero_compatibility"] is None


def test_zero_limits_joint_vanishing():
    # equal weights: eps -> -inf takes a and b to zero together
    obj = mkobj([mkdir({"a": "a", "b": "b"}, "scaling", {"a": "1", "b": "1"})], ["a", "b", "other"])
    v = red(obj, zero_limits=True).to_dict()["zero_compatibility"]
    assert [row["coordinates"] for row in v] == ["a, b"]
    assert v[0]["verdict"] == "yes"
    assert v[0]["limit"] is True
    # opposite weights: a -> 0 only with b -> inf, no zero limit
    obj2 = mkobj([mkdir({"a": "a", "b": "-b"}, "scaling", {"a": "1", "b": "-1"})], ["a", "b", "other"])
    r2 = red(obj2, zero_limits=True)
    assert all(row["verdict"] == "no" for row in r2.to_dict()["zero_compatibility"])
    assert "Zero limits" not in repr(r2)


def test_log_parametrised_direction_reduced_and_mapped_back():
    tr = {"k1": "exp(lk1)", "k2": "exp(lk2)", "s": "exp10(ls)", "B": "10^lB"}
    g = {"y": "log10(s*A)"}
    r = red(det(F_AB, g, trafo=tr, reconstruct=True))
    assert set(r.removed) == {"X₁", "X₂"}
    assert set(r.trafo) == {"A", "lB", "lk1", "lk2", "ls"}
    comp = {**compose(tr, r.trafo), "A": r.trafo["A"]}
    assert identifiable(F_AB, g, trafo=comp)


def test_surplus_gauge_pinned_before_balance_search():
    pos = ["C", "gL", "gt"]
    r = red(det(F_OHM, {"y": "V"}, positive=pos, reconstruct=True), positive=pos)
    assert not r.remaining
    # the real invariant gL*EL + gt*Es is not hidden inside exp()
    assert not any("exp(" in iv for b in r.blocks for iv in b.get("invariants") or [])
    assert identifiable(F_OHM, {"y": "V"}, trafo=r.trafo, positive=pos)


def test_exp10_translation_reduced_in_chart():
    tr = {"C": "exp10(lC)", "gL": "exp10(lgL)", "gt": "exp10(lgt)"}
    res = det(F_OHM, {"y": "V"}, trafo=tr, positive=False, reconstruct=True)
    assert len(res.symmetries) == 3
    r = red(res, positive=False)
    assert not r.remaining
    comp = {**compose(tr, r.trafo), "EL": r.trafo["EL"], "Es": r.trafo["Es"]}
    assert identifiable(F_OHM, {"y": "V"}, trafo=comp, positive=False)


@pytest.mark.slow
def test_log_of_positive_coordinate_reduced():
    # Hill function with a free exponent: n moves with log(u) and log(K)
    f = {"x": "v*u^n/(K^n + u^n) - d*x", "u": "-e*u"}
    res = det(f, {"y": "x"}, reconstruct=True)
    assert any(any("log(" in str(c) for c in d["generator"].values()) for d in res.symmetries)
    r = red(res)
    assert not r.remaining
    assert identifiable(f, {"y": "x"}, trafo=r.trafo)


def test_symbolic_weight_scaling_reduced_in_log_coordinates():
    # S-system: x2 moves with a1^g12 and b2^(h22 - 1)
    f = {"x1": "a1*x2^g12 - b1*x1^h11", "x2": "a2*x1^g21 - b2*x2^h22"}
    r = red(det(f, {"y": "x1"}, reconstruct=True))
    assert not r.remaining
    assert identifiable(f, {"y": "x1"}, trafo=r.trafo)


def test_carriers_matched_not_greedy():
    # two unobserved routes into one observed state
    f = {"A": "-(ka + kb)*A", "B1": "ka*A - k1*B1", "B2": "kb*A - k2*B2", "C": "k3*B1 + k4*B2 - kc*C"}
    tr = {"A": "A0", "B1": "0", "B2": "0", "C": "0"}
    r = red(det(f, {"y": "C"}, trafo=tr, fixed=["A0"], reconstruct=True))
    assert not r.remaining
    assert not set(tr) & set(r.trafo)
    assert identifiable(f, {"y": "C"}, trafo={**tr, **r.trafo}, fixed=["A0"])


@pytest.mark.slow
def test_face_section_switches_leaks_off():
    # catenary compartments with leaks from all: every orbit reaches two leaks at zero
    f = {"x1": "-(k01 + kf1)*x1 + kb1*x2",
         "x2": "-(k02 + kf2 + kb1)*x2 + kf1*x1 + kb2*x3",
         "x3": "-(k03 + kb2)*x3 + kf2*x2"}
    tr = {"x1": "D", "x2": "0", "x3": "0"}
    res = det(f, {"y": "x1"}, trafo=tr, reconstruct=True)
    assert len(res.symmetries) == 2
    t0 = time.monotonic()
    r = red(res)
    assert time.monotonic() - t0 < 60
    assert not r.remaining
    zeros = [k for k, v in r.trafo.items() if v == "0"]
    assert len(zeros) == 2
    assert set(zeros) <= {"k01", "k02", "k03"}
    b = blocks_of(r, "curved")[0]
    assert any("face section" in c for c in b["certificates"])
    # the face is a zero set every orbit reaches
    zb = blocks_of(red(res, zero_limits=True), "curved")[0]
    assert any(row["verdict"] == "yes" and row["coordinates"] == ", ".join(sorted(zeros))
               for row in zb["zero_compatibility"])
    tr2 = {**tr, **{k: v for k, v in r.trafo.items() if k not in tr}}
    assert identifiable(f, {"y": "x1"}, trafo=tr2)
    # each rate the face keeps becomes a fresh q_<k>
    kept = [k for k in b["support"] if k not in zeros]
    assert all(re.fullmatch(r"q_[0-9]+", r.trafo[k]) for k in kept)
    assert set(b["survivor_meaning"]) == {r.trafo[k] for k in kept}
    others = list(tr.values()) + [v for k, v in r.trafo.items() if k not in kept]
    assert not any(v.startswith("q_") for v in others)
    out = lines(repr(r))
    assert grep(r"^Sections:", out)
    assert grep("face: every orbit", out)


def test_summary_reports_once():
    res = det({"A": "k1 + u*k2 - kdeg*A"}, {"y": "A"}, fixed=["u"], reconstruct=True)
    r = red(res)
    out = lines(r.summary())
    assert sum(s.startswith("Reduced ") for s in out) == 1
    assert grep(r"^Blocks", out)
    assert str(r.summary()) == str(r.summary())
    out2 = lines(res.summary())
    assert sum(s.startswith("Result:") for s in out2) == 1


def test_translation_group_searched_in_sum():
    # the generators see Km, R1, R2 only through Km + R1 + R2
    g1 = {"Km": "Km + R1 + R2", "kdg": "kdg"}
    g2 = {"Km": "ksec*(Km + R1 + R2)^2", "ksec": "kdg*ksec", "ktl": "-kdg*ktl"}
    obj = mkobj([mkdir(g1, "general"), mkdir(g2, "general")], ["Km", "kdg", "ksec", "ktl", "R1", "R2"])
    r = red(obj)
    assert not r.remaining
    cb = blocks_of(r, "curved")[0]
    assert any("translation-group" in c for c in cb["certificates"])
    assert any(expr_equal(iv, "kdg/(Km + R1 + R2) + ksec")
               or expr_equal(iv, "(Km + R1 + R2)/(kdg + ksec*(Km + R1 + R2))") for iv in cb["invariants"])
    pt = {"Km": 0.7, "kdg": 2.3, "ksec": 1.9, "ktl": 0.31, "R1": 1.7, "R2": 0.45}
    for iv in cb["invariants"]:
        assert abs(lie_at(g1, iv, pt)) < 1e-4
        assert abs(lie_at(g2, iv, pt)) < 1e-4


def test_unmoved_monomial_searched_as_one_symbol():
    # R1*R2*R3 never moves and enters only as a product
    W = "R1*R2*R3"
    g1 = {"Km": f"Km + {W}", "kdg": "kdg"}
    g2 = {"Km": f"ksec*(Km + {W})^2", "ksec": f"{W}*kdg*ksec", "ktl": f"-{W}*kdg*ktl"}
    obj = mkobj([mkdir(g1, "general"), mkdir(g2, "general")],
                ["Km", "kdg", "ksec", "ktl", "R1", "R2", "R3"])
    r = red(obj)
    assert not r.remaining
    cb = blocks_of(r, "curved")[0]
    assert any("unmoved monomial" in c for c in cb["certificates"])
    pt = {"Km": 0.7, "kdg": 2.3, "ksec": 1.9, "ktl": 0.31, "R1": 1.7, "R2": 0.45, "R3": 1.2}
    for iv in cb["invariants"]:
        assert abs(lie_at(g1, iv, pt)) < 1e-4
        assert abs(lie_at(g2, iv, pt)) < 1e-4
    texts = cb["invariants"] + list(r.trafo.values()) + list(cb["survivor_meaning"].values())
    assert not any("si_w" in s for s in texts)


def test_chart_search_stops_at_timeout():
    f = {"c": "k0 - kd*c", "x": "w*(q + b*c) - x"}
    pos = ["c", "k0", "kd", "w", "b", "x"]
    res = det(f, {"y": "x"}, gauge=None, reconstruct=True, positive=pos)
    t0 = time.monotonic()
    r = red(res, positive=pos, timeout=0.01)
    assert time.monotonic() - t0 < 30
    assert any(b["status"] == "invariant_only" and "timeout" in (b.get("reason") or "")
               for b in blocks_of(r, "curved"))


def test_square_root_of_a_perfect_square_is_taken():
    from symident.reduction.core import _tidy_root
    assert _tidy_root("sqrt(16*q_2^2 + 8*q_2*q_3 + q_3^2)") == "4*q_2 + q_3"
    assert _tidy_root("sqrt(4*kb2 + q_1^2)") == "sqrt(4*kb2 + q_1^2)"


@pytest.mark.slow
def test_partial_chart_is_named_in_the_verdict():
    f = {"x1": "kin1*u - k2*x1 - d3*x1 + k4*x2", "x2": "k2*x1 - k4*x2 + k5*x3", "x3": "-k5*x3"}
    m = si.Model(f, {"y0": "x2", "y1": "s1*x2"}, trafo={"x2": "1", "x3": "0"},
                 conditions={"rows": ["a", "b"], "cols": {"u": [1, 3]}})
    red = si.detect(m, reconstruct=True).reduce()
    assert red.partial == ["X₂"]
    assert "covers only part of the positive orthant" in str(red).splitlines()[0]


def test_labels_partition_the_directions():
    rx = {"species": ["A", "B", "C"], "stoichiometry": [[1, -1, 0], [0, -1, 1], [0, 1, -1]],
          "rates": ["k1*B", "k2*B", "k3*C"]}
    red = si.detect(si.Model(reactions=rx, observables={"y": "B"}), equilibrate=True, reduce_cq=True,
                    reconstruct=True).reduce()
    labels = red.removed + red.remaining
    assert len(labels) == len(set(labels)) == 3
    assert red.remaining == []


def test_dependent_scaling_is_named_not_reduced():
    from symident.reduction import symmetry_reduction

    def mk(g, w):
        return {"type": "scaling", "generator": g, "weights": w, "support": list(g), "explicit": True}
    obj = {"identifiable": False, "rank": 0, "dim": 2,
           "info": {"engine": "modular", "settings": {}, "coordinates": ["k1", "k2"]},
           "symmetries": [mk({"k1": "k1"}, {"k1": "1"}), mk({"k2": "k2"}, {"k2": "1"}),
                          mk({"k1": "k1", "k2": "k2"}, {"k1": "1", "k2": "1"})]}
    r = symmetry_reduction(obj)
    assert (r["removed"], r["remaining"], r["dependent"]) == (["X₁", "X₂"], [], ["X₃"])


@pytest.mark.slow
def test_alternative_charts_are_valid_and_ranked():
    from symident.reduction.core import _chart_valid
    m = si.Model({"x1": "kin1*u - k2*x1 - d3*x1 + k4*x2", "x2": "k2*x1 - k4*x2 + k5*x3", "x3": "-k5*x3"},
                 {"y0": "x2", "y1": "s1*x2"}, trafo={"x2": "1", "x3": "0"},
                 conditions={"rows": ["a", "b"], "cols": {"u": [1, 3]}})
    det = si.detect(m, reconstruct=True)
    red = det.reduce(alternatives=True)
    assert _chart_valid(red._raw, det.dim)
    for alt in red.alternatives:
        assert set(alt) == {"trafo", "pins", "removed", "remaining", "rational"}
        assert len(alt["removed"]) <= len(red.removed)
    simple = si.detect(si.Model({"A": "-k1*A", "B": "k1*A - k2*B"}, {"y": "s*B"}), reconstruct=True)
    assert simple.reduce().alternatives == []

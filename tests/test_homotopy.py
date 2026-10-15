"""The homotopy route: series helpers, exact directions along parameter lines, its guards."""

from types import SimpleNamespace

import numpy as np
import pytest
import sympy as spy

from symident.detection import symmetry_detection
from symident.detection.classify import classify_direction
from symident.detection.fields import ser_inv, ser_mul
from symident.detection.observability.reconstruct import Reconstruction
from symident.detection.series import pade_at, pade_mod, ser_div, series_null
from symident.options import reconst_control

P = 1000003


def test_series_division_and_inverse_are_exact():
    a, b = [3, 1, 4, 1], [1, 5, 0, 2]
    assert ser_mul(ser_div(a, b, P), b, P) == a
    assert ser_mul(ser_inv(b, P), b, P) == [1, 0, 0, 0]


def test_pade_recovers_a_rational_function_of_t():
    pq = pade_mod([1, 3, 3, 3, 3], 1, 1, P)
    assert pq == {"P": [1, 2], "Q": [1, P - 1]}
    assert pade_at(pq, 3, P) == 7 * pow(-2, -1, P) % P
    assert pade_at(pq, 1, P) is None
    assert pade_mod([1, 3, 3, 3, 4], 1, 1, P) is None


def test_series_kernel_of_a_row_with_series_entries():
    N = 3
    S = np.zeros((1, 2 * N), dtype=np.int64)
    S[0, 0] = 1
    S[0, N:N + 2] = [P - 1, P - 1]
    ns = series_null(S, 2, N, P)
    assert ns["free"] == [1] and ns["piv"] == [0]
    assert ns["V"][0].tolist() == [[1, 1, 0], [1, 0, 0]]


LIN2 = dict(f={"x1": "kb + kin*u - (k1 + kl)*x1", "x2": "k1*x1 - k2*x2"}, g={"y": "s*x2"},
            conditions={"rows": ["a", "b"], "cols": {"u": [1, 2]}})
BIND = dict(f={"A": "ka - kd*A - kon*A*L", "C": "kon*A*L - koff*C"}, g={"y": "s*C"},
            conditions={"rows": ["a", "b"], "cols": {"L": [1, 2]}})
SAT = dict(f={"x": "v*u*x/(K + x) + kb - d*x"}, g={"y": "s*x"},
           conditions={"rows": ["a", "b"], "cols": {"u": [1, 2]}})


def _expr(x):
    return spy.sympify(str(x).replace("^", "**"))


def _same(u, v):
    return set(u) == set(v) and all(spy.simplify(_expr(u[k]) - _expr(v[k])) == 0 for k in u)


def homotopy_directions(model, monkeypatch):
    """Every residual direction of the joint reconstruction handed to the homotopy route."""
    out = []
    run = Reconstruction.run

    def first(self):
        if self.an.joint:
            out.extend(self.homotopy([{"closed_form": False} for _ in self.residual]))
        return run(self)
    monkeypatch.setattr(Reconstruction, "run", first)
    symmetry_detection(**model, equilibrate=True, reconstruct=True)
    return [classify_direction(d) for d in out if d.get("closed_form")]


def test_homotopy_fits_entries_in_the_parameters(monkeypatch):
    dirs = homotopy_directions(BIND, monkeypatch)
    assert [d["route"] for d in dirs] == ["homotopy fit"] * 2
    assert [d["type"] for d in dirs] == ["scaling"] * 2
    assert _same(dirs[0]["vector"], {"ka": 1, "kd": 1, "koff": 1, "kon": 1})
    assert _same(dirs[1]["vector"], {"ka": 1, "koff": 1})


def test_homotopy_replaces_linear_resting_states(monkeypatch):
    dirs = homotopy_directions(LIN2, monkeypatch)
    want = [{"k1": "k1^2 + k1*kl", "k2": "k1*k2 + k2*kl", "kb": "k1*kb", "kin": "k1*kin"},
            {"k1": "k1", "k2": "k2", "kl": "-k1"},
            {"k1": "k1^2 + k1*kl", "k2": "k2*kl"}]
    assert len(dirs) == 3
    for d, w in zip(dirs, want):
        assert d["type"] == "general" and _same(d["vector"], w)


def test_homotopy_closes_what_the_other_routes_leave_open():
    def general(hom):
        r = symmetry_detection(**SAT, equilibrate=True, reconstruct=True,
                               control=reconst_control(homotopy=hom))
        return [d for d in r["symmetries"] if d["type"] == "general"]
    assert [d.get("generator") for d in general(False)] == [None, None]
    on = general(True)
    assert [d["route"] for d in on] == ["homotopy fit in the parameters and the resting states"] * 2
    assert [d["level"] for d in on] == ["exact"] * 2
    for d in on:
        syms = {str(s) for x in d["generator"].values() for s in _expr(x).free_symbols}
        assert syms <= {"K", "d", "kb", "v", "x_c1", "x_c2"} and {"x_c1", "x_c2"} <= syms


def test_states_entering_a_resting_equation_linearly_are_eliminated():
    from symident.detection.observability.homstate import reduce_relations
    sat = "x = v*u*x/(K + x) + kb - d*x"
    core, rel = reduce_relations([sat, "z = k*x - c*z"], ["x", "z"])
    assert core == ["x"] and len(rel) == 1
    # a resting state held at zero drops out before the elimination
    lines = [sat, "R = k*x - dR*R - b*L*R", "L = -b*L*R", "A = b*L*R - dA*A"]
    core, rel = reduce_relations(lines, ["x", "R", "A"], ["L"])
    assert core == ["x"] and len(rel) == 1
    # an all-linear system keeps its states for the linear solve
    lin = ["x1 = kb + kin - (k1 + kl)*x1", "x2 = k1*x1 - k2*x2"]
    assert reduce_relations(lin, ["x1", "x2"])[0] == ["x1", "x2"]


def test_homotopy_closes_a_saturating_state_seen_through_a_linear_one():
    m = dict(f={"x": "v*u*x/(K + x) + kb - d*x", "z": "k*x - c*z"}, g={"y": "s*z"},
             conditions={"rows": ["a", "b"], "cols": {"u": [1, 2]}})

    def general(hom):
        r = symmetry_detection(**m, equilibrate=True, reconstruct=True,
                               control=reconst_control(homotopy=hom))
        return [d for d in r["symmetries"] if d["type"] == "general"]
    off = general(False)
    assert len(off) >= 2 and all(d.get("generator") is None for d in off)
    on = general(True)
    assert [d["route"] for d in on] == ["homotopy fit in the parameters and the resting states"] * 2
    assert [d["level"] for d in on] == ["exact"] * 2


def test_homotopy_closes_a_direction_in_the_recast_coordinates_of_a_symbolic_exponent():
    m = dict(f={"R": "k_pr/(1 + k_fb*FB^nh) - k_dg*R", "FB": "k_pf*R - k_df*FB"},
             g={"R_obs": "s1*R", "FB_obs": "s2*FB"}, equilibrate=True, reconstruct=True)
    off = symmetry_detection(**m, control=reconst_control(homotopy=False))
    assert [s["level"] for s in off["symmetries"]].count("open") == 1
    on = symmetry_detection(**m)
    (hom,) = [s for s in on["symmetries"] if s["route"] == "homotopy fit"]
    assert "nh" in hom["support"] and "log(FB)" in "".join(hom["generator"].values())
    assert {s["level"] for s in on["symmetries"]} == {"exact"}
    assert on["info"]["rank_status"]["status"] == "proven"


def _guarded(recast=(), joint=True, gaps=False, ss=True, transient=False, at_one=False, tau="", **ctrl):
    rec = Reconstruction.__new__(Reconstruction)
    rec.an = SimpleNamespace(recast=list(recast), joint=joint, transient=transient, znames=["k", "tau"],
                             multi={"tapes": [{"ver_time": tau}]},
                             run=SimpleNamespace(ss_constraint=ss, has_gaps=gaps, timing=False, leaf_names=["k"]))
    rec.sc = {"ref": {"at_one": at_one}}
    rec.ctrl = reconst_control(**ctrl)
    rec.residual = [3]
    rec.attempts = []
    interp = [{"closed_form": False}]
    return rec.homotopy(interp) is interp and rec.attempts[0][0] == "homotopy"


@pytest.mark.parametrize("kw", [{"joint": False}, {"gaps": True, "at_one": True}, {"gaps": True, "tau": "tau"},
                                {"homotopy": False}, {"joint": False, "ss": False, "transient": True}])
def test_homotopy_declines_where_it_does_not_apply(kw):
    assert _guarded(**kw)


def test_open_directions_list_the_routes_tried():
    r = symmetry_detection(**SAT, equilibrate=True, reconstruct=True,
                           control=reconst_control(homotopy=False))
    att = r["info"]["attempts"]
    assert len(att) == 2
    assert all(["homotopy", "switched off (reconst_control(homotopy=False))"] in a["attempts"] for a in att)
    assert all(len({route for route, _ in a["attempts"]}) > 1 for a in att)


def test_homotopy_degree_is_its_own_setting():
    from symident.detection.observability.homotopy import Homotopy
    rec = SimpleNamespace(an=SimpleNamespace(sys=SimpleNamespace(hom_prepare=lambda base: None, hom_params=[]),
                                             run=SimpleNamespace(slot=lambda nm: None),
                                             ctrl=reconst_control(degree_cap=1, homotopy_degree=5)))
    assert Homotopy(rec, [0], 1).d_cap == 5
    with pytest.raises(ValueError):
        reconst_control(homotopy_degree=0)


def test_radical_initial_values_have_their_polynomial_relation():
    from symident.sym.algebraic import radical_relation
    x, a, b = spy.symbols("x a b")
    (f,) = radical_relation("x", "(b + sqrt(b^2 + 4*a))/2".replace("^", "**"))
    assert spy.expand(f.xreplace({x: (b + spy.sqrt(b**2 + 4*a)) / 2})) == 0
    (f,) = radical_relation("x", "sqrt(1 + sqrt(a))")
    assert spy.degree(f, x) == 4
    (f,) = radical_relation("x", "a**(1/3) + b")
    assert spy.degree(f, x) == 3
    assert radical_relation("x", "a + b") is None


def test_series_of_a_root_follows_the_kernel_branch():
    from symident.gfp import sqrt_mod
    from symident.sym.taylor import _mul, series_of
    a = spy.Symbol("a")
    p = 1000003
    r = series_of(spy.sqrt(a), {a: [9, 3, 5]}, 5, p)
    assert r[0] == sqrt_mod(9, p) and _mul(r, r, 5, p) == [9, 3, 5, 0, 0]


def test_homotopy_closes_a_direction_of_a_start_with_radicals():
    # the start is a resting state given in closed form through a square root
    b = "(v*u + kb - d*K)"
    m = dict(f={"x": "v*u*x/(K + x) + kb - d*x"}, g={"y": "s*x"}, trafo={"x": f"({b} + sqrt({b}^2 + 4*d*kb*K))/(2*d)"},
             conditions={"rows": ["a", "b"], "cols": {"u": [1, 2]}})

    def run(hom):
        return symmetry_detection(**m, reconstruct=True, control=reconst_control(homotopy=hom))["symmetries"]
    assert not all(d["explicit"] for d in run(False))
    on = run(True)
    assert all(d["explicit"] for d in on) and any(str(d["route"]).startswith("homotopy fit") for d in on)


def test_homotopy_follows_lines_through_events_after_the_start():
    # a resting state seen only through downstream states, with a dose at the start and one later
    f = {"x": "v*u*x/(K + x) + kb - d*x", "Rec": "k*x - degRec*Rec - Bind*L*Rec", "L": "-Bind*L*Rec",
         "A": "Bind*L*Rec - degA*A"}
    cond = {"rows": ["a", "b"], "cols": {"u": [1, 2], "d0": [1, 2], "d1": [1, 1]}}
    ev = [{"var": "L", "time": 0, "value": "d0", "method": "add"},
          {"var": "L", "time": 24, "value": "d1", "method": "add"}]
    r = symmetry_detection(f=f, g={"y": "s*A"}, events=ev, conditions=cond, equilibrate=True,
                           reconstruct=True, cores=2)
    gen = [d for d in r["symmetries"] if d["type"] == "general"]
    assert len(gen) == 2 and all(d["explicit"] for d in gen)
    assert {d["route"] for d in gen} == {"homotopy fit in the parameters and the resting states"}

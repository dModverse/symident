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


@pytest.mark.slow
def test_homotopy_closes_what_the_other_routes_leave_open():
    def general(hom):
        r = symmetry_detection(**SAT, equilibrate=True, reconstruct=True,
                               control=reconst_control(homotopy=hom))
        return [d for d in r["symmetries"] if d["type"] == "general"]
    assert [d.get("generator") for d in general(False)] == [None, None]
    on = general(True)
    assert [d["route"] for d in on] == ["homotopy fit in the parameters and the resting states"] * 2
    for d in on:
        syms = {str(s) for x in d["generator"].values() for s in _expr(x).free_symbols}
        assert syms <= {"K", "d", "kb", "v", "x_c1", "x_c2"} and {"x_c1", "x_c2"} <= syms


def _guarded(recast=(), joint=True, gaps=False, **ctrl):
    rec = Reconstruction.__new__(Reconstruction)
    rec.an = SimpleNamespace(recast=list(recast), joint=joint,
                             run=SimpleNamespace(ss_constraint=True, has_gaps=gaps))
    rec.ctrl = reconst_control(**ctrl)
    rec.residual = [3]
    interp = [{"closed_form": False}]
    return rec.homotopy(interp) is interp


@pytest.mark.parametrize("kw", [{"recast": [{"E": "_E", "L": "_L", "base": "x", "exp": "n"}]},
                                {"joint": False}, {"gaps": True}, {"homotopy": False}])
def test_homotopy_declines_where_it_does_not_apply(kw):
    assert _guarded(**kw)

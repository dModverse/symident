"""Certificates of the Lie and gap orders and the random points they and the verification take."""

import numpy as np
import pytest

import symident as si
from symident.gfp import PRIMES


def _rotation(value, method="replace"):
    # the observable moves only in the segment that an event value reading a state starts
    return si.Model({"A": "u*D", "D": "-k*A*u", "B": "c*(1 - u)", "u": "0"}, {"y": "A"},
                    trafo={"A": "0", "D": "0", "B": "0", "u": "0"},
                    events=[{"var": "u", "time": 0, "value": "0", "method": "replace"},
                            {"var": "A", "time": 5, "value": value, "method": method},
                            {"var": "u", "time": 5, "value": "1", "method": "replace"}])


def _moving_gap():
    return si.Model({"x": "u*w - k*x", "w": "c*(1 - u)*v", "v": "1 - u", "u": "0"}, {"y": "s*x"},
                    trafo={"w": "0", "v": "0", "u": "0"},
                    events=[{"var": "u", "time": 0, "value": "0", "method": "replace"},
                            {"var": "u", "time": "tau", "value": "1", "method": "replace"}])


def _sum_after_dose():
    # a direction that no scaling proves, behind a gap
    return si.Model({"x": "-k*x"}, {"y": "(a + b)*x"}, trafo={"x": "1"},
                    events=[{"var": "x", "time": 2, "value": "1", "method": "add"}])


def test_an_event_value_that_reads_a_state_keeps_the_lie_certificate_sound():
    r = si.detect(_rotation("B"))
    assert r.identifiable and r.rank == 2 and r.info["lie_certified"]
    assert si.detect(_rotation("5*c")).identifiable
    r = si.detect(_rotation("B + d", "add"))
    assert (r.rank, r.dim) == (2, 3) and r.info["lie_certified"]


def test_a_certified_order_below_the_scanned_rank_keeps_the_scan(monkeypatch):
    import symident.detection.observability.analysis as an
    monkeypatch.setattr(an, "lie_certificate", lambda segs, *a, **k: [1] * len(segs))
    r = si.detect(_rotation("B + d", "add"))
    assert (r.rank, r.dim) == (2, 3)
    assert not r.info["lie_certified"]
    assert any("Lie order certificate" in lim for lim in r.info["limits"])


def test_the_affine_space_after_an_event_is_a_basis():
    from symident.detection.fields import event_space
    p = PRIMES[0]
    V = np.array([[1, 1, 0], [0, 1, 0]], dtype=np.int64)
    # a replace by a value that reads no state resets its target
    W = event_space(V, [0], [0], [True], 3, p)
    assert W.shape == (1, 3) and W.tolist() == [[0, 1, 0]]
    # a value that reads a state frees its target
    W = event_space(V, [2], [1], [False], 3, p)
    assert W.shape == (3, 3)
    W = event_space(V, [1], [2], [True], 3, p)
    assert W.shape == (2, 3)


def test_no_gap_bound_with_an_event_time_in_the_parameters():
    r = si.detect(_moving_gap())
    assert r.info["gap_bound"] is None
    assert r.info["gap_certified"] and r.info["rank_proof"] == "exact scalings"


def test_a_used_gap_bound_is_at_least_the_rank():
    r = si.detect(_sum_after_dose())
    assert (r.rank, r.dim) == (2, 3) and r.info["gap_certified"]
    assert r.info["gap_bound"] is not None and r.rank <= r.info["gap_bound"]
    for m in (_rotation("B"), _rotation("B + d", "add")):
        r = si.detect(m)
        assert r.info["gap_bound"] is None or r.rank <= r.info["gap_bound"]


def test_a_rank_above_the_gap_bound_certifies_nothing(monkeypatch):
    import symident.detection.gapcert as gc
    monkeypatch.setattr(gc, "gap_bound", lambda *a, **k: {"bound": 1, "why": None, "terms": 0.0})
    r = si.detect(_sum_after_dose())
    assert r.info["gap_certified"] is False
    assert any("gap bound" in lim for lim in r.info["limits"])


CERTIFIED = [
    ({"A": "-k1*A", "B": "k1*A - k2*B"}, {"y": "s*B"}, {"A": "A0", "B": "0"}),
    ({"x": "th"}, {"y": "x^5"}, {"x": "0"}),
    ({"x1": "-k1*x1", "x2": "k1*x1 - k2*x2 + a*x3", "x3": "-k3*x3"}, {"y": "x2"},
     {"x1": "1", "x2": "0", "x3": "0"}),
    ({"x1": "a - (d + f)*x1", "x2": "b - (d + f)*x2", "f": "-c*f"}, {"y": "x1 + x2"},
     {"x1": "a/d", "x2": "b/d", "f": "1"}),
]


@pytest.mark.parametrize("f, g, trafo", CERTIFIED)
def test_certified_orders_have_an_error_bound(f, g, trafo):
    r = si.detect(si.Model(f, g, trafo=trafo))
    assert r.info["lie_certified"]
    assert r.info["order_bound"] is not None and 0 <= r.info["order_bound"] < 1e-6


def test_a_verified_rank_includes_the_order_bound():
    f = {"A": "k1 - d1*A + kx*X", "B": "k2*A - d2*B", "X": "kp*plasmid - dx*X"}
    rest = [{"X": f"kp*{u}/dx", "A": f"(k1 + kx*kp*{u}/dx)/d1", "B": f"k2*(k1 + kx*kp*{u}/dx)/(d1*d2)",
             "plasmid": str(u)} for u in (0, 1)]
    m = si.Model(f, {"yA": "s*(A + X)", "yB": "B"}, trafo=rest,
                 conditions={"rows": ["ctrl", "oe"], "cols": {"dose": [1, 1]}},
                 events=[{"var": "A", "time": 0, "value": "dose", "method": "add"}])
    r = si.detect(m, reconstruct=True)
    st, ob = r.info["rank_status"], r.info["order_bound"]
    assert st["status"] == "verified" and ob is not None and ob > 0
    assert st["probability"] >= ob and r.info["bound"]["probability"] >= ob
    r = si.detect(_rotation("B"))
    assert r.info["order_bound"] is not None and r.info["order_bound"] < 1e-3


def test_a_free_exponent_of_a_positive_state_keeps_the_order_bound():
    m = si.Model({"x": "v0 + b*v1 - V*x^n/(K^n + x^n)", "z": "V*x^n/(K^n + x^n) - e*z"}, {"y": "s*z"},
                 trafo={"x": "x0", "z": "z0"}, positive=True)
    r = si.detect(m, reconstruct=True)
    assert r.info["lie_certified"] and r.info["order_bound"] is not None and r.info["order_bound_why"] is None
    assert r.info["rank_status"]["status"] == "verified"
    assert all(s.level in ("exact", "verified") for s in r.symmetries)


def test_a_missing_order_bound_names_its_reason():
    m = si.Model({"x": "k1 - k2*x*z", "z": "k3 - k4*z"}, {"y": "s*x"})
    r = si.detect(m, equilibrate=True)
    assert r.info["order_bound"] is None
    assert r.info["order_bound_why"] == "the resting states are solved modulo the engine prime only"


def test_a_known_input_chain_certifies_no_order_past_its_exact_jets():
    m = si.Model({"x": "-k*x + b*u"}, {"y": "s*x"}, known_inputs=["u"])
    r = si.detect(m)
    K = r.info["known_inputs"]["jet_order"]
    assert r.info["lie_order_used"] <= K - 1 or not r.info["lie_certified"]
    r2 = si.detect(m, control=si.reconst_control(input_jet_order=2 * K))
    assert (r.rank, r.dim) == (r2.rank, r2.dim) == (3, 4)


def test_verification_points_are_uniform_modulo_q():
    from symident import session
    from symident.gfp import uniform_point
    q = 1000000007
    with session.opened({}):
        pts = np.concatenate([uniform_point(1000, q, seed=s) for s in range(4)])
    assert pts.min() >= 0 and pts.max() < q
    # no residue class below min(PRIMES) - q has double weight
    assert abs(np.mean(pts < min(PRIMES) - q) - (min(PRIMES) - q) / q) < 0.05


def test_initial_values_enter_the_degree_over_their_common_denominator():
    from symident.detection.bound import _model_degrees
    md = _model_degrees((("x", "-k*x"), ("z", "-k*z")), (("y", "x + z"),), (("x", "1/b"), ("z", "1/d")))
    assert md[3] == 2


def test_a_unit_fill_is_checked_at_the_verification_points(monkeypatch):
    import symident.detection.observability.analysis as an
    from symident.detection.observability.reconstruct import Reconstruction
    monkeypatch.setattr(Reconstruction, "run", lambda self: [])
    m = si.Model({"x": "-k*x", "z": "-a*z"}, {"y": "x"}, trafo={"x": "1", "z": "1"})
    ctrl = si.reconst_control(exact_proof=False)
    r = si.detect(m, reconstruct=True, control=ctrl)
    (s,) = r.symmetries
    assert s.support == ["a"] and s.explicit
    monkeypatch.setattr(an, "verify_in_nullspace", lambda *a, **k: False)
    (s,) = si.detect(m, reconstruct=True, control=ctrl).symmetries
    assert s.support == ["a"] and not s.explicit


def test_a_homotopy_direction_is_checked_at_two_points(monkeypatch):
    import symident.detection.observability.homotopy as hm
    passed = []
    orig = hm.in_kernel

    def counted(*a, **k):
        ok = orig(*a, **k)
        passed.extend([1] if ok else [])
        return ok
    monkeypatch.setattr(hm, "in_kernel", counted)
    m = si.Model({"x": "v*u*x/(K + x) + kb - d*x", "z": "k*x - c*z"}, {"y": "s*z"},
                 conditions={"rows": ["a", "b"], "cols": {"u": [1, 2]}})
    r = si.detect(m, equilibrate=True, reconstruct=True, control=si.reconst_control(homotopy=True))
    hom = [s for s in r.symmetries if (s.route or "").startswith("homotopy")]
    assert len(hom) == 2 and len(passed) >= 2 * len(hom)


def test_a_skipped_verification_point_widens_the_bound(monkeypatch):
    import symident.detection.verify as vf
    m = si.Model({"x1": "kb + kin*u - (k1 + kl)*x1", "x2": "k1*x1 - k2*x2"}, {"y": "s*x2"},
                 trafo={"x1": "D", "x2": "0"}, conditions={"rows": ["a", "b"], "cols": {"u": [1, 2]}})
    with si.switches(NOFIELDS=True):
        r0 = si.detect(m, reconstruct=True)
    orig, calls = vf._in_nullspace_at, []

    def skip_first(*a, **k):
        calls.append(1)
        return None if len(calls) == 1 else orig(*a, **k)
    monkeypatch.setattr(vf, "_in_nullspace_at", skip_first)
    with si.switches(NOFIELDS=True):
        r1 = si.detect(m, reconstruct=True)
    b0, b1 = r0.info["bound"], r1.info["bound"]
    assert b0["draws"] == 2 and b1["draws"] == 3
    # three pairs among three draws, one among two
    kernel = [r.info["bound"]["probability"] - r.info["order_bound"] for r in (r0, r1)]
    assert kernel[0] > 0 and kernel[1] == pytest.approx(3 * kernel[0], rel=1e-6)

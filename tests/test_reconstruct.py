"""Closed forms of wide general directions: an inhibitor known only up to an affine map,
with a dose after the start, under both steady-state modes."""

import pytest
import sympy as spy

import symident as si


def core(two=False, mode="equilibrate", basal=True):
    """A receptor inhibited by S (and T), each made from a transcript and, with `basal`,
    from a basal source; L is dosed at 0 and again at 24."""
    inh = "(1 + kinh*S)" + ("*(1 + kinh3*T)" if two else "")
    odes = {"L": f"-Bind*L*Rec/{inh}",
            "Rec": f"prodRec - degRec*Rec - degRecS*Rec*S - Bind*L*Rec/{inh}",
            "A": f"Bind*L*Rec/{inh} - degA*A - degAS*A*S",
            "M": "synthM*A - hlM*M",
            "S": "synthS*M + bS - dS*S" if basal else "synthS*M - dS*S"}
    obs = {"y1": "s1*A", "y2": "M"}
    if two:
        odes.update(N="synthN*A - hlN*N", T="synthT*N + bT - dT*T")
        obs["y3"] = "N"
    ev = [{"var": "L", "time": 0, "value": "d0", "method": "add"},
          {"var": "L", "time": 24, "value": "d1", "method": "add"}]
    cond = {"rows": ["a", "b", "c"], "cols": {"d0": [1, 2, 1], "d1": [0, 1, 2]}}
    if mode == "equilibrate":
        return si.Model(odes, obs, events=ev, conditions=cond), {"equilibrate": True}
    s0 = "bS/dS" if basal else "0"
    rest = {"L": "0", "A": "0", "M": "0", "S": s0, "Rec": f"prodRec/(degRec + degRecS*{s0})"}
    if two:
        rest.update(N="0", T="bT/dT")
    return si.Model(odes, obs, events=ev, conditions=cond, trafo=rest), {}


def quadratic_in(gen, k):
    """Whether the component of `k` in `gen` has degree two in `k`."""
    K = spy.Symbol(k)
    return spy.degree(spy.numer(spy.together(spy.sympify(gen[k].replace("^", "**")))), K) == 2


@pytest.mark.parametrize("mode", ["equilibrate", "trafo"])
def test_one_inhibitor_shift_wider_than_six(mode):
    m, kw = core(mode=mode)
    r = si.detect(m, reconstruct=True, cores=2, **kw)
    gens = [s for s in r.symmetries if s.type == "general"]
    assert len(gens) == 1 and gens[0].explicit
    assert r.dim - r.rank == len(r.symmetries) == 2
    assert len(gens[0].support) <= 7 and quadratic_in(gens[0].generator, "kinh")
    # the lift moves the inhibitor alone
    assert gens[0].level == "exact" and set(gens[0].state_components) == {"S"}


@pytest.mark.parametrize("mode", ["equilibrate", "trafo"])
def test_two_inhibitors_two_shifts(mode):
    m, kw = core(two=True, mode=mode)
    r = si.detect(m, reconstruct=True, cores=2, **kw)
    gens = [s for s in r.symmetries if s.type == "general"]
    assert len(gens) == 2 and all(g.explicit for g in gens)
    assert r.dim - r.rank == len(r.symmetries) == 4
    assert r.info["attempts"] == [] and r.info["errors"] == []
    assert all(s.level == "exact" for s in r.symmetries) and r.info["rank_proof"] == "proven symmetries"


@pytest.mark.parametrize("mode", ["equilibrate", "trafo"])
def test_inhibitor_without_basal_source_is_a_scaling(mode):
    m, kw = core(mode=mode, basal=False)
    r = si.detect(m, reconstruct=True, cores=2, **kw)
    assert all(s.type == "scaling" and s.explicit for s in r.symmetries)
    assert r.dim - r.rank == len(r.symmetries) and r.info["rank_proven"] is not None


def test_an_entry_past_degree_cap_is_fitted_at_the_degree_found():
    # the entries of these directions have a degree above the default degree_cap
    m = si.Model({"x1": "v1 - k*x1/(K + x1)", "x2": "v2 - k*x2/(K + x2)"}, {"y": "s*(x1 + x2)"},
                 conditions={"rows": ["c1", "c2"], "cols": {"k": ["k1", "k2"]}})
    r = si.detect(m, reconstruct=True, equilibrate=True, cores=2)
    assert all(s.explicit for s in r.symmetries) and len(r.symmetries) == r.dim - r.rank == 4
    low = si.detect(m, reconstruct=True, equilibrate=True, cores=2,
                    control=si.reconst_control(degree_cap_max=4, homotopy=False))
    assert not all(s.explicit for s in low.symmetries)


@pytest.mark.parametrize("reconstruct", [False, True])
def test_every_direction_is_reported_once(reconstruct):
    models = [({"A": "-k1*A + k2*B", "B": "k1*A - k2*B"}, {"y": "s*A"}),
              ({"x1": "kb + kin*u - (k1 + kl)*x1", "x2": "k1*x1 - k2*x2", "u": "0"}, {"y": "s*x2"}),
              ({"x": "v*u^n/(K^n + u^n) - d*x", "u": "-e*u"}, {"y": "x"})]
    for f, g in models:
        r = si.detect(si.Model(f, g), reconstruct=reconstruct)
        assert len(r.symmetries) == r.dim - r.rank
        sup = [tuple(s.support) for s in r.symmetries if not s.explicit]
        assert len(sup) == len(set(sup))

"""Outputs that are not analytic in the coordinates at the start: a ratio whose numerator and
denominator both vanish at t = 0, and a free power of a state that starts at 0."""

import numpy as np
import pytest
import sympy as spy

import symident as si

P = 2**31 - 1


def det(f, g, **kw):
    kw.setdefault("cores", 1)
    model = si.Model(f, g, trafo=kw.pop("trafo", None))
    return si.detect(model, **kw)


def supports(res):
    return sorted(tuple(sorted(d["support"])) for d in res.symmetries)


# ---- an independent evaluation: truncated series in t with gradients, by Picard steps ----

class _Ser:
    """Series sum_j c[j] t^(lo + j) mod P; every c[j] holds the value and the gradient."""

    def __init__(self, lo, c):
        self.lo, self.c = lo, c


def _const(v, d, n):
    c = np.zeros((n, d + 1), dtype=object)
    c[0, 0] = v % P
    return _Ser(0, c)


def _add(a, b, n):
    lo = min(a.lo, b.lo)
    c = np.zeros((n, a.c.shape[1]), dtype=object)
    for s in (a, b):
        for j in range(n):
            if s.lo + j - lo < n:
                c[s.lo + j - lo] += s.c[j]
    return _Ser(lo, c % P)


def _dmul(x, y):
    out = np.zeros_like(x)
    out[0] = x[0] * y[0]
    out[1:] = x[0] * y[1:] + y[0] * x[1:]
    return out % P


def _mul(a, b, n):
    c = np.zeros((n, a.c.shape[1]), dtype=object)
    for i in range(n):
        for j in range(n - i):
            c[i + j] += _dmul(a.c[i], b.c[j])
    return _Ser(a.lo + b.lo, c % P)


def _inv(a, n):
    # the first order with a value leads; below it the gradient must vanish
    v = next(j for j in range(n) if a.c[j, 0] % P)
    assert not np.any(a.c[:v] % P)
    u = a.c[v:]
    m = len(u)
    b = np.zeros((n, a.c.shape[1]), dtype=object)
    i0 = pow(int(u[0, 0]), P - 2, P)
    b[0, 0] = i0
    b[0, 1:] = (-u[0, 1:] * i0 * i0) % P
    for k in range(1, min(n, m)):
        s = np.zeros(a.c.shape[1], dtype=object)
        for j in range(1, k + 1):
            s += _dmul(u[j], b[k - j])
        b[k] = _dmul(b[0], (-s) % P)
    return _Ser(-(a.lo + v), b % P)


def _eval(e, env, d, n, memo):
    if e in memo:
        return memo[e]
    if e.is_Symbol:
        r = env[str(e)]
    elif e.is_Integer:
        r = _const(int(e), d, n)
    elif e.is_Rational:
        r = _const(int(e.p) * pow(int(e.q), P - 2, P), d, n)
    elif e.is_Add:
        r = _eval(e.args[0], env, d, n, memo)
        for t in e.args[1:]:
            r = _add(r, _eval(t, env, d, n, memo), n)
    elif e.is_Mul:
        r = _eval(e.args[0], env, d, n, memo)
        for t in e.args[1:]:
            r = _mul(r, _eval(t, env, d, n, memo), n)
    elif e.is_Pow and e.exp.is_Integer:
        b = _eval(e.base, env, d, n, memo)
        r = _const(1, d, n)
        for _ in range(abs(int(e.exp))):
            r = _mul(r, b, n)
        if e.exp < 0:
            r = _inv(r, n)
    else:
        raise ValueError(e)
    memo[e] = r
    return r


def series_rank(f, g, ic, params, N, seed=1):
    """Rank of the gradients of the Taylor coefficients 0..N of the outputs, each output computed
    from the state series by Laurent division, at a random point modulo P."""
    rng = np.random.default_rng(seed)
    d = len(params)
    vals = {p: int(rng.integers(2, 10**6)) for p in params}
    n = 2 * N + 6
    env = {}
    for i, p in enumerate(params):
        c = np.zeros((n, d + 1), dtype=object)
        c[0, 0] = vals[p]
        c[0, 1 + i] = 1
        env[p] = _Ser(0, c)
    F = {x: spy.sympify(e.replace("^", "**")) for x, e in f.items()}
    X = {x: np.zeros((n, d + 1), dtype=object) for x in f}
    for x in f:
        X[x][0, 0] = ic[x]
    for k in range(n - 1):
        env.update({x: _Ser(0, X[x]) for x in f})
        memo = {}
        for x, e in F.items():
            fx = _eval(e, env, d, n, memo)
            assert fx.lo >= 0
            if k - fx.lo >= 0:
                X[x][k + 1] = fx.c[k - fx.lo] * pow(k + 1, P - 2, P) % P
    env.update({x: _Ser(0, X[x]) for x in f})
    rows = []
    for e in g.values():
        y = _eval(spy.sympify(e.replace("^", "**")), env, d, n, {})
        z = next(j for j in range(n) if np.any(y.c[j] % P))
        y = _Ser(y.lo + z, y.c[z:])
        assert y.lo >= 0
        rows += [list(y.c[k - y.lo, 1:]) for k in range(y.lo, N + 1)]
    return _rank_mod(rows)


def _rank_mod(rows):
    A = [[int(x) % P for x in r] for r in rows]
    rank, col = 0, 0
    ncol = len(A[0]) if A else 0
    while rank < len(A) and col < ncol:
        piv = next((i for i in range(rank, len(A)) if A[i][col]), None)
        if piv is None:
            col += 1
            continue
        A[rank], A[piv] = A[piv], A[rank]
        inv = pow(A[rank][col], P - 2, P)
        A[rank] = [x * inv % P for x in A[rank]]
        for i in range(len(A)):
            if i != rank and A[i][col]:
                fct = A[i][col]
                A[i] = [(x - fct * y) % P for x, y in zip(A[i], A[rank])]
        rank += 1
        col += 1
    return rank


# ---- 0/0 at the start ----

F_RATIO = {"x": "p1 - k*x", "z": "p2 - m*z"}
G_RATIO = {"y": "x/(x + z)"}


def test_ratio_of_states_at_zero():
    res = det(F_RATIO, G_RATIO, trafo={"x": "0", "z": "0"})
    assert (res.rank, res.dim) == (3, 4)
    assert supports(res) == [("p1", "p2")]
    assert res.info["lie_certified"]
    # the independent series of the same output, after cancelling the common power of t
    assert series_rank(F_RATIO, G_RATIO, {"x": 0, "z": 0}, ["k", "m", "p1", "p2"], 8) == 3


def test_ratio_translation_order_proven():
    # a direction that is no scaling: the order is proven by the rank of the rows at the start,
    # where the rows along the states have poles
    f = {"x": "a + b - k*x", "z": "1 - m*z"}
    g = {"y": "x/(x + z)"}
    res = det(f, g, trafo={"x": "0", "z": "0"})
    assert (res.rank, res.dim) == (3, 4)
    assert supports(res) == [("a", "b")]
    assert res.info["lie_certified"]
    assert series_rank(f, g, {"x": 0, "z": 0}, ["a", "b", "k", "m"], 8) == 3


def test_ratio_higher_valuation():
    # numerator and denominator vanish to second order at the start
    f = {"u": "p", "x": "u - k*x", "z": "q*u - m*z"}
    g = {"y": "x/(x + z)", "w": "u"}
    res = det(f, g, trafo={"u": "0", "x": "0", "z": "0"})
    ref = series_rank(f, g, {"u": 0, "x": 0, "z": 0}, ["k", "m", "p", "q"], 10)
    assert res.rank == ref == 4


def test_ratio_matches_shifted_start():
    # the same output with free initial values: the zero start loses the initial values only
    free = det(F_RATIO, G_RATIO)
    zero = det(F_RATIO, G_RATIO, trafo={"x": "0", "z": "0"})
    assert free.dim - zero.dim == 2
    assert free.rank - zero.rank == 2


# ---- free powers of a state that starts at 0 ----

F_POW = {"x": "p - m*x"}


@pytest.mark.parametrize("obs, rank, dim, sup", [
    ("x^a", 3, 3, []),
    ("q*x^a + x", 4, 4, []),
    ("x^a/(g^a + x^a)", 3, 4, [("g", "p")]),
])
def test_power_of_state_at_zero(obs, rank, dim, sup):
    res = det(F_POW, {"y": obs}, trafo={"x": "0"})
    assert (res.rank, res.dim) == (rank, dim)
    assert supports(res) == sup


def test_power_feeds_back():
    # the power enters the dynamics of another state and of the base
    f = {"x": "p - m*x + c*x^a", "z": "x^a - k*z"}
    res = det(f, {"y": "z"}, trafo={"x": "0", "z": "0"})
    assert (res.rank, res.dim) == (5, 5)


def test_power_with_a_second_state():
    res = det({"x": "p - m*x", "z": "x^a/(g^a + x^a) - k*z"}, {"y": "z"}, trafo={"x": "0", "z": "0"})
    assert (res.rank, res.dim) == (4, 5)
    assert supports(res) == [("g", "p")]


def test_power_order_not_proven():
    res = det(F_POW, {"y": "x^a/(g^a + x^a)"}, trafo={"x": "0"})
    assert not res.info["lie_certified"]
    assert any("starts at 0" in s for s in res.info["limits"])


def test_power_refusals():
    # two states at 0 with one exponent
    with pytest.raises(ValueError, match="starts at 0 and has powers|not independent"):
        det({"x": "p - m*x", "z": "q - m*z"}, {"y": "x^a + z^a"}, trafo={"x": "0", "z": "0"})

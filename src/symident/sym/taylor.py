"""Taylor coefficients of observables along the solution from a point, modulo a prime, by
truncated power series arithmetic on the expression trees, without symbolic Lie derivatives."""

import math

import sympy as spy


class NoSeriesError(ValueError):
    """An operation without a series modulo p: a non-integer power, a function, a pole."""


def _inv(a, p):
    if a % p == 0:
        raise NoSeriesError("a pole at the point")
    return pow(a % p, p - 2, p)


def _mul(a, b, n, p):
    out = [0] * n
    for i, x in enumerate(a[:n]):
        if x:
            for j in range(min(len(b), n - i)):
                out[i + j] = (out[i + j] + x * b[j]) % p
    return out


def _recip(b, n, p):
    out = [0] * n
    b0 = _inv(b[0], p)
    out[0] = b0
    for k in range(1, n):
        acc = 0
        for j in range(1, min(k, len(b) - 1) + 1):
            acc += b[j] * out[k - j]
        out[k] = (-acc * b0) % p
    return out


def _root(a, q, n, p):
    """The q-th root of the series a from the smaller square or the unique odd root of a[0], then q A B' = A' B."""
    from ..gfp import sqrt_mod
    a0 = a[0] % p
    if q == 2:
        r = sqrt_mod(a0, p)
    elif q % 2 and math.gcd(q, p - 1) == 1:
        r = pow(a0, pow(q, -1, p - 1), p)
    else:
        r = None
    if not r:
        raise NoSeriesError("no root of the constant term")
    b = [0] * n
    b[0] = r
    den = _inv(q * a0, p)
    for m in range(1, n):
        acc = 0
        for i in range(1, min(m, len(a) - 1) + 1):
            acc += a[i] * b[m - i] * (i - q * (m - i))
        b[m] = acc % p * _inv(m, p) % p * den % p
    return b


def series_of(expr, env, n, p):
    """The first n coefficients mod p of `expr`, `env` mapping symbols to series (lists) or values."""
    return _eval(spy.sympify(expr), env, n, p, {})


def _const(c, n, p):
    out = [0] * n
    out[0] = c % p
    return out


def _eval(e, env, n, p, memo):
    got = memo.get(e)
    if got is not None:
        return got
    if e.is_Symbol:
        if e not in env:
            raise NoSeriesError(f"no value for {e}")
        v = env[e]
        out = list(v[:n]) + [0] * (n - len(v)) if isinstance(v, list) else _const(v, n, p)
    elif e.is_Rational:
        out = _const(int(e.p) * _inv(int(e.q), p), n, p)
    elif e.is_Add:
        out = [0] * n
        for a in e.args:
            out = [(x + y) % p for x, y in zip(out, _eval(a, env, n, p, memo))]
    elif e.is_Mul:
        out = _const(1, n, p)
        for a in e.args:
            out = _mul(out, _eval(a, env, n, p, memo), n, p)
    elif e.is_Pow and e.exp.is_Rational:
        k = int(e.exp.p)
        base = _eval(e.base, env, n, p, memo)
        if e.exp.q != 1:
            base = _root(base, int(e.exp.q), n, p)
        if k < 0:
            base, k = _recip(base, n, p), -k
        out = _const(1, n, p)
        while k:
            if k & 1:
                out = _mul(out, base, n, p)
            base = _mul(base, base, n, p)
            k >>= 1
    else:
        raise NoSeriesError(f"{type(e).__name__} has no series modulo p")
    memo[e] = out
    return out


def output_series(f, observables, point, order, p):
    """Per observable the first `order` coefficients L_f^k h(x0)/k! mod p of x' = f(x) from `point`."""
    states = list(f)
    x = {s: _const(point[s], order, p) for s in states}
    for k in range(order - 1):
        env = dict(point)
        env.update({s: x[s][:k + 1] for s in states})
        memo = {}
        for s in states:
            c = _eval(f[s], env, k + 1, p, memo)[k]
            x[s][k + 1] = c * _inv(k + 1, p) % p
    env = dict(point)
    env.update(x)
    memo = {}
    return [_eval(spy.sympify(h), env, order, p, memo) for h in observables]

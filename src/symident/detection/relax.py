"""The relaxed model of discrete(): every non-rational term of the states as a new state that starts at a number,
times or plus a new coordinate, the value of the term at the start (docs/method/global.rst, "Rational form of the
fiber")."""

import random

import sympy as spy

from ..sym.domain import is_positive
from ..sym.parse import _RPrinter, model_sympify
from .startpow import leading, power_exponent


def _str(e):
    return _RPrinter().doprint(e).replace("**", "^")


def _non_rational(a):
    return isinstance(a, (spy.exp, spy.log)) or (a.is_Pow and not a.exp.is_Integer)


def _innermost(exprs):
    """The non-rational subterms whose own arguments are rational, in a fixed order."""
    out = {}
    for e in exprs:
        for a in spy.postorder_traversal(e):
            if _non_rational(a) and not any(_non_rational(b) for b in spy.preorder_traversal(a) if b is not a):
                out.setdefault(a, None)
    return sorted(out, key=spy.default_sort_key)


def _fresh(stem, taken):
    k = 1
    while f"{stem}{k}" in taken:
        k += 1
    taken.add(f"{stem}{k}")
    return spy.Symbol(f"{stem}{k}")


def _monotone_in(w, c):
    """Whether the value `w`, b^mu, exp(u) or log(b), is strictly monotone in the positive coordinate c with the
    other coordinates held: mu free of c and b = c^(+-1) times a factor free of c, or u = a*c^(+-1) + d with a, d
    free of c."""
    if isinstance(w, spy.log):
        w = w.args[0] ** spy.Integer(2)
    if isinstance(w, spy.exp):
        u = spy.expand(w.args[0])
        for v in (c, 1 / c):
            a = u.coeff(v)
            if a != 0 and not a.has(c) and not (u - a * v).has(c):
                return True
        return False
    if not w.is_Pow or w.exp.has(c):
        return False
    rest = spy.cancel(w.base / c)
    if not rest.has(c):
        return True
    return not spy.cancel(w.base * c).has(c)


def _same_leaf(leaves, value):
    """W^k of the leaf whose value is `value` or, for an exponential, an integer power k of it; else None."""
    for lf in leaves:
        v, W = lf["value"], spy.Symbol(lf["name"])
        if isinstance(v, spy.exp) and isinstance(value, spy.exp):
            k = spy.cancel(value.args[0] / v.args[0])
            if k.is_Integer:
                return W**k
        elif v == value:
            return W
    return None


def _log_states(f, g, pos):
    """{x: (R, k)} of the positive states x that the model reads only through log(x), through the factor x of
    f_x = x*R(log x, ...) and through observables x^k with an integer k: L = log x then solves L' = R."""
    out = {}
    for x, e in f.items():
        if pos is not True and str(x) not in pos:
            continue
        L = spy.Dummy("L")
        R = spy.cancel(e / x).replace(lambda a: isinstance(a, spy.log) and a.args[0] == x, lambda a: L)
        if R.has(x):
            continue
        rest = [
            v.replace(lambda a: isinstance(a, spy.log) and a.args[0] == x, lambda a: L) for z, v in f.items() if z != x
        ] + [
            v.replace(lambda a: isinstance(a, spy.log) and a.args[0] == x, lambda a: L)
            for v in g.values()
            if not (v.is_Pow and v.base == x and v.exp.is_Integer) and v != x
        ]
        if any(v.has(x) for v in rest):
            continue
        out[x] = R
    return out


def _start_power(x, al, B, f, g, tr, starts, states, ins, leaves, rewrites, taken, positive):
    """(f, g, tr) with B = x^alpha of a state x at 0 as Z N, N = (x/c_x)^alpha and Z = c_x^alpha, c_x the leading
    coefficient of x. Z goes where it cancels against a leaf c_x^(q alpha); else on the slice Z = 1 of the scaling
    of Z and the leaves b^(q alpha) that keeps the model, each such leaf then (b/c_x)^(q alpha); else Z is a leaf
    (docs/method/global.rst, "Rational form of the fiber")."""
    f0 = {str(s): v.xreplace({B: spy.Integer(0)}) for s, v in f.items()}
    ic = {str(s): starts.get(s, s) for s in f}
    # a known input is a chain of jets with free starts: the leading coefficient must not read them
    jsyms = set()
    for u in sorted(ins, key=str):
        names = [str(_fresh(f"si_j{u}_", taken)) for _ in range(12)]
        f0[str(u)] = spy.Symbol(names[0])
        for a_, b_ in zip([str(u)] + names, names + [None]):
            if b_ is None:
                f0[a_] = spy.Integer(0)
            ic[a_] = _fresh(f"si_j{a_}0_", taken)
            jsyms.add(ic[a_])
        for a_, b_ in zip(names, names[1:]):
            f0[a_] = spy.Symbol(b_)
    k, c = leading(f0, ic, x, al)
    if c.free_symbols & jsyms:
        raise ValueError(f"the leading coefficient {_str(c)} of {x} reads a known input")
    if positive is None or not is_positive(c, positive):
        raise ValueError(f"the leading coefficient {_str(c)} of {x}, which starts at 0, is not positive on the domain")
    Zs, Nn, lam = spy.Dummy("Z"), spy.Dummy("N"), spy.Dummy("lam", positive=True)

    def power_of(lf):
        # (b, q) of a leaf b^(q*alpha) of the coordinates
        v = lf["value"]
        if lf["kind"] != "constant" or not v.is_Pow:
            return None
        q = spy.cancel(v.exp / al)
        return (v.base, int(q)) if q.is_Integer and q != 0 else None

    sub = {B: Zs * Nn}
    for lf in leaves:
        bq = power_of(lf)
        if bq is not None and (spy.cancel(bq[0] - c) == 0 or spy.cancel(bq[0] * c - 1) == 0):
            sub[spy.Symbol(lf["name"])] = Zs ** (bq[1] if spy.cancel(bq[0] - c) == 0 else -bq[1])
    exprs = {
        **{("f", s): v for s, v in f.items()},
        **{("g", y): v for y, v in g.items()},
        **{("t", s): v for s, v in tr.items() if s in states},
    }

    vals = {k_: v.xreplace(sub) for k_, v in exprs.items()}
    # Z cancels in an expression where the cancelled form is free of it
    vals = {k_: spy.cancel(v) if v.has(Zs) and not spy.cancel(v).has(Zs) else v for k_, v in vals.items()}
    moved = {}
    if any(v.has(Zs) for v in vals.values()):
        # the scaling Z -> lam Z, b^(q alpha) -> lam^q b^(q alpha) of the leaves left
        for lf in leaves:
            bq = power_of(lf)
            if bq is not None and spy.Symbol(lf["name"]) not in sub:
                moved[spy.Symbol(lf["name"])] = (lf, bq)
        scale = {Zs: lam * Zs, **{P: lam ** bq[1] * P for P, (_, bq) in moved.items()}}
        if moved and all(spy.cancel(v.xreplace(scale) - v) == 0 for v in vals.values()):
            for lf, (b, q) in moved.values():
                lf["value"] = spy.Pow(spy.cancel(b / c), q * al)
            vals = {k_: v.xreplace({Zs: spy.Integer(1)}) for k_, v in vals.items()}
        else:
            moved = {}
            W = str(_fresh("si_W", taken))
            leaves.append({"name": W, "kind": "power", "value": spy.Pow(c, al), "state": None})
            vals = {k_: v.xreplace({Zs: spy.Symbol(W)}) for k_, v in vals.items()}
    N = x**al * spy.Pow(c, -al)
    vals = {k_: v.xreplace({Nn: N}) for k_, v in vals.items()}
    # leaves that are powers of c_x or have the value 1 are gone
    gone = {P for P in sub if P != B}
    for lf in leaves:
        if lf["kind"] == "constant" and lf["value"] == 1:
            gone.add(spy.Symbol(lf["name"]))
            vals = {k_: v.xreplace({spy.Symbol(lf["name"]): spy.Integer(1)}) for k_, v in vals.items()}
    leaves[:] = [lf for lf in leaves if spy.Symbol(lf["name"]) not in gone]
    f = {s: vals[("f", s)] for s in f}
    g = {y: vals[("g", y)] for y in g}
    tr = {s: vals.get(("t", s), v) for s, v in tr.items()}
    if moved or any(
        lf["kind"] == "power" and lf.get("state") is None and lf["value"] == spy.Pow(c, al) for lf in leaves
    ):
        rewrites.append(
            {
                "from": _str(x**al),
                "to": _str(N),
                "state": None,
                "rate": None,
                "start": "0",
                "order": k,
                "leading": _str(c),
            }
        )
    return f, g, tr


def terms_of_states(odes, observables):
    """Whether a non-rational term of the model reads a state."""
    states = {spy.Symbol(x) for x in odes}
    exprs = [model_sympify(str(v)) for v in list(odes.values()) + list(dict(observables).values())]
    return any(a.free_symbols & states for e in exprs for a in spy.preorder_traversal(e) if _non_rational(a))


def relax(odes, observables, trafo, coords, inputs=(), positive=True):
    """The relaxed model as {"odes", "observables", "trafo", "leaves", "dropped", "positive", "rewrites"}.

    A term phi(psi) of the states (exp, log, a power with a non-integer exponent) is W*nu for exp and powers, W + nu
    for log, with nu a new state that starts at 1 (0 for log), so that nu = exp(psi - psi0), (b/b0)^mu or
    log(psi/psi0), and W = phi(psi0) a new coordinate. A term of the coordinates alone is a new coordinate P. A
    coordinate that no equation reads afterwards and that one P or W determines on the positive reals is dropped. A
    known input inside a term becomes a state with a known start and a new known input as rate. A ValueError names
    a term that cannot be rewritten."""
    taken = {str(s) for s in coords} | set(odes) | set(observables) | set(inputs)
    f = {spy.Symbol(x): model_sympify(v) for x, v in odes.items()}
    g = {y: model_sympify(v) for y, v in observables.items()}
    tr = {spy.Symbol(k): model_sympify(v) for k, v in (trafo or {}).items()}
    for e in list(f.values()) + list(g.values()) + list(tr.values()):
        taken |= {str(s) for s in e.free_symbols}
    states = set(f)
    ins = {spy.Symbol(u) for u in inputs}
    time = spy.Symbol("time")
    starts = {x: tr.get(x, x) for x in f}
    params = {s: v for s, v in tr.items() if s not in states}
    leaves, rewrites = [], []
    new_inputs, known = [], []
    cset, zpow = {spy.Symbol(c) for c in coords}, {}
    # a positive state read only through log(x) and the factor x is L = log x with a rational rate
    pos = True if positive is True else {str(p) for p in (positive or ())}
    logx = _log_states(f, g, pos) if positive is not None else {}
    if logx:
        Ls = {x: _fresh("si_L", taken) for x in logx}
        to_L = {spy.log(x): L for x, L in Ls.items()}
        for x, L in Ls.items():
            P = _fresh("si_P", taken)
            leaves.append({"name": str(P), "kind": "constant", "value": spy.log(starts[x].xreplace(params))})
            rewrites.append(
                {
                    "from": _str(x),
                    "to": f"exp({L})",
                    "state": str(L),
                    "rate": _str(logx[x].xreplace({d: Ls[x] for d in logx[x].atoms(spy.Dummy)})),
                    "start": str(P),
                }
            )
            tr[L] = starts[L] = P
            tr.pop(x, None)
            starts.pop(x)
        f = {
            Ls.get(x, x): (logx[x].xreplace({d: Ls[x] for d in logx[x].atoms(spy.Dummy)}) if x in logx else v)
            for x, v in f.items()
        }
        f = {x: v.xreplace(to_L) for x, v in f.items()}
        # an observable x^k is k*log(x), injective on the positive reals
        g = {
            y: (v.exp * Ls[v.base] if v.is_Pow and v.base in Ls else Ls[v] if v in Ls else v.xreplace(to_L))
            for y, v in g.items()
        }
        states = set(f)
    while True:
        exprs = list(f.values()) + list(g.values())
        atoms = _innermost(exprs)
        if not atoms:
            break
        a = atoms[0]
        if a.is_Pow and not a.exp.is_Integer and a.exp.free_symbols & (states | ins):
            # a state in the exponent: b^e = exp(e*log(b))
            new = spy.exp(a.exp * spy.log(a.base))
            f = {x: v.xreplace({a: new}) for x, v in f.items()}
            g = {y: v.xreplace({a: new}) for y, v in g.items()}
            continue
        pe = power_exponent(a.exp, cset - states) if a.is_Pow and a.base in states - ins else None
        if pe is not None and starts.get(a.base) == 0:
            # a free power of a state at 0: B = x^alpha in place, Z = c_x^alpha after the loop
            key = (a.base, pe[2])
            if key not in zpow:
                zpow[key] = _fresh("si_B", taken)
            new = a.base ** pe[0] * zpow[key] ** pe[1]
            f = {x: v.xreplace({a: new}) for x, v in f.items()}
            g = {y: v.xreplace({a: new}) for y, v in g.items()}
            continue
        if a.is_Pow and a.exp.is_Rational:
            raise ValueError(
                f"a root or rational power, {_str(a)}, has an algebraic start: the relaxation needs a "
                "transcendental term"
            )
        reads = a.free_symbols
        for u in sorted(reads & ins, key=str):
            # a known input u inside a term is a state with a known start and the known input du as rate
            du, u0 = _fresh(f"si_d{u}_", taken), _fresh(f"si_{u}0_", taken)
            f[u], tr[u], starts[u] = du, u0, u0
            states.add(u)
            ins.discard(u)
            new_inputs.append(str(du))
            known.append(str(u0))
        if time in reads:
            raise ValueError(f"{_str(a)} reads the time")
        if not reads & states:
            # a term that equals the start of a leaf is that leaf
            new = _same_leaf(leaves, a.xreplace(params))
            if new is None:
                new = _fresh("si_P", taken)
                leaves.append({"name": str(new), "kind": "constant", "value": a.xreplace(params)})
        else:
            arg = a.args[0] if not a.is_Pow else a.base
            dpsi = sum((spy.diff(arg, x) * f[x] for x in states & arg.free_symbols), spy.Integer(0))
            nu = _fresh("si_nu", taken)
            W = _fresh("si_W", taken)
            psi0 = arg.xreplace(starts).xreplace(params)
            if isinstance(a, spy.exp):
                f[nu], kind, w0 = nu * dpsi, "exp", spy.exp(psi0)
                new = W * nu
            elif isinstance(a, spy.log):
                f[nu], kind, w0 = dpsi / arg, "log", spy.log(psi0)
                new = W + nu
            else:
                if psi0 == 0:
                    raise ValueError(f"the base of {_str(a)} starts at 0")
                f[nu], kind, w0 = a.exp * nu * dpsi / arg, "power", psi0 ** a.exp.xreplace(params)
                new = W * nu
            tr[nu] = spy.Integer(0) if kind == "log" else spy.Integer(1)
            starts[nu] = tr[nu]
            states.add(nu)
            same = _same_leaf(leaves, w0)
            if w0.is_number and w0.is_Rational:
                new = new.xreplace({W: w0})
            elif same is not None:
                new = new.xreplace({W: same})
            else:
                leaves.append({"name": str(W), "kind": kind, "value": w0, "state": str(nu)})
            rewrites.append(
                {"from": _str(a), "to": _str(new), "state": str(nu), "rate": _str(f[nu]), "start": _str(tr[nu])}
            )
        f = {x: v.xreplace({a: new}) for x, v in f.items()}
        g = {y: v.xreplace({a: new}) for y, v in g.items()}
    for (x, al), B in zpow.items():
        f, g, tr = _start_power(x, al, B, f, g, tr, starts, states, ins, leaves, rewrites, taken, positive)
    # a start that holds a term of the coordinates, a resting state, reads the leaf of that term
    for x in [x for x in tr if x in states]:
        while True:
            at = _innermost([tr[x]])
            if not at:
                break
            a = at[0]
            if a.free_symbols & states:
                raise ValueError(f"the start {_str(tr[x])} of {x} reads a state")
            new = _same_leaf(leaves, a.xreplace(params))
            if new is None:
                new = _fresh("si_P", taken)
                leaves.append({"name": str(new), "kind": "constant", "value": a.xreplace(params)})
            tr[x] = tr[x].xreplace({a: new})
    for e in list(f.values()) + list(g.values()):
        bad = next((a for a in spy.preorder_traversal(e) if a.is_Function), None)
        if bad is not None:
            raise ValueError(f"{_str(bad)} is no exponential, logarithm or power")
    # the starts of the leaves in the coordinates of the model: a nested term reads the start of an inner leaf
    back = {}
    for lf in leaves:
        lf["value"] = lf["value"].xreplace(back)
        back[spy.Symbol(lf["name"])] = lf["value"]
    used = set()
    for e in list(f.values()) + list(g.values()) + list(tr.values()):
        used |= {str(s) for s in e.free_symbols}
    used |= {str(x) for x in f if x not in tr}
    dropped = {}
    for c in coords:
        if c in used:
            continue
        cs = spy.Symbol(c)
        hit = [
            lf
            for lf in leaves
            if lf["kind"] != "log"
            and cs in lf["value"].free_symbols
            and lf["name"] not in dropped.values()
            and _monotone_in(lf["value"], cs)
        ]
        if not hit:
            raise ValueError(f"{c} enters only through non-rational terms of the coordinates that do not determine it")
        dropped[c] = hit[0]["name"]
    for lf in leaves:
        lf["absorbs"] = next((c for c, P in dropped.items() if P == lf["name"]), None)
    if zpow and not leaves and not rewrites:
        raise ValueError(
            "the free powers of states at 0 enter as (x/c_x)^alpha, which the count takes without relaxation"
        )
    pos = None
    if positive is not None:
        dec = set(coords) if positive is True else {str(p) for p in positive}
        # a logarithm of the coordinates is signed
        pos = sorted(
            (dec - set(dropped))
            | {lf["name"] for lf in leaves if lf["kind"] != "log" and not isinstance(lf["value"], spy.log)}
            | {str(x) for x in f if str(x).startswith("si_nu") and tr.get(x) == 1}
        )
        pos = [p for p in pos if p in used or p in {str(x) for x in f}]
    return {
        "odes": {str(x): _str(v) for x, v in f.items()},
        "observables": {y: _str(v) for y, v in g.items()},
        "trafo": {str(k): _str(v) for k, v in tr.items()},
        "leaves": leaves,
        "dropped": dropped,
        "positive": pos,
        "rewrites": rewrites,
        "inputs": [u for u in inputs if spy.Symbol(u) in ins] + new_inputs,
        "known": known,
    }


def scalings_keep_leaves(symmetries, leaves, dropped):
    """None if each orbit of the continuous directions of the relaxed model meets the starts of the leaves in a
    connected set, else the reason: every direction that moves a leaf W is exact, a scaling or a constant shift of the
    leaves it moves, and moves no coordinate the starts read, the weights or shifts on these leaves form an
    invertible matrix, and no direction moves a coordinate read by the start of a leaf that none of them moves."""

    def reads(lfs):
        out = set()
        for lf in lfs:
            out |= {str(s) for s in lf["value"].free_symbols}
        # a coordinate replaced by a constant term P is a coordinate of the relaxed model
        return out - set(dropped)

    names = {lf["name"] for lf in leaves if lf["absorbs"] is None}
    every = reads(leaves)
    rows, moved = [], set()
    for s in symmetries:
        sup = set(s.get("support") or [])
        if not sup & names:
            continue
        gen = s.get("generator") or {}
        if s.get("level") != "exact":
            return f"the direction on {', '.join(sorted(sup))} that moves a new coordinate is not exact"
        if s.get("type") == "scaling":
            rows.append(s.get("weights") or {})
        elif all(model_sympify(str(gen.get(n, 0))).is_number for n in sup & names):
            # a translation of the leaves it moves, with a log leaf in place of the log of a scaled one
            rows.append({n: gen.get(n, 0) for n in sup & names})
        else:
            return f"a direction on {', '.join(sorted(sup))} moves a new coordinate neither by a scaling nor by a shift"
        if sup & every:
            return f"a direction moves {', '.join(sorted(sup & every))}, read by the starts of the new states"
        moved |= sup & names
    cols = sorted(moved)
    if rows and (
        len(rows) != len(cols)
        or spy.Matrix([[model_sympify(str(w.get(n, 0))) for n in cols] for w in rows]).rank() < len(cols)
    ):
        return "the weights or shifts on the new coordinates they move are no invertible matrix"
    still = reads([lf for lf in leaves if lf["absorbs"] is None and lf["name"] not in moved])
    for s in symmetries:
        hit = set(s.get("support") or []) & still
        if hit:
            return f"a direction moves {', '.join(sorted(hit))}, read by the start of a new coordinate no scaling moves"
    return None


def keeps_leaves(sigma, leaves, dropped):
    """Whether a map sigma of the relaxed coordinates is a map of the model: on the starts of the leaves, the image of
    each leaf W is the start of W at the image of the coordinates, sigma(W)(omega) = omega_W(sigma(theta))."""
    sub = {spy.Symbol(k): model_sympify(v) for k, v in sigma.items()}
    on = {spy.Symbol(lf["name"]): lf["value"] for lf in leaves if lf["absorbs"] is None}
    absorbed = {spy.Symbol(c): spy.Symbol(P) for c, P in dropped.items()}
    img = {s: e.xreplace(on) for s, e in sub.items() if s not in on}
    for lf in leaves:
        if lf["absorbs"] is not None:
            continue
        # a coordinate the start reads through a constant term P is moved where P is
        if any(P in sub for c, P in absorbed.items() if c in lf["value"].free_symbols):
            return False
        W = spy.Symbol(lf["name"])
        lhs = sub.get(W, W).xreplace(on)
        rhs = lf["value"].xreplace(img)
        if spy.simplify(spy.powsimp(lhs - rhs, force=True)) != 0:
            return False
    return True


def independent_leaves(leaves, coords, seed=0):
    """Whether the Jacobian of the exponents of the leaves (psi0 of exp(psi0), mu*log(b0) of b0^mu, and log(psi0))
    in the coordinates has full rank at a random rational point, the logarithms of the bases read as new symbols."""
    rng = random.Random(seed)
    cs = [spy.Symbol(c) for c in coords]
    lam = {}
    rows = []
    for lf in leaves:
        if lf["absorbs"] is not None:
            continue
        w = lf["value"]
        if isinstance(w, spy.exp):
            u = w.args[0]
        elif isinstance(w, spy.log):
            u = w
        elif w.is_Pow:
            L = lam.setdefault(w.base, spy.Dummy("L"))
            u = w.exp * L
        else:
            return False
        rows.append(u)
    if not rows:
        return True

    def d(u, c):
        out = spy.diff(u, c)
        for b, L in lam.items():
            out += spy.diff(u, L) * spy.diff(b, c) / b
        return out

    pt = {c: spy.Rational(rng.randint(2, 10**6), rng.randint(2, 10**6)) for c in cs}
    pt.update({L: spy.Rational(rng.randint(2, 10**6), rng.randint(2, 10**6)) for L in lam.values()})
    J = spy.Matrix(
        [
            [
                d(u, c).xreplace(pt)
                if not isinstance(u, spy.log)
                else (spy.diff(u.args[0], c) / u.args[0]).xreplace(pt)
                for c in cs
            ]
            for u in rows
        ]
    )
    return J.rank() == len(rows)

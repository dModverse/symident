"""Closed forms of narrow directions from jets of the observables."""

import random

import sympy as spy

from .parse import _as_list, _make_local_parse


def jet_generator(conds, support, anchor, max_order=8, max_chars=200000,
                 time_limit=60.0):
    """A direction with support S in closed form from the jets.

    Every Lie derivative L_f^k h evaluated at the initial values is invariant under
    the direction X, so X_S is orthogonal to the S-gradient of every jet. With
    |S| - 1 jets whose S-gradients are independent, X_S is their generalised cross
    product: X_i = (-1)^i det(M without column i), M the gradient rows. For |S| = 2
    this is X = (d_b Phi, -d_a Phi), the Hamiltonian field of one jet Phi.

    `conds` holds per condition {'f': ['x = rhs', ...], 'g': [h, ...],
    'ic': ['x = x0', ...]} over the leaves, or a chain {'segments': [...]} whose later
    segments start from their boundary events ('ev') instead of initial values: at order 0
    in every gap length the state does not move across a gap, so a later segment
    starts from the earlier initial values with its events applied, and its jets
    there are invariant like any other (each gap monomial of a row annihilates X).
    A segment whose boundary time depends on the coordinates ends the chain. The
    first segments are tried alone first (fewer, smaller jets); the later ones join
    only when those fall short of the rank.
    Jets are taken order by order, lowest first, rows kept greedily while they raise
    the rank at a random rational point. Returns {'ok': True, 'vector': {z: entry}}
    normalised to 1 at `anchor`, or {'ok': False, 'reason': ...}. The caller
    verifies the result."""
    import time as _time
    t0 = _time.time()
    primary, extra = _jet_expand_chains(conds)
    res = _jet_core(primary, support, anchor, max_order, max_chars, time_limit, t0)
    if not res['ok'] and extra and str(res.get('reason', '')).startswith('rank'):
        res = _jet_core(primary + extra, support, anchor, max_order, max_chars,
                        time_limit, t0)
    return res


def _jet_core(conds, support, anchor, max_order, max_chars, time_limit, t0):
    """jet_generator() on a fixed list of jet conditions, within time_limit from t0."""
    import time as _time
    support = [str(s) for s in _as_list(support)]
    anchor = str(anchor)
    m = len(support)
    if m < 2 or anchor not in support:
        return {'ok': False, 'reason': 'support'}
    lines = []
    for c in conds:
        lines += list(_as_list(c['f'])) + list(_as_list(c['ic'])) + \
            [str(x) for x in _as_list(c['g'])]
    lines += support
    local, parse = _make_local_parse(lines)
    zs = [local.get(s, spy.Symbol(s)) for s in support]
    rng = random.Random(20260927)
    point = {}

    def num(e):
        for sym in e.free_symbols:
            if sym not in point:
                point[sym] = spy.Integer(rng.randint(2, 997))
        return e.xreplace(point)

    parsed = []
    for c in conds:
        fl = [l.split(' = ', 1) for l in _as_list(c['f'])]
        states = [local.get(a.strip(), spy.Symbol(a.strip())) for a, _ in fl]
        rhs = [spy.sympify(parse(b)) for _, b in fl]
        icd = {}
        for l in _as_list(c['ic']):
            a, b = l.split(' = ', 1)
            icd[local.get(a.strip(), spy.Symbol(a.strip()))] = spy.sympify(parse(b))
        gs = [spy.sympify(parse(str(x))) for x in _as_list(c['g'])]
        parsed.append((states, rhs, icd, gs))
    rows, numrows = [], []
    rank = 0
    jets = [list(p[3]) for p in parsed]
    for order in range(max_order + 1):
        for ci, (_states, _rhs, icd, _gs) in enumerate(parsed):
            for h in jets[ci]:
                if _time.time() - t0 > time_limit:
                    return {'ok': False, 'reason': 'time limit'}
                he = h.xreplace(icd) if icd else h
                row = [spy.diff(he, z) for z in zs]
                if all(r == 0 for r in row):
                    continue
                # a minor of rows this large does not simplify in bounded time
                if sum(len(str(r)) for r in row) > max_chars // 20:
                    continue
                nr = [num(r) for r in row]
                if spy.Matrix(numrows + [nr]).rank() > rank:
                    rows.append(row)
                    numrows.append(nr)
                    rank += 1
                    if rank == m - 1:
                        break
            if rank == m - 1:
                break
        if rank == m - 1:
            break
        if order == max_order:
            break
        for ci, (states, rhs, _icd, _gs) in enumerate(parsed):
            new = []
            for h in jets[ci]:
                d = sum((spy.diff(h, states[i]) * rhs[i] for i in range(len(states))),
                        spy.Integer(0))
                if len(str(d)) > max_chars:
                    return {'ok': False, 'reason': 'jet too large'}
                new.append(d)
            jets[ci] = new
    if rank < m - 1:
        return {'ok': False, 'reason': 'rank %d of %d within order %d' % (rank, m - 1, max_order)}
    M = spy.Matrix(rows)
    comps = []
    for i in range(m):
        sub = M[:, [j for j in range(m) if j != i]]
        # the minor by cofactor expansion, kept as a product/sum tree (no expansion)
        comps.append((-1) ** i * (sub.det(method='berkowitz') if m > 2 else sub[0, 0]))
        if _time.time() - t0 > time_limit:
            return {'ok': False, 'reason': 'time limit'}
    ia = support.index(anchor)
    if comps[ia] == 0:
        return {'ok': False, 'reason': 'anchor component vanishes'}
    vec = {}
    for i in range(m):
        e = spy.together(comps[i] / comps[ia])
        # cancel() and factor() only while they stay cheap; the caller verifies the
        # entry modulo a fresh prime either way
        if len(str(e)) <= max_chars // 40:
            e = spy.factor(spy.cancel(e))
        if len(str(e)) > max_chars:
            return {'ok': False, 'reason': 'closed form too large'}
        if e != 0:
            vec[support[i]] = str(e)
        if _time.time() - t0 > time_limit:
            return {'ok': False, 'reason': 'time limit'}
    return {'ok': True, 'vector': vec, 'order': int(order)}


def _jet_expand_chains(conds):
    """Chains of segments as independent jet conditions at gap order 0: segment k starts
    from segment k-1's initial values with its own boundary events applied. Returns the
    first segments and the later ones as two lists."""
    out, extra = [], []
    for c in _as_list(conds):
        c = dict(c)
        if 'segments' not in c:
            out.append(c)
            continue
        segs = [dict(sg) for sg in _as_list(c['segments'])]
        if not segs:
            continue
        ic = [str(l) for l in _as_list(segs[0]['ic'])]
        out.append({'f': segs[0]['f'], 'g': segs[0]['g'], 'ic': ic})
        for sg in segs[1:]:
            if not sg.get('time_fixed', True):
                break
            icd = dict(l.split(' = ', 1) for l in ic)
            for ev in _as_list(sg.get('ev', [])):
                ev = dict(ev)
                X, v, how = str(ev['var']), str(ev['value']), str(ev['method'])
                old = icd.get(X, X)
                icd[X] = ('(%s)' % v if how == 'replace' else
                          '(%s) + (%s)' % (old, v) if how == 'add' else
                          '(%s)*(%s)' % (old, v))
            ic = ['%s = %s' % (X, e) for X, e in icd.items()]
            extra.append({'f': sg['f'], 'g': sg['g'], 'ic': ic})
    return out, extra

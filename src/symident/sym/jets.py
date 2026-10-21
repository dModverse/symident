"""Closed forms of narrow directions from jets of the observables."""

import sympy as spy

from .. import session
from ..rexpr import replace_symbols
from .parse import _as_list, _make_local_parse


def jet_generator(conds, support, anchor, max_order=8, max_chars=200000, work_limit=None):
    """{'ok': True, 'vector'} with support S from |S| - 1 independent jet gradients, else {'ok': False}."""
    work = [0, 50 * max_chars if work_limit is None else int(work_limit)]
    primary, extra = _jet_expand_chains(conds)
    res = _jet_core(primary, support, anchor, max_order, max_chars, work)
    if not res['ok'] and extra and str(res.get('reason', '')).startswith('rank'):
        res = _jet_core(primary + extra, support, anchor, max_order, max_chars, work)
    return res


def _jet_core(conds, support, anchor, max_order, max_chars, work):
    """jet_generator() on a fixed list of jet conditions; `work` = [done, limit] is shared between calls."""
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
    rng = session.current().rng("jets.point")
    point = {}

    def num(e):
        for sym in sorted(e.free_symbols, key=str):
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
                if work[0] > work[1]:
                    return {'ok': False, 'reason': 'work limit'}
                he = h.xreplace(icd) if icd else h
                row = [spy.diff(he, z) for z in zs]
                if all(r == 0 for r in row):
                    continue
                size = sum(len(str(r)) for r in row)
                work[0] += size
                # a minor of rows this large does not simplify in bounded time
                if size > max_chars // 20:
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
                size = len(str(d))
                work[0] += size
                if size > max_chars:
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
    return {'ok': True, 'vector': vec, 'order': int(order)}


def _jet_expand_chains(conds):
    """Chains of segments as jet conditions at gap order 0: the first and the later segments as two lists."""
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
            pre = dict(icd)
            for ev in _as_list(sg.get('ev', [])):
                ev = dict(ev)
                X, v, how = str(ev['var']), str(ev['value']), str(ev['method'])
                # at gap order 0 a state in the value is its start value before the events
                v = replace_symbols(list(pre), ['(%s)' % e for e in pre.values()], v)
                old = icd.get(X, X)
                icd[X] = ('(%s)' % v if how == 'replace' else
                          '(%s) + (%s)' % (old, v) if how == 'add' else
                          '(%s)*(%s)' % (old, v))
            ic = ['%s = %s' % (X, e) for X, e in icd.items()]
            extra.append({'f': sg['f'], 'g': sg['g'], 'ic': ic})
    return out, extra

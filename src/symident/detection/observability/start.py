"""Diagonal start: free parameters Z whose zero makes every resting condition coincide with the forward one."""

import sympy as spy

from ...sym.parse import _clean, _make_local_parse


def _rhs(model):
    _, parse = _make_local_parse(list(model))
    out = {}
    for raw in model:
        line = _clean(raw)
        if "=" not in line:
            continue
        lhs, rhs = line.split("=", 1)
        out[lhs.strip()] = parse(rhs)
    return out


def _zero(expr):
    return spy.expand(spy.fraction(spy.together(expr))[0]) == 0


class DiagonalStart:
    """Z and the invariant zero set S(Z) of the conditions `reps`, or `ok = False`."""

    def __init__(self, models, reps, fwd, held_of, candidates, keep=(), cap=12):
        self.reps, self.fwd, self.held_of = list(reps), fwd, held_of
        self.keep = set(keep)
        self.rhs = {ci: _rhs(models[ci]) for ci in dict.fromkeys(self.reps + [fwd])}
        self.cand = set(candidates)
        base = {str(s) for s in self._symbols(self.fwd)}
        priv = [str(s) for ci in self.reps for s in self._symbols(ci)
                if str(s) in self.cand and str(s) not in base]
        Z = [z for z in dict.fromkeys(priv) if self._regular(Z=[z])]
        self.ok = False
        for _ in range(cap + 1):
            S = self._closure(Z)
            diff = self._differences(Z, S)
            if not diff:
                # without Z the zero set would be a boundary rest of the actual point
                self.Z, self.S, self.ok = sorted(Z), S, bool(Z) or not S
                self.dead = self._dead(Z, S)
                self.dens = self._denominators()
                return
            pool = sorted({str(s) for d in diff for s in d.free_symbols} & self.cand - set(Z))
            best = None
            for c in pool:
                if not self._regular(Z + [c]):
                    continue
                n = len(self._differences(Z + [c], self._closure(Z + [c])))
                if best is None or n < best[0]:
                    best = (n, c)
            if best is None or best[0] >= len(diff):
                return
            Z = Z + [best[1]]

    def _denominators(self):
        """The distinct denominators of every condition, as polynomials."""
        out = {}
        for ci in self.rhs:
            for e in self.rhs[ci].values():
                den = spy.fraction(spy.together(e))[1]
                for f in spy.Mul.make_args(spy.factor(den)):
                    b = f.as_base_exp()[0]
                    if b.free_symbols:
                        out[str(b)] = spy.Poly(b, *sorted(b.free_symbols, key=str))
        return list(out.values())

    def pole(self, values, p):
        """Whether a denominator vanishes mod p at `values` (name to residue)."""
        for d in self.dens:
            gens = [int(values.get(str(g), 0)) % p for g in d.gens]
            acc = 0
            for mon, c in d.terms():
                t = int(c) % p
                for v, e in zip(gens, mon):
                    t = t * pow(v, e, p) % p
                acc = (acc + t) % p
            if acc == 0:
                return True
        return False

    def _dead(self, Z, S):
        """Parameters of the forward condition that drop out at Z = 0 and S = 0, not to be solved for."""
        sub = {spy.Symbol(n): 0 for n in list(Z) + list(S) + list(self.held_of(self.fwd))}
        f0 = self.rhs[self.fwd]
        before = set().union(*(e.free_symbols for e in f0.values()))
        after = set().union(*(spy.cancel(spy.together(e.xreplace(sub))).free_symbols
                              for k, e in f0.items() if k not in S))
        return sorted(str(x) for x in before - after if str(x) in self.cand and str(x) not in Z)

    def _symbols(self, ci):
        return set().union(*(e.free_symbols for e in self.rhs[ci].values()))

    def _regular(self, Z):
        """No denominator of a condition vanishes at Z = 0."""
        sub = {spy.Symbol(z): 0 for z in Z}
        for ci in self.rhs:
            for e in self.rhs[ci].values():
                den = spy.fraction(spy.together(e))[1]
                if den != 1 and spy.expand(den.xreplace(sub)) == 0:
                    return False
        return True

    def _closure(self, Z):
        """The largest set of states whose zero is invariant in every condition at Z = 0."""
        S = set(self.rhs[self.fwd])
        for ci in self.rhs:
            S &= set(self.rhs[ci])
        S -= set.intersection(*(set(self.held_of(ci)) for ci in self.rhs)) | self.keep
        while True:
            drop = set()
            for ci in self.rhs:
                held = set(self.held_of(ci))
                sub = {spy.Symbol(n): 0 for n in list(Z) + sorted(S | held)}
                drop |= {s for s in S - held if not _zero(self.rhs[ci][s].xreplace(sub))}
            if not drop:
                return sorted(S)
            S -= drop

    def _differences(self, Z, S):
        """Numerators of the balances that differ from the forward condition at Z = 0, S = 0."""
        f0 = self.rhs[self.fwd]
        out = []
        for ci in self.reps:
            if ci == self.fwd:
                continue
            zero = set(self.held_of(ci)) | set(self.held_of(self.fwd))
            sub = {spy.Symbol(n): 0 for n in list(Z) + list(S) + sorted(zero)}
            for s, e in self.rhs[ci].items():
                if s in S or s in zero or s not in f0:
                    continue
                d = (e - f0[s]).xreplace(sub)
                if not _zero(d):
                    out.append(spy.expand(spy.fraction(spy.together(d))[0]))
            for s in set(self.held_of(self.fwd)) - set(self.held_of(ci)) - set(S):
                out.append(spy.Symbol(s))
        return out

"""Degrees of the Taylor coefficients of the observables in the coordinates, from the recursion of the series run
on degree bounds instead of values (docs/method/global.rst, "The verified count")."""

from fractions import Fraction


class Atoms:
    """The factors of the denominators, each with a degree bound; one factor per object divided by."""

    def __init__(self):
        self.deg, self._divisors = [], {}

    def divisor(self, x):
        if id(x) not in self._divisors:
            self.deg.append(x.dn)
            self._divisors[id(x)] = (len(self.deg) - 1, x)
        return self._divisors[id(x)][0]

    def of(self, den):
        """The degree of the product of the factors with the exponents in `den`."""
        return sum(e * self.deg[a] for a, e in den.items())


class Deg:
    """A rational function of the coordinates by degree bounds only: the degree of the numerator and the
    denominator as a product of factors of Atoms; a number is kept exactly, zero only where exactly zero."""

    __slots__ = ("A", "dn", "den", "c")

    def __init__(self, A, dn=0, den=None, c=None):
        self.A, self.dn, self.den, self.c = A, dn, dict(den or {}), c

    @classmethod
    def const(cls, A, x):
        return cls(A, 0, None, Fraction(x))

    def _of(self, x):
        return x if isinstance(x, Deg) else Deg.const(self.A, x)

    def __add__(self, x):
        x = self._of(x)
        if self.c is not None and x.c is not None:
            return Deg.const(self.A, self.c + x.c)
        if self.c == 0:
            return x
        if x.c == 0:
            return self
        L = {a: max(self.den.get(a, 0), x.den.get(a, 0)) for a in set(self.den) | set(x.den)}
        da = self.A.of({a: e - self.den.get(a, 0) for a, e in L.items()})
        db = self.A.of({a: e - x.den.get(a, 0) for a, e in L.items()})
        return Deg(self.A, max(self.dn + da, x.dn + db), L)

    __radd__ = __add__

    def __neg__(self):
        return Deg(self.A, self.dn, self.den, None if self.c is None else -self.c)

    def __sub__(self, x):
        return self + (-self._of(x))

    def __rsub__(self, x):
        return self._of(x) - self

    def __mul__(self, x):
        x = self._of(x)
        if self.c is not None and x.c is not None:
            return Deg.const(self.A, self.c * x.c)
        if self.c == 0 or x.c == 0:
            return Deg.const(self.A, 0)
        den = dict(self.den)
        for a, e in x.den.items():
            den[a] = den.get(a, 0) + e
        return Deg(self.A, self.dn + x.dn, den)

    __rmul__ = __mul__

    def __truediv__(self, x):
        x = self._of(x)
        if x.c is not None:
            if x.c == 0:
                raise ZeroDivisionError("division by zero")
            return self * Deg.const(self.A, 1 / x.c)
        # 1/x: the factors of the denominator of x over the numerator of x, a new factor
        return self * Deg(self.A, self.A.of(x.den), {self.A.divisor(x): 1})

    def __rtruediv__(self, x):
        return self._of(x) / self

    def __pow__(self, e):
        out = Deg.const(self.A, 1)
        for _ in range(int(e)):
            out = out * self
        return out

    def __eq__(self, x):
        if isinstance(x, (int, Fraction)) and x == 0:
            return self.c == 0
        return NotImplemented

    __hash__ = None  # type: ignore[assignment]


def common(values):
    """(deg q, [deg of each numerator over q]): the lcm q of the denominators of `values` (Deg)."""
    A = values[0].A if values else Atoms()
    L = {}
    for v in values:
        for a, e in v.den.items():
            L[a] = max(L.get(a, 0), e)
    return A.of(L), [None if v.c == 0 else v.dn + A.of({a: e - v.den.get(a, 0) for a, e in L.items()}) for v in values]

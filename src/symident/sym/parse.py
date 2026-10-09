"""Parsing of model lines into sympy expressions with a clean symbol table."""

import io
import tokenize

import sympy as spy
from sympy.parsing.sympy_parser import parse_expr

# primes below 1518500213, msolve's bound for FGLM
_PRIMES = [1518500183, 1518500173, 1518500171, 1518500143,
           1518500141, 1518500131, 1518500101, 1518500077]


def _as_list(x):
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    return list(x)


def _clean(line):
    return line.replace('"', '').replace('^', '**').strip()


def _build_symbol_table(all_lines):
    """Map every non-function identifier to a sympy Symbol, so names like
    E, I, S, O, Q, N are taken as model variables, not sympy constants."""
    syms, funcs = set(), set()
    for raw in all_lines:
        line = _clean(raw)
        if not line:
            continue
        try:
            toks = list(tokenize.generate_tokens(io.StringIO(line).readline))
        except (tokenize.TokenError, IndentationError):
            toks = []
        for i, t in enumerate(toks):
            if t.type == tokenize.NAME:
                nxt = toks[i + 1] if i + 1 < len(toks) else None
                if nxt is not None and nxt.type == tokenize.OP and nxt.string == '(':
                    funcs.add(t.string)
                else:
                    syms.add(t.string)
    syms -= funcs
    return {s: spy.Symbol(s) for s in syms}


def _function_aliases():
    """Functions of the model language that sympy does not provide by name; mapped to
    differentiable sympy expressions so their gradients are rational."""
    x = spy.Symbol('_x_')
    return {
        'log10': spy.Lambda(x, spy.log(x) / spy.log(10)),
        'log2': spy.Lambda(x, spy.log(x) / spy.log(2)),
        'exp10': spy.Lambda(x, spy.Integer(10) ** x),
    }


def _make_parse(local_dict):
    # memoised per string: conditions and segments repeat the same lines
    memo = {}

    def parse(rhs):
        key = str(rhs)
        e = memo.get(key)
        if e is None:
            e = parse_expr(_clean(rhs), local_dict=local_dict, evaluate=True)
            memo[key] = e
        return e
    return parse


def _make_local_parse(all_lines):
    """Symbol table (model names as symbols, not sympy constants) with the
    function aliases, plus a parser bound to it."""
    local = _build_symbol_table(all_lines)
    local.update(_function_aliases())
    return local, _make_parse(local)


def _read_equations(lines, parse):
    variables, functions = [], []
    for raw in lines:
        line = _clean(raw)
        if '=' not in line:
            continue
        lhs, rhs = line.split('=', 1)
        variables.append(spy.Symbol(lhs.strip()))
        functions.append(parse(rhs))
    allsyms = set()
    for f in functions:
        allsyms |= set(spy.sympify(f).free_symbols)
    parameters = sorted(allsyms - set(variables), key=spy.default_sort_key)
    return variables, functions, parameters


# raised when an expression is not a rational function of the coordinates
class _NotRationalError(Exception):
    pass


def _is_rational_expr(e):
    """True if e is a rational function of its symbols (no transcendental
    function, no non-integer power)."""
    e = spy.sympify(e)
    for a in spy.preorder_traversal(e):
        if a.is_Function:
            return False
        if a.is_Pow and not a.exp.is_Integer:
            return False
    return True

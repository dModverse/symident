"""Parsing of model lines into sympy expressions with a clean symbol table."""

import io
import tokenize

import sympy as spy
from sympy.parsing.sympy_parser import parse_expr
from sympy.printing.str import StrPrinter

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
    """Map every non-function identifier but pi to a sympy Symbol, so E, I, S, N are model variables."""
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
    syms -= funcs | {'pi'}
    return {s: spy.Symbol(s) for s in syms}


def _function_aliases():
    """Functions of the model language that sympy lacks by name or defines otherwise, as sympy expressions."""
    x = spy.Symbol('_x_')
    return {
        'log10': spy.Lambda(x, spy.log(x) / spy.log(10)),
        'log2': spy.Lambda(x, spy.log(x) / spy.log(2)),
        'exp10': spy.Lambda(x, spy.Integer(10) ** x),
        'exp2': spy.Lambda(x, spy.Integer(2) ** x),
        'ln': spy.log,
        'acot': spy.Lambda(x, spy.atan(1 / x)),
        'asec': spy.Lambda(x, spy.acos(1 / x)),
        'acsc': spy.Lambda(x, spy.asin(1 / x)),
        'acoth': spy.Lambda(x, spy.atanh(1 / x)),
        'asech': spy.Lambda(x, spy.acosh(1 / x)),
        'acsch': spy.Lambda(x, spy.asinh(1 / x)),
        'round': spy.Lambda(x, spy.floor(x + spy.Rational(1, 2))),
    }


# the functions of the model language that are not switches, by name
_MODEL_FUNCTIONS = {
    'exp': spy.exp, 'log': spy.log, 'sqrt': spy.sqrt, 'cbrt': spy.cbrt, 'root': spy.root, 'pow': spy.Pow,
    'sin': spy.sin, 'cos': spy.cos, 'tan': spy.tan, 'cot': spy.cot, 'sec': spy.sec, 'csc': spy.csc,
    'asin': spy.asin, 'acos': spy.acos, 'atan': spy.atan, 'atan2': spy.atan2,
    'sinh': spy.sinh, 'cosh': spy.cosh, 'tanh': spy.tanh, 'coth': spy.coth, 'sech': spy.sech,
    'csch': spy.csch, 'asinh': spy.asinh, 'acosh': spy.acosh, 'atanh': spy.atanh,
    'erf': spy.erf, 'erfc': spy.erfc, 'floor': spy.floor, 'ceiling': spy.ceiling,
}


def model_function(name):
    """The sympy callable of a function of the model language, None for an unknown name."""
    return _function_aliases().get(name) or _MODEL_FUNCTIONS.get(name)


class _RPrinter(StrPrinter):
    """Prints Euler's number as exp(1)."""

    def _print_Exp1(self, expr):
        return "exp(1)"


def to_r(e):
    """A sympy expression as a model string, `^` for powers and Euler's number as exp(1); a string gets only `^`."""
    if isinstance(e, str):
        return e.replace("**", "^")
    return _RPrinter().doprint(spy.sympify(e)).replace("**", "^")


def _make_parse(local_dict):
    # memoized per string: conditions and segments repeat the same lines
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
    """Symbol table with model names as symbols and the function aliases, plus a parser bound to it."""
    local = _build_symbol_table(all_lines)
    local.update(_function_aliases())
    return local, _make_parse(local)


def model_sympify(text):
    """`text` (R syntax) as a SymPy expression with every name but pi a Symbol, also S, N, E, I or beta."""
    _, parse = _make_local_parse([str(text)])
    return spy.sympify(parse(str(text)))


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
    """True if e is a rational function of its symbols: no transcendental function, no non-integer power."""
    e = spy.sympify(e)
    for a in spy.preorder_traversal(e):
        if a.is_Function:
            return False
        if a.is_Pow and not a.exp.is_Integer:
            return False
    return True
